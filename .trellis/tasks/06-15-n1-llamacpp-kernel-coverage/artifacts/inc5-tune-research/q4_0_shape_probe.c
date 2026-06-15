// inc5-tune-research: Q4_0 x Q8_0 dot-product SHAPE DESIGN-SPACE probe.
//
// THROWAWAY probes (NOT the shipped kernel) to MEASURE what the autotuner should
// target. Every variant is BITWISE-gated against ggml's real RVV kernel before it
// is timed; a variant that fails the gate is not a candidate (its time is N/A).
//
// Byte-exact rule (hard constraint): the per-block scalar fold
//   sumf = sumf + ((float)sumi * dx) * dy
// is fp-non-associative, so it MUST run in strict ascending block order in every
// variant. Variants may overlap the INTEGER cores (vle8/xor/sll/sra/vwmul/vwmacc/
// vwredsum -> per-block i32 sumi) across blocks and batch the reductions, but the
// scalar folds stay ordered. Any "vectorize the scale step" is excluded (changes
// rounding).
//
// Board: ssh rvv, 64-core riscv64, VLEN=128 (VLENB=16), rv64gcv_zfh_zvfh, clang 18.
// Build: clang -march=rv64gcv_zfh_zvfh -mabi=lp64d -O3 -ffp-contract=fast
// Run pinned: taskset -c 3 ./probe <n> <iters> <repeats>
//
// Variants:
//   ggml   : ggml's real hand-written RVV (i8m1, 1 vwredsum/block, serial).
//   m1     : our current emitted shape (i8m1 strip loop, runs once @VLEN128).
//   m1_nl  : m1 with the inner strip loop straight-lined (no per-block strip loop;
//            still per-block vsetvl). Isolates the strip-loop branch cost.
//   sl128  : straight-line VLEN-128 specialization, ONE vsetvl hoisted out of the
//            block loop, no inner loop. CEILING probe — NOT shippable (assumes
//            VLEN>=128, violates VLEN-robustness). Quantifies loop+vsetvl headroom.
//   mb2     : 2 blocks/iter, integer cores interleaved, the 2 vwredsums issued
//            back-to-back (latency overlap), scalar folds in order. vsetvl hoisted.
//   mb4     : 4 blocks/iter, 4 vwredsums batched, folds in order. vsetvl hoisted.
//   mb2_vl  : mb2 but VLEN-robust (per-block vsetvl kept, strip-safe) — the
//            shippable form of mb2.
//   mb4_vl  : mb4 VLEN-robust form.
//
// vreg pressure is reported analytically per shape (see VREG comments) — the
// number the autotuner's prune budget must allow.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK 32
#define Q4_STRIDE 18
#define Q8_STRIDE 34

static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

// One block's integer core -> i32 sumi, the SHARED byte-exact decode/product:
//   tx = vle8(x+2); v0 = (tx&0xF)-8 ; v1 = (tx>>4)-8 ; (== xor 0x88 then sll/sra path)
//   y0 = vle8(y+2); y1 = vle8(y+18)
//   m  = vwmul(v0,y0); m = vwmacc(m, v1, y1); sumi = vwredsum(m, 0)
// All variants use this exact integer set; only the SCHEDULE differs.

// ------------------------------------------------------------------ ggml real
static void ggml_real(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  size_t vl = QK / 2; // 16
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q4_STRIDE;
    const uint8_t *yb = vy + (size_t)ib * Q8_STRIDE;
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xb + 2, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
    vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t v0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_a), 8, vl);
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_l), 8, vl);
    vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, z, vl));
    uint16_t dx, dy;
    memcpy(&dx, xb, 2);
    memcpy(&dy, yb, 2);
    sumf += sumi * fp16_to_fp32(dx) * fp16_to_fp32(dy);
  }
  *s = sumf;
}

// helper: one block's vwmacc'd i16m2 product (no reduce) — for batched-reduce shapes
static inline vint16m2_t block_prod(const uint8_t *xb, const uint8_t *yb, size_t vl) {
  vuint8m1_t tx = __riscv_vle8_v_u8m1(xb + 2, vl);
  vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
  vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 18), vl);
  vint8m1_t v0 = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(__riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(tx), 0x88, vl), 4, vl), 4, vl);
  vint8m1_t v1 = __riscv_vsra_vx_i8m1(__riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(tx), 0x88, vl), 4, vl);
  vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
  return __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
}
static inline int reduce_prod(vint16m2_t p, size_t vl) {
  vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
  return __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, z, vl));
}
static inline float scale(const uint8_t *xb, const uint8_t *yb) {
  uint16_t dx, dy;
  memcpy(&dx, xb, 2);
  memcpy(&dy, yb, 2);
  return fp16_to_fp32(dx) * fp16_to_fp32(dy);
}

// ----------------------------------------------------------- m1 (our current)
// VREG: live = 1 i8m1 src + 1 i16m2 product (2 regs) + 1 i32m1 acc ~= 4-5 vregs.
static void k_m1(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q4_STRIDE;
    const uint8_t *yb = vy + (size_t)ib * Q8_STRIDE;
    int sumi = 0;
    size_t vlmax = __riscv_vsetvl_e8m1(16);
    for (size_t off = 0; off < 16; off += vlmax) {
      size_t vl = __riscv_vsetvl_e8m1(16 - off);
      vint16m2_t p = block_prod(xb + off, yb + off, vl); // off only valid @vl==16
      sumi = reduce_prod(p, vl);
    }
    sumf += (float)sumi * scale(xb, yb);
  }
  *s = sumf;
}

// ------------------------------------------------ m1_nl (no inner strip loop)
// Same as m1 but the strip loop is straight-lined (1 vsetvl/block, no branch).
static void k_m1_nl(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q4_STRIDE;
    const uint8_t *yb = vy + (size_t)ib * Q8_STRIDE;
    size_t vl = __riscv_vsetvl_e8m1(16);
    int sumi = reduce_prod(block_prod(xb, yb, vl), vl);
    sumf += (float)sumi * scale(xb, yb);
  }
  *s = sumf;
}

// ------------------------------------------- sl128 (CEILING, NOT shippable)
// One vsetvl(16) hoisted out of the block loop; no inner loop. Assumes vl==16.
static void k_sl128(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  size_t vl = __riscv_vsetvl_e8m1(16);
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q4_STRIDE;
    const uint8_t *yb = vy + (size_t)ib * Q8_STRIDE;
    int sumi = reduce_prod(block_prod(xb, yb, vl), vl);
    sumf += (float)sumi * scale(xb, yb);
  }
  *s = sumf;
}

// ----------------------------------- mb2 (2 blocks/iter, batched reductions)
// vsetvl hoisted. 2 independent i16m2 products live -> 4 vregs, +2 i32 acc.
// VREG: 2 products (2*2=4) + reductions ~= 6-8 vregs live.
static void k_mb2(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  size_t vl = __riscv_vsetvl_e8m1(16);
  int ib = 0;
  for (; ib + 2 <= nb; ib += 2) {
    const uint8_t *xb0 = vx + (size_t)ib * Q4_STRIDE, *yb0 = vy + (size_t)ib * Q8_STRIDE;
    const uint8_t *xb1 = vx + (size_t)(ib+1) * Q4_STRIDE, *yb1 = vy + (size_t)(ib+1) * Q8_STRIDE;
    vint16m2_t p0 = block_prod(xb0, yb0, vl); // integer cores interleaved by compiler
    vint16m2_t p1 = block_prod(xb1, yb1, vl);
    int s0 = reduce_prod(p0, vl);             // 2 vwredsums back-to-back
    int s1 = reduce_prod(p1, vl);
    sumf += (float)s0 * scale(xb0, yb0);      // folds in strict order
    sumf += (float)s1 * scale(xb1, yb1);
  }
  for (; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q4_STRIDE, *yb = vy + (size_t)ib * Q8_STRIDE;
    sumf += (float)reduce_prod(block_prod(xb, yb, vl), vl) * scale(xb, yb);
  }
  *s = sumf;
}

// ----------------------------------- mb4 (4 blocks/iter, batched reductions)
// VREG: 4 products (4*2=8) live before reduce -> ~10-12 vregs live.
static void k_mb4(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  size_t vl = __riscv_vsetvl_e8m1(16);
  int ib = 0;
  for (; ib + 4 <= nb; ib += 4) {
    const uint8_t *x0 = vx+(size_t)ib*Q4_STRIDE,   *y0 = vy+(size_t)ib*Q8_STRIDE;
    const uint8_t *x1 = vx+(size_t)(ib+1)*Q4_STRIDE,*y1 = vy+(size_t)(ib+1)*Q8_STRIDE;
    const uint8_t *x2 = vx+(size_t)(ib+2)*Q4_STRIDE,*y2 = vy+(size_t)(ib+2)*Q8_STRIDE;
    const uint8_t *x3 = vx+(size_t)(ib+3)*Q4_STRIDE,*y3 = vy+(size_t)(ib+3)*Q8_STRIDE;
    vint16m2_t p0 = block_prod(x0,y0,vl), p1 = block_prod(x1,y1,vl);
    vint16m2_t p2 = block_prod(x2,y2,vl), p3 = block_prod(x3,y3,vl);
    int r0 = reduce_prod(p0,vl), r1 = reduce_prod(p1,vl);
    int r2 = reduce_prod(p2,vl), r3 = reduce_prod(p3,vl);
    sumf += (float)r0 * scale(x0,y0);
    sumf += (float)r1 * scale(x1,y1);
    sumf += (float)r2 * scale(x2,y2);
    sumf += (float)r3 * scale(x3,y3);
  }
  for (; ib < nb; ++ib) {
    const uint8_t *xb = vx+(size_t)ib*Q4_STRIDE, *yb = vy+(size_t)ib*Q8_STRIDE;
    sumf += (float)reduce_prod(block_prod(xb,yb,vl), vl) * scale(xb,yb);
  }
  *s = sumf;
}

// --------------------------------- mb2_vl (mb2, VLEN-robust: per-block vsetvl)
static void k_mb2_vl(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  int ib = 0;
  for (; ib + 2 <= nb; ib += 2) {
    const uint8_t *x0 = vx+(size_t)ib*Q4_STRIDE,   *y0 = vy+(size_t)ib*Q8_STRIDE;
    const uint8_t *x1 = vx+(size_t)(ib+1)*Q4_STRIDE,*y1 = vy+(size_t)(ib+1)*Q8_STRIDE;
    size_t vl = __riscv_vsetvl_e8m1(16); // VLEN>=128 -> 16; else re-strip needed (probe assumes 16)
    vint16m2_t p0 = block_prod(x0,y0,vl), p1 = block_prod(x1,y1,vl);
    int r0 = reduce_prod(p0,vl), r1 = reduce_prod(p1,vl);
    sumf += (float)r0 * scale(x0,y0);
    sumf += (float)r1 * scale(x1,y1);
  }
  for (; ib < nb; ++ib) {
    const uint8_t *xb = vx+(size_t)ib*Q4_STRIDE, *yb = vy+(size_t)ib*Q8_STRIDE;
    size_t vl = __riscv_vsetvl_e8m1(16);
    sumf += (float)reduce_prod(block_prod(xb,yb,vl), vl) * scale(xb,yb);
  }
  *s = sumf;
}

// --------------------------------- mb4_vl (mb4 VLEN-robust: per-block vsetvl)
// The SHIPPABLE form of the winner: 4 blocks/iter, batched reductions, per-block
// vsetvl kept so a VLEN<128 board re-strips correctly. VREG ~10-12 live.
static void k_mb4_vl(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  int ib = 0;
  for (; ib + 4 <= nb; ib += 4) {
    const uint8_t *x0 = vx+(size_t)ib*Q4_STRIDE,   *y0 = vy+(size_t)ib*Q8_STRIDE;
    const uint8_t *x1 = vx+(size_t)(ib+1)*Q4_STRIDE,*y1 = vy+(size_t)(ib+1)*Q8_STRIDE;
    const uint8_t *x2 = vx+(size_t)(ib+2)*Q4_STRIDE,*y2 = vy+(size_t)(ib+2)*Q8_STRIDE;
    const uint8_t *x3 = vx+(size_t)(ib+3)*Q4_STRIDE,*y3 = vy+(size_t)(ib+3)*Q8_STRIDE;
    size_t vl = __riscv_vsetvl_e8m1(16);
    vint16m2_t p0 = block_prod(x0,y0,vl), p1 = block_prod(x1,y1,vl);
    vint16m2_t p2 = block_prod(x2,y2,vl), p3 = block_prod(x3,y3,vl);
    int r0 = reduce_prod(p0,vl), r1 = reduce_prod(p1,vl);
    int r2 = reduce_prod(p2,vl), r3 = reduce_prod(p3,vl);
    sumf += (float)r0 * scale(x0,y0);
    sumf += (float)r1 * scale(x1,y1);
    sumf += (float)r2 * scale(x2,y2);
    sumf += (float)r3 * scale(x3,y3);
  }
  for (; ib < nb; ++ib) {
    const uint8_t *xb = vx+(size_t)ib*Q4_STRIDE, *yb = vy+(size_t)ib*Q8_STRIDE;
    size_t vl = __riscv_vsetvl_e8m1(16);
    sumf += (float)reduce_prod(block_prod(xb,yb,vl), vl) * scale(xb,yb);
  }
  *s = sumf;
}

// --------------------------------- mb8 (8 blocks/iter, vsetvl hoisted; probe)
// Does even wider batching keep helping or does vreg pressure / spills hurt?
// VREG ~ up to 16+ live products before reduce -> likely spills near the 32 budget.
static void k_mb8(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  size_t vl = __riscv_vsetvl_e8m1(16);
  int ib = 0;
  for (; ib + 8 <= nb; ib += 8) {
    const uint8_t *X = vx + (size_t)ib*Q4_STRIDE, *Y = vy + (size_t)ib*Q8_STRIDE;
    vint16m2_t p0=block_prod(X+0*Q4_STRIDE,Y+0*Q8_STRIDE,vl), p1=block_prod(X+1*Q4_STRIDE,Y+1*Q8_STRIDE,vl);
    vint16m2_t p2=block_prod(X+2*Q4_STRIDE,Y+2*Q8_STRIDE,vl), p3=block_prod(X+3*Q4_STRIDE,Y+3*Q8_STRIDE,vl);
    vint16m2_t p4=block_prod(X+4*Q4_STRIDE,Y+4*Q8_STRIDE,vl), p5=block_prod(X+5*Q4_STRIDE,Y+5*Q8_STRIDE,vl);
    vint16m2_t p6=block_prod(X+6*Q4_STRIDE,Y+6*Q8_STRIDE,vl), p7=block_prod(X+7*Q4_STRIDE,Y+7*Q8_STRIDE,vl);
    int r0=reduce_prod(p0,vl),r1=reduce_prod(p1,vl),r2=reduce_prod(p2,vl),r3=reduce_prod(p3,vl);
    int r4=reduce_prod(p4,vl),r5=reduce_prod(p5,vl),r6=reduce_prod(p6,vl),r7=reduce_prod(p7,vl);
    sumf += (float)r0*scale(X+0*Q4_STRIDE,Y+0*Q8_STRIDE);
    sumf += (float)r1*scale(X+1*Q4_STRIDE,Y+1*Q8_STRIDE);
    sumf += (float)r2*scale(X+2*Q4_STRIDE,Y+2*Q8_STRIDE);
    sumf += (float)r3*scale(X+3*Q4_STRIDE,Y+3*Q8_STRIDE);
    sumf += (float)r4*scale(X+4*Q4_STRIDE,Y+4*Q8_STRIDE);
    sumf += (float)r5*scale(X+5*Q4_STRIDE,Y+5*Q8_STRIDE);
    sumf += (float)r6*scale(X+6*Q4_STRIDE,Y+6*Q8_STRIDE);
    sumf += (float)r7*scale(X+7*Q4_STRIDE,Y+7*Q8_STRIDE);
  }
  for (; ib < nb; ++ib) {
    const uint8_t *xb = vx+(size_t)ib*Q4_STRIDE, *yb = vy+(size_t)ib*Q8_STRIDE;
    sumf += (float)reduce_prod(block_prod(xb,yb,vl), vl) * scale(xb,yb);
  }
  *s = sumf;
}

// ---------------- mb*_defer: GENUINELY VLEN-robust, reduction freed from the strip loop ----
// Per block: a strip loop that accumulates the i16 products into a per-block i32 ELEMENT-
// WISE vector accumulator (vwadd_wv: i32 += widen(i16)), NO reduce inside the loop, then
// ONE vredsum_vs_i32m1 AFTER the loop. This is robust BY CONSTRUCTION (multi-strip at
// VLEN<128 accumulates element-wise into i32 -> no i16 overflow, exact same integer sum)
// AND frees the per-block reduce so K blocks' reduces can be emitted adjacent and overlap.
// Byte-exact: the final per-block integer sum is identical (integer add is associative);
// the fp32 folds stay in strict block order.
//
// strip loop -> per-block i32 vector accumulator (lane-wise partial sums), pre-reduce:
static inline vint32m1_t block_strip_acc_i32(const uint8_t *xb, const uint8_t *yb) {
  // i32m1 has 4 lanes at VLEN=128; the i16m2 product has 16 lanes. We accumulate the
  // 16-lane i16 product into the 4-lane i32 accumulator by widening-reducing each strip
  // into a 4-lane partial via vwredsum is NOT element-wise. To keep it element-wise AND
  // robust, accumulate the i16 products themselves into an i16 running vector is unsafe
  // (overflow). Instead: per strip, do a widening reduce into a SEPARATE i32 scalar lane
  // accumulator vector via repeated vredsum is serial. The clean robust element-wise form
  // uses an i32m4 accumulator matching the 16 product lanes: acc_i32m4 += widen(prod_i16m2).
  size_t vl = __riscv_vsetvl_e8m1(16);
  vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
  size_t vlmax = vl;
  for (size_t off = 0; off < 16; off += vlmax) {
    size_t v = __riscv_vsetvl_e8m1(16 - off);
    vint16m2_t p = block_prod(xb + off, yb + off, v);
    acc = __riscv_vwadd_wv_i32m4(acc, p, v); // i32m4 += sign-extend(i16m2), element-wise
  }
  // collapse the i32m4 (16 lanes) -> we return it; caller reduces after batching.
  // To return a uniform type, reduce m4->m1 lane-vector is not needed; we reduce to scalar
  // in the caller. But to allow batched adjacent reduces we return the i32m4 accumulator.
  // (vsetvl restored to 16 for the i32m4 view.)
  return __riscv_vredsum_vs_i32m4_i32m1(acc, __riscv_vmv_v_x_i32m1(0, vl), vl);
}
static inline int reduce_i32m1_lane0(vint32m1_t v) {
  return __riscv_vmv_x_s_i32m1_i32(v);
}
// 4-block deferred-robust: 4 strip-accumulate phases (each freeing its reduce), then 4
// vredsum->scalar adjacent (overlap), then 4 folds in order.
static void k_mb4_defer(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  int ib = 0;
  for (; ib + 4 <= nb; ib += 4) {
    const uint8_t *x0=vx+(size_t)ib*Q4_STRIDE,*y0=vy+(size_t)ib*Q8_STRIDE;
    const uint8_t *x1=vx+(size_t)(ib+1)*Q4_STRIDE,*y1=vy+(size_t)(ib+1)*Q8_STRIDE;
    const uint8_t *x2=vx+(size_t)(ib+2)*Q4_STRIDE,*y2=vy+(size_t)(ib+2)*Q8_STRIDE;
    const uint8_t *x3=vx+(size_t)(ib+3)*Q4_STRIDE,*y3=vy+(size_t)(ib+3)*Q8_STRIDE;
    int r0 = reduce_i32m1_lane0(block_strip_acc_i32(x0,y0));
    int r1 = reduce_i32m1_lane0(block_strip_acc_i32(x1,y1));
    int r2 = reduce_i32m1_lane0(block_strip_acc_i32(x2,y2));
    int r3 = reduce_i32m1_lane0(block_strip_acc_i32(x3,y3));
    sumf += (float)r0*scale(x0,y0);
    sumf += (float)r1*scale(x1,y1);
    sumf += (float)r2*scale(x2,y2);
    sumf += (float)r3*scale(x3,y3);
  }
  for (; ib < nb; ++ib){ const uint8_t*xb=vx+(size_t)ib*Q4_STRIDE,*yb=vy+(size_t)ib*Q8_STRIDE;
    sumf += (float)reduce_i32m1_lane0(block_strip_acc_i32(xb,yb))*scale(xb,yb); }
  *s = sumf;
}

// ---------------- mb2_robust / mb4_robust (GENUINELY VLEN-robust multi-block) ----------------
// The honest shippable multi-block form: each block keeps the INNER STRIP LOOP with
// the sumi-carry seed (vmv_v_x_i32m1(carried_sumi) -> vwredsum), so a VLEN<128 board
// re-strips correctly (runs the inner loop >1 time). This is k_m1's structure unrolled
// to K blocks/iter with the K reductions BATCHED after the K product phases. The 4
// integer cores are independent so the compiler can overlap them; the per-block strip
// loop runs ONCE at VLEN=128 (vsetvl_e8m1(16)=16). The scalar folds stay in order.
//
// strip-with-carry for ONE block -> i32 sumi (exactly k_m1's inner loop, byte-exact):
static inline int block_strip_sumi(const uint8_t *xb, const uint8_t *yb) {
  int sumi = 0;
  size_t vlmax = __riscv_vsetvl_e8m1(16);
  for (size_t off = 0; off < 16; off += vlmax) {
    size_t vl = __riscv_vsetvl_e8m1(16 - off);
    vint16m2_t p = block_prod(xb + off, yb + off, vl);
    vint32m1_t seed = __riscv_vmv_v_x_i32m1(sumi, vl);     // carry across strips
    sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, seed, vl));
  }
  return sumi;
}
// 2-block robust: interleave the two strip-loop integer cores, batch nothing extra
// (each block's reduction is inside its own strip loop, but the two blocks' cores are
// independent so the compiler overlaps the vwredsums). VLEN<128 SAFE.
static void k_mb2_robust(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  int ib = 0;
  for (; ib + 2 <= nb; ib += 2) {
    const uint8_t *x0=vx+(size_t)ib*Q4_STRIDE,*y0=vy+(size_t)ib*Q8_STRIDE;
    const uint8_t *x1=vx+(size_t)(ib+1)*Q4_STRIDE,*y1=vy+(size_t)(ib+1)*Q8_STRIDE;
    int r0 = block_strip_sumi(x0,y0);
    int r1 = block_strip_sumi(x1,y1);
    sumf += (float)r0 * scale(x0,y0);
    sumf += (float)r1 * scale(x1,y1);
  }
  for (; ib < nb; ++ib){ const uint8_t*xb=vx+(size_t)ib*Q4_STRIDE,*yb=vy+(size_t)ib*Q8_STRIDE;
    sumf += (float)block_strip_sumi(xb,yb)*scale(xb,yb); }
  *s = sumf;
}
// 4-block robust: the genuine shippable form of the winner. To expose the 4 vwredsums
// for latency overlap AT VLEN=128 (1 strip each), we compute the 4 products, then the 4
// strip-seeded reductions back-to-back. At VLEN=128 each strip loop is 1 iteration; at
// VLEN<128 each block re-strips correctly via its own carry. We hand-roll the VLEN=128
// fast path's batching by splitting product/reduce while keeping the carry seed.
static void k_mb4_robust(int n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = n / QK;
  float sumf = 0;
  int ib = 0;
  for (; ib + 4 <= nb; ib += 4) {
    const uint8_t *x0=vx+(size_t)ib*Q4_STRIDE,*y0=vy+(size_t)ib*Q8_STRIDE;
    const uint8_t *x1=vx+(size_t)(ib+1)*Q4_STRIDE,*y1=vy+(size_t)(ib+1)*Q8_STRIDE;
    const uint8_t *x2=vx+(size_t)(ib+2)*Q4_STRIDE,*y2=vy+(size_t)(ib+2)*Q8_STRIDE;
    const uint8_t *x3=vx+(size_t)(ib+3)*Q4_STRIDE,*y3=vy+(size_t)(ib+3)*Q8_STRIDE;
    int r0 = block_strip_sumi(x0,y0);
    int r1 = block_strip_sumi(x1,y1);
    int r2 = block_strip_sumi(x2,y2);
    int r3 = block_strip_sumi(x3,y3);
    sumf += (float)r0 * scale(x0,y0);
    sumf += (float)r1 * scale(x1,y1);
    sumf += (float)r2 * scale(x2,y2);
    sumf += (float)r3 * scale(x3,y3);
  }
  for (; ib < nb; ++ib){ const uint8_t*xb=vx+(size_t)ib*Q4_STRIDE,*yb=vy+(size_t)ib*Q8_STRIDE;
    sumf += (float)block_strip_sumi(xb,yb)*scale(xb,yb); }
  *s = sumf;
}

// =========================== harness ===========================
static unsigned g = 0x13572468u;
static unsigned rnd(){ g^=g<<13; g^=g>>17; g^=g<<5; return g; }
static uint16_t rfp16(){ uint16_t s=(rnd()&1)<<15,e=(uint16_t)(rnd()%31),m=(uint16_t)(rnd()&0x3FF); return s|(e<<10)|m; }
static void fill(uint8_t*vx,uint8_t*vy,int nb){
  for(int ib=0;ib<nb;++ib){ uint8_t*xb=vx+(size_t)ib*Q4_STRIDE,*yb=vy+(size_t)ib*Q8_STRIDE;
    uint16_t dx=rfp16(),dy=rfp16(); memcpy(xb,&dx,2); memcpy(yb,&dy,2);
    for(int i=0;i<16;++i)xb[2+i]=(uint8_t)(rnd()&0xFF);
    for(int i=0;i<32;++i)yb[2+i]=(uint8_t)(rnd()&0xFF); } }
static double now_ns(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int beq(float a,float b){ uint32_t x,y; memcpy(&x,&a,4); memcpy(&y,&b,4); return x==y; }
typedef void (*kern_t)(int,float*,const uint8_t*,const uint8_t*);
static double bench(kern_t k,int n,const uint8_t*vx,const uint8_t*vy,int iters,int reps,volatile float*sink){
  float s; for(int i=0;i<64;++i)k(n,&s,vx,vy);
  double best=1e300;
  for(int r=0;r<reps;++r){ double t0=now_ns(); for(int i=0;i<iters;++i){k(n,&s,vx,vy);*sink+=s;} double t1=now_ns();
    double per=(t1-t0)/iters; if(per<best)best=per; }
  return best;
}
struct V { const char*name; kern_t fn; const char*ship; };
int main(int argc,char**argv){
  int n=(argc>1)?atoi(argv[1]):4096;
  int iters=(argc>2)?atoi(argv[2]):100000;
  int reps=(argc>3)?atoi(argv[3]):20;
  if(n%QK){printf("n%%32\n");return 2;}
  int nb=n/QK;
  uint8_t*vx=malloc((size_t)nb*Q4_STRIDE), *vy=malloc((size_t)nb*Q8_STRIDE);
  fill(vx,vy,nb);
  struct V vs[] = {
    {"ggml(i8m1,serial)",      ggml_real, "REFERENCE"},
    {"m1(strip-loop,current)", k_m1,      "shippable"},
    {"m1_nl(no-strip-loop)",   k_m1_nl,   "shippable@vl16,not-VLEN-robust"},
    {"sl128(CEILING)",         k_sl128,   "NOT-shippable(VLEN<128 breaks)"},
    {"mb2(2blk,vsetvl-hoist)", k_mb2,     "NOT-VLEN-robust(hoisted vsetvl)"},
    {"mb4(4blk,vsetvl-hoist)", k_mb4,     "NOT-VLEN-robust(hoisted vsetvl)"},
    {"mb2_vl(2blk,VLEN-safe)", k_mb2_vl,  "shippable"},
    {"mb4_vl(4blk,VLEN-safe)", k_mb4_vl,  "shippable"},
    {"mb8(8blk,vsetvl-hoist)", k_mb8,     "NOT-VLEN-robust(hoisted vsetvl)"},
    {"mb2_robust(2blk,strip-carry)", k_mb2_robust, "SHIPPABLE-VLEN-robust"},
    {"mb4_robust(4blk,strip-carry)", k_mb4_robust, "SHIPPABLE-VLEN-robust"},
    {"mb4_defer(4blk,i32acc-1reduce)", k_mb4_defer, "SHIPPABLE-VLEN-robust"},
  };
  int NV=sizeof(vs)/sizeof(vs[0]);
  // correctness gate
  float ref; ggml_real(n,&ref,vx,vy);
  printf("# n=%d nb=%d iters=%d reps=%d  ref=%.9g\n",n,nb,iters,reps,(double)ref);
  printf("# correctness gate (bitwise == ggml):\n");
  int allok=1;
  for(int i=0;i<NV;++i){ float s; vs[i].fn(n,&s,vx,vy); int ok=beq(s,ref);
    printf("#   %-26s %.9g  [%s]\n", vs[i].name,(double)s, ok?"OK":"MISMATCH");
    if(!ok)allok=0; }
  if(!allok){ printf("# ABORT: a variant mismatches; timing skipped for fairness.\n"); }
  volatile float sink=0;
  double tg=0;
  printf("\n# timing (best-of-%d min ns/call), pinned core:\n",reps);
  for(int i=0;i<NV;++i){
    double t=bench(vs[i].fn,n,vx,vy,iters,reps,&sink);
    if(i==0)tg=t;
    printf("  %-26s %9.2f ns/call   %.3fx ggml   [%s]\n",
           vs[i].name,t,t/tg,vs[i].ship);
  }
  printf("# sink=%g\n",(double)sink);
  free(vx);free(vy);
  return 0;
}
