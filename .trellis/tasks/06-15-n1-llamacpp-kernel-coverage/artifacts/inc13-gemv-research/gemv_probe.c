// inc13-gemv-research: Q4_0 x Q8_0 MULTI-ROW GEMV design-space probe.
//
// THROWAWAY probe (NOT the shipped kernel). Question: for an N-row matvec
// (one shared q8_0 activation, N weight rows -> N fp32 outputs), does a
// multi-row gemv beat calling ggml's per-row vec_dot N times? And WHICH
// multi-row structure wins: activation-hoist (b1) or cross-row-SIMD (b2)?
//
// Board: ssh rvv, 64-core riscv64, VLEN=128 (VLENB=16), rv64gcv_zfh_zvfh, clang 18.
// Build: clang -march=rv64gcv_zfh_zvfh -mabi=lp64d -O3 -ffp-contract=fast
// Run pinned: taskset -c 3 ./gemv_probe <K> <N_rows> <iters> <repeats>
//
// LAYOUTS:
//   All weight rows are STANDARD ggml q4_0 (AoS, 18B blocks, no repack). The
//   activation is STANDARD q8_0 (AoS, 34B blocks). No repack/interleave needed:
//   the win we test is ACTIVATION REUSE across rows, not a special memory format.
//
// VARIANTS (all produce N fp32 outputs s[0..N-1] for the same activation):
//   a_vecdot   : ggml's real per-row RVV vec_dot called N times (THE BASELINE we
//                currently match; this is also ggml's ACTUAL used path at VLEN=128,
//                because ggml's repack gemv is DISABLED at VLEN=128 -- see research).
//   b1_hoist   : standard-layout, ONE row's reduction at a time but the q8 activation
//                block is loaded ONCE per block and reused across all N rows. Per-row
//                vwredsum kept. Isolates the activation-LOAD saving only.
//   b2_xrow4   : cross-row SIMD, 4 rows per tile. Mirrors ggml's 8x8 structure:
//                4 row integer-reductions per block, but the per-block SCALE step
//                (sumi*d_act) folded into a 4-LANE f32 vector across the 4 rows, and
//                the activation block decoded ONCE per block, shared across the 4 rows.
//   b2_xrow8   : cross-row SIMD, 8 rows per tile (two 4-lane f32m1 accumulators).
//
// BYTE-EXACT REFERENCE = a_vecdot (ggml per-row vec_dot, the USED path). Every
// variant's N outputs are memcmp-gated (fp32 bits) against a_vecdot before timing.
// vec_dot folds (sumi*d_wt)*d_act per block; the cross-row variants REPLICATE that
// exact per-lane multiply order so they stay bitwise-equal to the used path.
//
// (We do NOT gate against ggml's 8x8 gemv: it folds (sumi*d_act)*d_wt -- different
//  rounding -- AND it is disabled at VLEN=128 anyway.)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK 32
#define Q4_STRIDE 18   // sizeof(block_q4_0)
#define Q8_STRIDE 34   // sizeof(block_q8_0)

static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}
static inline uint16_t fp32_to_fp16(float f) {
  _Float16 hf = (_Float16)f;
  uint16_t h;
  memcpy(&h, &hf, sizeof(uint16_t));
  return h;
}

// ---- ggml's real per-row RVV vec_dot (transcribed from quants.c:222) ----
static void ggml_vec_dot_q4_0_q8_0_rvv(int n, float *s, const uint8_t *x, const uint8_t *y) {
  const int nb = n / QK;
  const size_t vl = QK / 2; // 16
  float sumf = 0.0f;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = x + (size_t)ib * Q4_STRIDE;
    const uint8_t *yb = y + (size_t)ib * Q8_STRIDE;
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xb + 2, vl);
    vint8m1_t  y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
    vint8m1_t  y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t v0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_a), 8, vl);
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_l), 8, vl);
    vint16m2_t m1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    vint16m2_t m2 = __riscv_vwmacc_vv_i16m2(m1, v1, y1, vl);
    vint32m1_t z   = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(m2, z, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);
    uint16_t dx, dy;
    memcpy(&dx, xb, 2); memcpy(&dy, yb, 2);
    sumf += sumi * fp16_to_fp32(dx) * fp16_to_fp32(dy);
  }
  *s = sumf;
}

// ---- shared single-block integer core: returns i32 sumi for one (q4 row, q8 act) ----
static inline int block_sumi(const uint8_t *xb, vint8m1_t y0, vint8m1_t y1, size_t vl) {
  vuint8m1_t tx = __riscv_vle8_v_u8m1(xb + 2, vl);
  vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
  vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
  vint8m1_t v0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_a), 8, vl);
  vint8m1_t v1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_l), 8, vl);
  vint16m2_t m1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
  vint16m2_t m2 = __riscv_vwmacc_vv_i16m2(m1, v1, y1, vl);
  vint32m1_t z   = __riscv_vmv_v_x_i32m1(0, vl);
  vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(m2, z, vl);
  return __riscv_vmv_x_s_i32m1_i32(vs2);
}

// ====================== VARIANT a: N x vec_dot ======================
static void mv_a_vecdot(int K, int N, float *s, const uint8_t *W, const uint8_t *y) {
  const size_t row_bytes = (size_t)(K / QK) * Q4_STRIDE;
  for (int r = 0; r < N; ++r) {
    ggml_vec_dot_q4_0_q8_0_rvv(K, &s[r], W + (size_t)r * row_bytes, y);
  }
}

// ====================== VARIANT b1: activation-hoist ======================
// load+decode the q8 activation block ONCE per block, reuse across all N rows.
// per-row vwredsum kept; per-row scalar fold in block order (== vec_dot rounding).
static void mv_b1_hoist(int K, int N, float *s, const uint8_t *W, const uint8_t *y) {
  const int nb = K / QK;
  const size_t vl = QK / 2;
  const size_t row_bytes = (size_t)nb * Q4_STRIDE;
  for (int r = 0; r < N; ++r) s[r] = 0.0f;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *yb = y + (size_t)ib * Q8_STRIDE;
    vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);        // loaded ONCE
    vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);   // for all N rows
    uint16_t dyb; memcpy(&dyb, yb, 2);
    float dy = fp16_to_fp32(dyb);
    for (int r = 0; r < N; ++r) {
      const uint8_t *xb = W + (size_t)r * row_bytes + (size_t)ib * Q4_STRIDE;
      int sumi = block_sumi(xb, y0, y1, vl);
      uint16_t dxb; memcpy(&dxb, xb, 2);
      s[r] += sumi * fp16_to_fp32(dxb) * dy;  // (sumi*d_wt)*d_act -- vec_dot order
    }
  }
}

// ---- cross-row group-of-4: 4 rows -> one f32m1 (4-lane) accumulator across all blocks ----
// The per-block SCALE step is a VECTOR fmacc across the 4 rows. The activation block is
// decoded ONCE per block and shared across the 4 rows. Byte-exact: each lane folds its
// blocks ascending, multiply order == vec_dot's (sumi*d_wt)*d_act.
static void xrow_group4(int nb, size_t vl, size_t row_bytes,
                        float *s4, const uint8_t *W4, const uint8_t *y) {
  vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 4);
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *yb = y + (size_t)ib * Q8_STRIDE;
    vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
    vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
    uint16_t dyb; memcpy(&dyb, yb, 2);
    float dy = fp16_to_fp32(dyb);
    float lane_sd[4];
    for (int j = 0; j < 4; ++j) {
      const uint8_t *xb = W4 + (size_t)j * row_bytes + (size_t)ib * Q4_STRIDE;
      int sumi = block_sumi(xb, y0, y1, vl);
      uint16_t dxb; memcpy(&dxb, xb, 2);
      lane_sd[j] = sumi * fp16_to_fp32(dxb);   // sumi*d_wt, per lane (scalar)
    }
    vfloat32m1_t v_sd = __riscv_vle32_v_f32m1(lane_sd, 4);
    acc = __riscv_vfmacc_vf_f32m1(acc, dy, v_sd, 4);  // acc += (sumi*d_wt) * d_act, x4 rows
  }
  __riscv_vse32_v_f32m1(s4, acc, 4);
}

// ====================== VARIANT b2: cross-row SIMD, TILE rows ======================
// TILE must be a multiple of 4. Each group-of-4 keeps one live f32m1 accumulator.
static void mv_b2_xrow(int K, int N, int TILE, float *s, const uint8_t *W, const uint8_t *y) {
  const int nb = K / QK;
  const size_t vl = QK / 2;
  const size_t row_bytes = (size_t)nb * Q4_STRIDE;
  for (int r0 = 0; r0 < N; r0 += 4) {
    xrow_group4(nb, vl, row_bytes, &s[r0], W + (size_t)r0 * row_bytes, y);
  }
  (void)TILE;
}

// ============================ harness ============================
static double now_ns(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e9 + t.tv_nsec;
}

static void fill_q4_row(uint8_t *row, int nb, unsigned *seed) {
  for (int b = 0; b < nb; ++b) {
    uint8_t *blk = row + (size_t)b * Q4_STRIDE;
    uint16_t d = fp32_to_fp16(0.005f + (rand_r(seed) % 100) * 0.0003f);
    memcpy(blk, &d, 2);
    for (int j = 0; j < QK / 2; ++j) blk[2 + j] = (uint8_t)(rand_r(seed) & 0xFF);
  }
}
static void fill_q8_act(uint8_t *y, int nb, unsigned *seed) {
  for (int b = 0; b < nb; ++b) {
    uint8_t *blk = y + (size_t)b * Q8_STRIDE;
    uint16_t d = fp32_to_fp16(0.01f + (rand_r(seed) % 100) * 0.0004f);
    memcpy(blk, &d, 2);
    for (int j = 0; j < QK; ++j) ((int8_t *)(blk + 2))[j] = (int8_t)((rand_r(seed) % 255) - 127);
  }
}

typedef void (*mvfn)(int, int, float *, const uint8_t *, const uint8_t *);

static double bench(mvfn fn, int K, int N, float *s, const uint8_t *W, const uint8_t *y,
                    long iters, int reps) {
  double best = 1e30;
  for (int rep = 0; rep < reps; ++rep) {
    double t0 = now_ns();
    for (long it = 0; it < iters; ++it) fn(K, N, s, W, y);
    double t1 = now_ns();
    double per = (t1 - t0) / iters;   // ns per N-row matvec
    if (per < best) best = per;
  }
  return best;
}

static void mv_b2_w(int K, int N, float *s, const uint8_t *W, const uint8_t *y) {
  mv_b2_xrow(K, N, 4, s, W, y);
}

int main(int argc, char **argv) {
  int  K     = argc > 1 ? atoi(argv[1]) : 4096;
  int  N     = argc > 2 ? atoi(argv[2]) : 8;     // rows
  long iters = argc > 3 ? atol(argv[3]) : 20000;
  int  reps  = argc > 4 ? atoi(argv[4]) : 30;

  if (K % QK != 0) { fprintf(stderr, "K must be %% %d\n", QK); return 1; }
  if (N % 4 != 0)  { fprintf(stderr, "N must be %% 4 (b2 tile)\n"); return 1; }
  const int nb = K / QK;
  const size_t row_bytes = (size_t)nb * Q4_STRIDE;

  uint8_t *W = malloc((size_t)N * row_bytes);
  uint8_t *y = malloc((size_t)nb * Q8_STRIDE);
  float *s_ref = malloc((size_t)N * sizeof(float));
  float *s     = malloc((size_t)N * sizeof(float));

  unsigned seed = 12345;
  for (int r = 0; r < N; ++r) fill_q4_row(W + (size_t)r * row_bytes, nb, &seed);
  fill_q8_act(y, nb, &seed);

  // reference = a_vecdot (the USED path)
  mv_a_vecdot(K, N, s_ref, W, y);

  printf("# Q4_0xQ8_0 multi-row GEMV probe  K=%d  N(rows)=%d  iters=%ld reps=%d\n",
         K, N, iters, reps);
  printf("# ns per N-row matvec (min of reps). ref s[0]=%.8e\n", s_ref[0]);

  struct { const char *name; mvfn fn; } V[] = {
    {"a_vecdot  (Nx vec_dot, BASELINE/USED path)",       mv_a_vecdot},
    {"b1_hoist  (act loaded once/blk, per-row reduce)",  mv_b1_hoist},
    {"b2_xrow   (cross-row SIMD, 4-row tiles)",          mv_b2_w},
  };

  for (size_t v = 0; v < sizeof(V)/sizeof(V[0]); ++v) {
    memset(s, 0, (size_t)N * sizeof(float));
    V[v].fn(K, N, s, W, y);
    int ok = (memcmp(s, s_ref, (size_t)N * sizeof(float)) == 0);
    double t = bench(V[v].fn, K, N, s, W, y, iters, reps);
    double per_row = t / N;
    double spd = -1.0;
    printf("  %-48s  %9.1f ns/matvec  %7.1f ns/row  gate=%s\n",
           V[v].name, t, per_row, ok ? "PASS" : "FAIL-BITEXACT");
    (void)spd;
  }

  free(W); free(y); free(s_ref); free(s);
  return 0;
}
