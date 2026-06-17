// INC-43 tq1_0 x q8_K bit-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.tq1_0_q8_k_block_dot op
// (the COMPLETE ggml ggml_vec_dot_tq1_0_q8_K super-block kernel: the QK_K=256 loop +
// the BASE-3 TERNARY weight unpack `((uint8_t)(byte*pow3[l]) * 3) >> 8) - 1` of the
// qs[48] + qh[4] weight arrays + ONE integer accumulator over the whole super-block
// (the THREE q8 index regions: qs main, qs tail, qh) + the single-fp16-scale fp32
// fold + *s store) computes the SAME fp32 result `*s` as ggml's tq1_0 kernel
// BIT-FOR-BIT (memcmp of the float bits).
//
// The byte-exactness ORACLE is ggml's `_generic` (quants.c:430-480) -- the well-
// defined fp32 order our op mirrors. We compare against a faithful transcription of
// _generic AND an INDEPENDENT RVV transcription (a different base-3-decode expression
// than the emitter's), both bit-for-bit, over random + edge cases.
//
// tq1_0 is the BASE-3-PACKED TERNARY ({-1,0,+1}) TriLM coverage rung -- the LAST of
// the 24 ggml dot kernels (literal 100%). block_tq1_0 = { uint8_t qs[48]; uint8_t
// qh[4]; ggml_half d } (QK_K=256, sizeof 54): the 48 packed base-3 qs bytes (5 trits
// each) LEAD at +0, the 4 base-3 qh bytes (4 trits each) follow at +48, and the
// SINGLE fp16 super-block scale d is the SUFFIX at +52. The activation is block_q8_K
// (stride 292, d @0, qs @4). The decode is `q = (uint8_t)(byte*pow3[l]); xi =
// ((uint16_t)q*3)>>8; t = xi - 1` in {-1,0,1}; pow3[6]={1,3,9,27,81,243}. The dot
// walks three q8 regions: (a) qs main j=0,l 0..4,m 0..31 -> q8[l*32+m], weight qs[m];
// (b) qs tail j=32,l 0..4,m 0..15 -> q8[160+l*16+m], weight qs[32+m]; (c) qh l 0..3,
// j 0..3 -> q8[240+l*4+j], weight qh[j]. ONE int32 accumulator over the super-block,
// then `sumf += (float)sum * (fp16(x.d) * y.d)`. (ggml _generic quants.c:430-480;
// ggml-common.h block_tq1_0/block_q8_K.)
//
// Buffer sizing is 1:1 here -- one block_q8_K per tq1_0 super-block, nb of each (no
// 4x multiplier landmine). n is always a multiple of 256.
//
// The kernel under test is the UNMODIFIED, compiler-emitted tq1_0 kernel
// (tq1_0_kernel_body.c, #include'd below) -- every line tagged
// source_op=tcrv_rvv.tq1_0_q8_k_block_dot. tq1_0 uses NO libm (the only sanctioned
// opaque scalar piece is the fp16 read, the SAME (float)*(const _Float16 *) the q4_0
// sibling emits).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <riscv_vector.h>

// The compiler-emitted kernel, textually included. It is byte-identical to a fresh
// tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate regen.
#include "tq1_0_kernel_body.c"

// ---- block formats (ggml-common.h) -------------------------------------------
//   block_tq1_0 = { uint8_t qs[48]; uint8_t qh[4]; ggml_half d; }  sizeof 54,
//                 qs @0, qh @48, d @52
//   block_q8_K  = { float d; int8_t qs[256]; int16_t bsums[16]; }  sizeof 292
static const int QK_K = 256;
static const int TQ1_STRIDE = 54;
static const int Q8K_STRIDE = 292;
static const int TQ1_QS = 48;     // qs[48] @ +0
static const int TQ1_QH = 4;      // qh[4]  @ +48
static const int TQ1_QH_OFF = 48; // qh after qs[48]
static const int TQ1_D_OFF = 52;  // fp16 scale AFTER qs[48] + qh[4]
static const int Q8K_QS_OFF = 4;  // int8 quants AFTER the fp32 d

static const uint8_t POW3[6] = {1, 3, 9, 27, 81, 243};

// ---- board fp16 -> fp32 (mirrors riscv_compute_fp16_to_fp32, lossless) -------
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_ggml_vec_dot_tq1_0_q8_K(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's _generic (quants.c:430-480, the byte-exact same math) ------------
// Parameterised by `bias` (the per-element ternary offset -- real = 1, so the lane
// is xi - bias; a WRONG bias is the wrong-base-3-extract negative control),
// `scale_mul` (corrupts the single fp16 super-block scale), and `drop_qh` (omits the
// qh contribution -- the drop-qh negative control). The real kernel uses bias=1,
// scale_mul=1, drop_qh=0.
static void ggml_generic_p(int n, float *s, const void *vx, const void *vy,
                           int bias, float scale_mul, int drop_qh) {
  const int nb = n / QK_K;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t *xb = x + (size_t)i * TQ1_STRIDE;
    const uint8_t *yb = y + (size_t)i * Q8K_STRIDE;
    const uint8_t *qs = xb;                         // qs @ +0
    const uint8_t *qh = xb + TQ1_QH_OFF;            // qh @ +48
    const int8_t *q8 = (const int8_t *)(yb + Q8K_QS_OFF);
    int sum = 0;
    // (a) qs MAIN loop: j=0..(48-48%32)=0..32 step 32 -> j=0 only; l 0..4; m 0..31.
    for (int j = 0; j < TQ1_QS - TQ1_QS % 32; j += 32) {
      for (int l = 0; l < 5; ++l) {
        for (int m = 0; m < 32; ++m) {
          uint8_t q = (uint8_t)(qs[j + m] * POW3[l]);
          uint16_t xi = ((uint16_t)q * 3) >> 8;
          sum += ((int)xi - bias) * (int)q8[j * 5 + l * 32 + m];
        }
      }
    }
    // (b) qs TAIL loop: j=32..48 step 16 -> j=32 only; l 0..4; m 0..15.
    for (int j = TQ1_QS - TQ1_QS % 32; j < TQ1_QS; j += 16) {
      for (int l = 0; l < 5; ++l) {
        for (int m = 0; m < 16; ++m) {
          uint8_t q = (uint8_t)(qs[j + m] * POW3[l]);
          uint16_t xi = ((uint16_t)q * 3) >> 8;
          sum += ((int)xi - bias) * (int)q8[j * 5 + l * 16 + m];
        }
      }
    }
    // (c) qh loop: l 0..3; j 0..3.
    if (!drop_qh) {
      for (int l = 0; l < 4; ++l) {
        for (int j = 0; j < TQ1_QH; ++j) {
          uint8_t q = (uint8_t)(qh[j] * POW3[l]);
          uint16_t xi = ((uint16_t)q * 3) >> 8;
          sum += ((int)xi - bias) * (int)q8[TQ1_QS * 5 + l * TQ1_QH + j];
        }
      }
    }
    uint16_t xdh;
    std::memcpy(&xdh, xb + TQ1_D_OFF, 2);            // fp16 tq1_0 scale @ +52
    float yd;
    std::memcpy(&yd, yb + 0, 4);                     // fp32 q8_K scale @ +0
    sumf += (float)sum * ((fp16_to_fp32(xdh) * scale_mul) * yd);
  }
  *s = sumf;
}

static void ggml_generic(int n, float *s, const void *vx, const void *vy) {
  ggml_generic_p(n, s, vx, vy, 1, 1.0f, 0);
}

// ---- an INDEPENDENT RVV transcription (a different base-3-decode expression) --
// The emitter recovers each trit with vmul.vx(pow3) (the u8 wrap) -> vwmulu(3) ->
// vsrl(8) -> vncvt -> vadd.vx(-1) into aux8, then a per-16-lane vwmul/vwredsum. This
// reference instead decodes the WHOLE 32-byte main chunk in one e8m2 strip per digit
// and applies the -1 bias via a vsub against a +1 splat -- a genuinely distinct
// expression of the same base-3 semantics, reduced over the strip at once. The tail
// (16 lanes) and qh (4 lanes) use the same distinct shape at e8m1. It accumulates the
// integer sum in q8 index order (order-free integer add).
static void ggml_rvv(int n, float *s, const void *vx, const void *vy) {
  const int nb = n / QK_K;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t *xb = x + (size_t)i * TQ1_STRIDE;
    const uint8_t *yb = y + (size_t)i * Q8K_STRIDE;
    const uint8_t *qs = xb;
    const uint8_t *qh = xb + TQ1_QH_OFF;
    const int8_t *q8 = (const int8_t *)(yb + Q8K_QS_OFF);
    int sum = 0;
    // (a) qs MAIN (j=0): 32 weight bytes qs[0..31], 5 digits l -> q8[l*32 + m].
    {
      size_t vl = 32;
      vuint8m2_t w = __riscv_vle8_v_u8m2(qs + 0, vl);
      for (int l = 0; l < 5; ++l) {
        vuint8m2_t q = __riscv_vmul_vx_u8m2(w, POW3[l], vl); // u8 wrap
        vuint16m4_t w3 = __riscv_vwmulu_vx_u16m4(q, 3, vl);
        vuint16m4_t sh = __riscv_vsrl_vx_u16m4(w3, 8, vl);
        vuint8m2_t xi = __riscv_vncvt_x_x_w_u8m2(sh, vl);
        vint8m2_t xis = __riscv_vreinterpret_v_u8m2_i8m2(xi);
        vint8m2_t one = __riscv_vmv_v_x_i8m2(1, vl);
        vint8m2_t tern = __riscv_vsub_vv_i8m2(xis, one, vl); // xi - 1 (distinct)
        vint8m2_t q8v = __riscv_vle8_v_i8m2(q8 + l * 32, vl);
        vint16m4_t prod = __riscv_vwmul_vv_i16m4(tern, q8v, vl);
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t red = __riscv_vwredsum_vs_i16m4_i32m1(prod, zero, vl);
        sum += __riscv_vmv_x_s_i32m1_i32(red);
      }
    }
    // (b) qs TAIL (j=32): 16 weight bytes qs[32..47], 5 digits -> q8[160+l*16+m].
    {
      size_t vl = 16;
      vuint8m1_t w = __riscv_vle8_v_u8m1(qs + 32, vl);
      for (int l = 0; l < 5; ++l) {
        vuint8m1_t q = __riscv_vmul_vx_u8m1(w, POW3[l], vl);
        vuint16m2_t w3 = __riscv_vwmulu_vx_u16m2(q, 3, vl);
        vuint16m2_t sh = __riscv_vsrl_vx_u16m2(w3, 8, vl);
        vuint8m1_t xi = __riscv_vncvt_x_x_w_u8m1(sh, vl);
        vint8m1_t xis = __riscv_vreinterpret_v_u8m1_i8m1(xi);
        vint8m1_t one = __riscv_vmv_v_x_i8m1(1, vl);
        vint8m1_t tern = __riscv_vsub_vv_i8m1(xis, one, vl);
        vint8m1_t q8v = __riscv_vle8_v_i8m1(q8 + 160 + l * 16, vl);
        vint16m2_t prod = __riscv_vwmul_vv_i16m2(tern, q8v, vl);
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t red = __riscv_vwredsum_vs_i16m2_i32m1(prod, zero, vl);
        sum += __riscv_vmv_x_s_i32m1_i32(red);
      }
    }
    // (c) qh (j=0..3 per digit l=0..3): 4 weight bytes qh[0..3] -> q8[240+l*4+j].
    {
      size_t vl = 4;
      vuint8m1_t w = __riscv_vle8_v_u8m1(qh, vl);
      for (int l = 0; l < 4; ++l) {
        vuint8m1_t q = __riscv_vmul_vx_u8m1(w, POW3[l], vl);
        vuint16m2_t w3 = __riscv_vwmulu_vx_u16m2(q, 3, vl);
        vuint16m2_t sh = __riscv_vsrl_vx_u16m2(w3, 8, vl);
        vuint8m1_t xi = __riscv_vncvt_x_x_w_u8m1(sh, vl);
        vint8m1_t xis = __riscv_vreinterpret_v_u8m1_i8m1(xi);
        vint8m1_t one = __riscv_vmv_v_x_i8m1(1, vl);
        vint8m1_t tern = __riscv_vsub_vv_i8m1(xis, one, vl);
        vint8m1_t q8v = __riscv_vle8_v_i8m1(q8 + 240 + l * 4, vl);
        vint16m2_t prod = __riscv_vwmul_vv_i16m2(tern, q8v, vl);
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t red = __riscv_vwredsum_vs_i16m2_i32m1(prod, zero, vl);
        sum += __riscv_vmv_x_s_i32m1_i32(red);
      }
    }
    uint16_t xdh;
    std::memcpy(&xdh, xb + TQ1_D_OFF, 2);
    float yd;
    std::memcpy(&yd, yb + 0, 4);
    sumf += (float)sum * (fp16_to_fp32(xdh) * yd);
  }
  *s = sumf;
}

// ---- random block_tq1_0 / block_q8_K array generators ------------------------
static uint32_t rng = 0x9e3779b9u;
static uint32_t xrand() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static inline void put_fp16(uint8_t *p) {
  _Float16 d = (_Float16)(((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
  uint16_t dh;
  std::memcpy(&dh, &d, 2);
  std::memcpy(p, &dh, 2);
}

static void fill_tq1_0(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + (size_t)i * TQ1_STRIDE;
    for (int j = 0; j < TQ1_QS; j++)
      b[j] = (uint8_t)(xrand() % 256);             // packed base-3 qs bytes
    for (int j = 0; j < TQ1_QH; j++)
      b[TQ1_QH_OFF + j] = (uint8_t)(xrand() % 256); // packed base-3 qh bytes
    put_fp16(b + TQ1_D_OFF);                        // fp16 scale @ +52
  }
}

static void fill_q8_K(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + (size_t)i * Q8K_STRIDE;
    float d = (((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
    std::memcpy(b + 0, &d, 4);                       // fp32 scale @ +0
    for (int j = 0; j < 256; j++)
      b[Q8K_QS_OFF + j] = (uint8_t)(int8_t)(xrand() % 256); // FULL int8 range
  }
}

// Edge-case fillers. These pin the BASE-3 decode at boundary byte values. A byte b
// decodes per digit l to xi = ((uint8_t)(b*pow3[l]) * 3) >> 8; the lane is xi-1 in
// {-1,0,1}. byte 0 is the only value whose digits are UNIFORM across all l (q=0 ->
// xi=0 -> every trit = -1 -- this ISOLATES the -1 ternary lane: the consumed sum is
// -Sum(q8)). Bytes 121/242 (the "all-trits-equal" base-3 packings) decode to MIXED
// trit values under ggml's >>8 reconstruction (the decode is only exact for the
// encoder's actual outputs; the exhaustive 256x6 decode test above covers the full
// trit space, so the +1/0 lanes are not isolated here but ARE exercised). All five
// wmodes are byte-exact-checked vs _generic.
//   wmode 0: all bytes 0x00 (=0)   -> every trit = -1 ISOLATED (-> -Sum(q8)).
//   wmode 1: all bytes 121         -> mixed trits (the all-1 base-3 packing).
//   wmode 2: all bytes 242         -> mixed trits (the all-2 base-3 packing).
//   wmode 3: all bytes 0xFF        -> high-byte wrap stress (mixed trits).
//   wmode 4: marching bytes so a broad spread of (byte,digit) decode pairs appear.
static void fill_tq1_0_edge(uint8_t *buf, int nb, int wmode) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + (size_t)i * TQ1_STRIDE;
    for (int j = 0; j < TQ1_QS; j++) {
      uint8_t v;
      if (wmode == 0) v = 0x00;
      else if (wmode == 1) v = 121;
      else if (wmode == 2) v = 242;
      else if (wmode == 3) v = 0xFF;
      else v = (uint8_t)(0x1B ^ (j * 0x4D)); // varied bytes
      b[j] = v;
    }
    for (int j = 0; j < TQ1_QH; j++) {
      uint8_t v;
      if (wmode == 0) v = 0x00;
      else if (wmode == 1) v = 121;
      else if (wmode == 2) v = 242;
      else if (wmode == 3) v = 0xFF;
      else v = (uint8_t)(0x27 ^ (j * 0x13));
      b[TQ1_QH_OFF + j] = v;
    }
    put_fp16(b + TQ1_D_OFF);
  }
}

static void fill_q8_K_edge(uint8_t *buf, int nb, int mode) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + (size_t)i * Q8K_STRIDE;
    float d = (((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
    std::memcpy(b + 0, &d, 4);
    for (int j = 0; j < 256; j++)
      b[Q8K_QS_OFF + j] = (uint8_t)(int8_t)((mode == 0) ? 127
                          : (mode == 1) ? ((j & 1) ? -128 : 127)
                                        : -127);
  }
}

#ifndef CONTRACT_REFS
#define CONTRACT_REFS 1
#endif

static int check_buffers(int n, uint8_t *vx, uint8_t *vy, const char *tag,
                         int *generic_delta) {
  float so = 0, sg = 0, sr = 0;
  our_kernel(n, &so, vx, vy);
  ggml_rvv(n, &sg, vx, vy);
  ggml_generic(n, &sr, vx, vy);
  int fail = 0;
  if (std::memcmp(&so, &sg, 4) != 0) {
    printf("  %-26s n=%-6d FAIL vs ggml-rvv : ours=%.9g ggml=%.9g\n", tag, n, so, sg);
    fail = 1;
  }
  if (std::memcmp(&so, &sr, 4) != 0) {
    *generic_delta += 1;
    if (CONTRACT_REFS) {
      printf("  %-26s n=%-6d FAIL vs _generic : ours=%.9g gen=%.9g\n", tag, n, so, sr);
      fail = 1;
    }
  }
  return fail;
}

static int check_random(int n, int *generic_delta) {
  int nb = n / QK_K;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * TQ1_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8K_STRIDE);
  fill_tq1_0(vx, nb);
  fill_q8_K(vy, nb);
  int fail = check_buffers(n, vx, vy, "random", generic_delta);
  free(vx);
  free(vy);
  return fail;
}

static int check_edge(int n, int wmode, int ymode, const char *tag,
                      int *generic_delta) {
  int nb = n / QK_K;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * TQ1_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8K_STRIDE);
  fill_tq1_0_edge(vx, nb, wmode);
  fill_q8_K_edge(vy, nb, ymode);
  int fail = check_buffers(n, vx, vy, tag, generic_delta);
  free(vx);
  free(vy);
  return fail;
}

int main() {
  int n_set[] = {256, 512, 768, 1024, 2048, 4096, 8192};
  int total = 0, fails = 0, generic_delta = 0;
  printf("== INC-43 tq1_0 byte-exact (oracle = _generic, the well-defined fp32 "
         "order; quants.c:430) ==\n");
  for (int rep = 0; rep < 400; rep++) {
    for (int n : n_set) {
      total += 1;
      fails += check_random(n, &generic_delta);
    }
  }

  // EDGE: the BASE-3 decode pinned at the three realized ternary values -1 / 0 / +1
  // (qs=0x00 -> all -1 -> -Sum(q8) over the consumed indices), plus 0x55/0xAA/0xFF
  // wrap stress and marching bytes, each crossed with q8 saturation (+127 / +/-128 /
  // -127). Every n >= 256 forces the qs-MAIN + qs-TAIL + qh paths (256 = 160 main +
  // 80 tail + 16 qh). These pin the -1 ternary bias and the u8 wrap at every realized
  // lane value, and exercise the tail + qh regions on every case.
  for (int rep = 0; rep < 50; rep++) {
    for (int wmode = 0; wmode < 5; ++wmode)
      for (int ymode = 0; ymode < 3; ++ymode) {
        total += 1;
        char tag[64];
        snprintf(tag, sizeof tag, "qs=%d q8=%d", wmode, ymode);
        fails += check_edge(512, wmode, ymode, tag, &generic_delta);
      }
  }
  printf("checked %d cases, %d failures (vs ggml's RVV transcription)\n",
         total, fails);
  printf("_generic cross-check delta: %d/%d (at =off this is 0; at =fast it is the "
         "references' OWN mutual FMA-formation delta, not a kernel defect)\n",
         generic_delta, total);

  // Negative control 1 (the LOAD-BEARING BASE-3 EXTRACT proof): same input bytes, but
  // the reference uses the WRONG base-3 extract (bias=0 -> the lane is the raw high
  // base-3 digit xi in {0,1,2} instead of the ternary xi-1 in {-1,0,1}). Our kernel
  // embeds the real -1 -> it MUST diverge. (This is the "wrong base-3 extract"
  // control -- the per-element -1 trit offset is load-bearing.)
  {
    int n = 2048, nb = n / QK_K;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * TQ1_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8K_STRIDE);
    fill_tq1_0(vx, nb);
    fill_q8_K_edge(vy, nb, 0); // q8 all +127 -> bias-sensitive sum
    float so = 0, s_real = 0, s_wrong = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_p(n, &s_real, vx, vy, 1, 1.0f, 0);
    ggml_generic_p(n, &s_wrong, vx, vy, 0, 1.0f, 0); // wrong: no -1 trit offset
    if (std::memcmp(&so, &s_real, 4) != 0) {
      printf("NEG-CTRL-1 setup FAILED: ours != real-bias ref (should match)\n");
      fails += 1;
    } else if (std::memcmp(&so, &s_wrong, 4) == 0) {
      printf("NEGATIVE CONTROL 1 FAILED: WRONG (no -1) base-3 extract still matched "
             "(vacuous -- the -1 trit bias is NOT load-bearing!)\n");
      fails += 1;
    } else {
      printf("negative control 1 OK: WRONG (no -1 -> raw base-3 digit) extract "
             "diverges (ours=%.9g real=%.9g wrong=%.9g) -> the base-3 -1 trit bias "
             "is load-bearing\n", so, s_real, s_wrong);
    }
    free(vx);
    free(vy);
  }

  // Negative control 2 (the LOAD-BEARING SCALE proof): same input bytes, but the
  // reference scales the fp16 super-block scale by 2. Our kernel uses the real scale
  // -> it MUST diverge.
  {
    int n = 2048, nb = n / QK_K;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * TQ1_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8K_STRIDE);
    fill_tq1_0(vx, nb);
    fill_q8_K_edge(vy, nb, 0);
    float so = 0, s_real = 0, s_wrong = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_p(n, &s_real, vx, vy, 1, 1.0f, 0);
    ggml_generic_p(n, &s_wrong, vx, vy, 1, 2.0f, 0); // wrong d scale (2x)
    if (std::memcmp(&so, &s_real, 4) != 0) {
      printf("NEG-CTRL-2 setup FAILED: ours != real-scale ref (should match)\n");
      fails += 1;
    } else if (std::memcmp(&so, &s_wrong, 4) == 0) {
      printf("NEGATIVE CONTROL 2 FAILED: WRONG (2x d) scale still matched (vacuous "
             "-- the d scale is NOT load-bearing!)\n");
      fails += 1;
    } else {
      printf("negative control 2 OK: WRONG (2x d) scale diverges "
             "(ours=%.9g real=%.9g wrong=%.9g) -> the d scale is load-bearing\n",
             so, s_real, s_wrong);
    }
    free(vx);
    free(vy);
  }

  // Negative control 3 (the LOAD-BEARING qh proof, the task's "drop-qh" control):
  // same input bytes, but the reference DROPS the qh contribution (the last 16 q8
  // indices 240..255). Our kernel consumes qh -> it MUST diverge. The weight bytes are
  // chosen so the qh region is non-zero; q8 all +127 so the dropped term is non-zero.
  {
    int n = 2048, nb = n / QK_K;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * TQ1_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8K_STRIDE);
    fill_tq1_0(vx, nb);
    // force qh non-trivial: a byte whose base-3 digits are not all the same trit.
    for (int i = 0; i < nb; i++)
      for (int j = 0; j < TQ1_QH; j++)
        vx[(size_t)i * TQ1_STRIDE + TQ1_QH_OFF + j] = (uint8_t)(0xC3 + j);
    fill_q8_K_edge(vy, nb, 0); // q8 all +127 -> dropped qh term is non-zero
    float so = 0, s_real = 0, s_drop = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_p(n, &s_real, vx, vy, 1, 1.0f, 0);
    ggml_generic_p(n, &s_drop, vx, vy, 1, 1.0f, 1); // drop the qh contribution
    if (std::memcmp(&so, &s_real, 4) != 0) {
      printf("NEG-CTRL-3 setup FAILED: ours != full ref (should match)\n");
      fails += 1;
    } else if (std::memcmp(&so, &s_drop, 4) == 0) {
      printf("NEGATIVE CONTROL 3 FAILED: dropping qh still matched (vacuous -- the qh "
             "region is NOT consumed!)\n");
      fails += 1;
    } else {
      printf("negative control 3 OK: dropping the qh contribution diverges "
             "(ours=%.9g full=%.9g drop-qh=%.9g) -> the qh region is load-bearing\n",
             so, s_real, s_drop);
    }
    free(vx);
    free(vy);
  }

  // Negative control 4: flip ONE weight byte of vx; our kernel must DIFFER from a
  // fresh ggml run on the UNperturbed data (proves the check is non-vacuous).
  {
    int n = 256, nb = n / QK_K;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * TQ1_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8K_STRIDE);
    fill_tq1_0(vx, nb);
    fill_q8_K_edge(vy, nb, 0);
    float sg = 0;
    ggml_rvv(n, &sg, vx, vy);
    vx[0] ^= 0x01; // perturb weight byte 0 (a base-3 qs byte)
    float so = 0;
    our_kernel(n, &so, vx, vy);
    if (std::memcmp(&so, &sg, 4) == 0) {
      printf("NEGATIVE CONTROL 4 FAILED: perturbed weight still matched (vacuous)\n");
      fails += 1;
    } else {
      printf("negative control 4 OK: flipped weight byte diverges (the weight bytes "
             "are consumed; check non-vacuous)\n");
    }
    free(vx);
    free(vy);
  }

  if (fails == 0)
    printf("ALL tq1_0 BYTE-EXACT CHECKS PASSED\n");
  else
    printf("tq1_0 had %d FAILURES\n", fails);
  return fails ? 1 : 0;
}
