// INC-42 tq2_0 x q8_K bit-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.tq2_0_q8_k_block_dot op
// (the COMPLETE ggml ggml_vec_dot_tq2_0_q8_K super-block kernel: the QK_K=256 loop +
// the 2-bit TERNARY weight unpack `((qs>>(l*2))&3)-1` + ONE integer accumulator over
// the whole super-block + the single-fp16-scale fp32 fold + *s store) computes the
// SAME fp32 result `*s` as ggml's tq2_0 kernel BIT-FOR-BIT (memcmp of the float bits).
//
// The byte-exactness ORACLE is ggml's `_generic` (quants.c:482-511) -- the well-
// defined fp32 order our op mirrors. (The board ALSO has an arch/riscv tq2_0 _vlN
// path (arch/riscv/quants.c:6454), but that is a DIFFERENT raw-RVV summation order
// and does NOT define the byte-exact reference -- every sibling op targets _generic.)
// We compare against a faithful transcription of _generic AND an INDEPENDENT RVV
// transcription (a different ternary-decode expression than the emitter's), both
// bit-for-bit, over random + edge cases.
//
// tq2_0 is the TERNARY ({-1,0,+1}) TriLM coverage rung -- one of the LAST TWO uncommon
// ggml dot kernels (the other is tq1_0). block_tq2_0 = { uint8_t qs[64]; ggml_half d }
// (QK_K=256, sizeof 66): the 64 packed 2-bit-weight bytes LEAD the block at +0, and
// the SINGLE fp16 super-block scale d is the SUFFIX at +64 (distinct from every
// sibling, where d is near the front). The activation is block_q8_K (stride 292,
// d @0, qs @4). The decode is `t = ((qs[j+k] >> (l*2)) & 3) - 1` in {-1,0,1,2}; the
// dot pairs q8[j*4 + l*32 + k] with t (j in {0,32}, l in 0..3, k in 0..31). ONE
// int32 accumulator over the super-block, then `sumf += (float)sumi * (y.d *
// fp16(x.d))`. (ggml _generic quants.c:482-511; ggml-common.h block_tq2_0/block_q8_K.)
//
// Buffer sizing is 1:1 here -- one block_q8_K per tq2_0 super-block, nb of each (no
// q1_0-style 4x multiplier landmine). n is always a multiple of 256.
//
// The kernel under test is the UNMODIFIED, compiler-emitted tq2_0 kernel
// (tq2_0_kernel_body.c, #include'd below) -- every line tagged
// source_op=tcrv_rvv.tq2_0_q8_k_block_dot. tq2_0 uses NO libm (the only sanctioned
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
#include "tq2_0_kernel_body.c"

// ---- block formats (ggml-common.h) -------------------------------------------
//   block_tq2_0 = { uint8_t qs[64]; ggml_half d; }      sizeof 66, qs @0, d @64
//   block_q8_K  = { float d; int8_t qs[256]; int16_t bsums[16]; } sizeof 292
static const int QK_K = 256;
static const int TQ2_STRIDE = 66;
static const int Q8K_STRIDE = 292;
static const int TQ2_D_OFF = 64;   // fp16 scale AFTER qs[64]
static const int Q8K_QS_OFF = 4;   // int8 quants AFTER the fp32 d

// ---- board fp16 -> fp32 (mirrors riscv_compute_fp16_to_fp32, lossless) -------
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_tq2_0_q8_K_kernel_ggml_vec_dot_tq2_0_q8_K(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's _generic (quants.c:482-511, the byte-exact same math) ------------
// Parameterised by `bias` (the per-element ternary offset -- real = 1, so the lane
// is ((qs>>shift)&3) - bias; a WRONG bias is the wrong-ternary negative control)
// and `scale_mul` (corrupts the single fp16 super-block scale). The real kernel
// uses bias=1, scale_mul=1.
static void ggml_generic_p(int n, float *s, const void *vx, const void *vy,
                           int bias, float scale_mul) {
  const int nb = n / QK_K;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t *xb = x + (size_t)i * TQ2_STRIDE;
    const uint8_t *yb = y + (size_t)i * Q8K_STRIDE;
    const uint8_t *qs = xb;                        // qs @ +0
    const int8_t *q8 = (const int8_t *)(yb + Q8K_QS_OFF);
    int32_t sumi = 0;
    for (int j = 0; j < 64; j += 32) {
      for (int l = 0; l < 4; ++l) {
        for (int k = 0; k < 32; ++k) {
          sumi += (int)q8[j * 4 + l * 32 + k] *
                  (((qs[j + k] >> (l * 2)) & 3) - bias);
        }
      }
    }
    float yd;
    std::memcpy(&yd, yb + 0, 4);                    // fp32 q8_K scale @ +0
    uint16_t xdh;
    std::memcpy(&xdh, xb + TQ2_D_OFF, 2);           // fp16 tq2_0 scale @ +64
    const float d = yd * (fp16_to_fp32(xdh) * scale_mul);
    sumf += (float)sumi * d;
  }
  *s = sumf;
}

static void ggml_generic(int n, float *s, const void *vx, const void *vy) {
  ggml_generic_p(n, s, vx, vy, 1, 1.0f);
}

// ---- an INDEPENDENT RVV transcription (a different ternary-decode expression) -
// The emitter unpacks 2-bit fields into aux8 with a vadd.vx(-1) bias, then does a
// per-16-lane vwmul/vwredsum. This reference instead builds the ternary lanes by
// unpacking the WHOLE 128-element chunk in a single m4 strip and applies the -1 bias
// via a vsub against a +1 splat -- a genuinely distinct expression of the same
// semantics, reduced over 128 lanes at once.
static void ggml_rvv(int n, float *s, const void *vx, const void *vy) {
  const int nb = n / QK_K;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t *xb = x + (size_t)i * TQ2_STRIDE;
    const uint8_t *yb = y + (size_t)i * Q8K_STRIDE;
    const uint8_t *qs = xb;
    const int8_t *q8 = (const int8_t *)(yb + Q8K_QS_OFF);
    int32_t sumi = 0;
    // chunk j in {0,32}: 32 weight bytes -> 128 ternary lanes pairing with
    // q8[j*4 .. j*4 + 128). Field l (l=0..3) -> q8[j*4 + l*32 + (0..31)]. The
    // 32-lane strip needs e8m2 (VLMAX(e8m1)=16 at VLEN=128 -- a strict e8m1 here
    // would silently cap at 16 lanes); use m2 so all 32 weight bytes participate.
    for (int j = 0; j < 64; j += 32) {
      const uint8_t *qsc = qs + j;
      const int8_t *q8c = q8 + j * 4;
      size_t vl = 32;
      vuint8m2_t w = __riscv_vle8_v_u8m2(qsc, vl);
      for (int l = 0; l < 4; ++l) {
        int sh = l * 2;
        vuint8m2_t fld =
            __riscv_vand_vx_u8m2(__riscv_vsrl_vx_u8m2(w, sh, vl), 0x03, vl);
        vint8m2_t fi = __riscv_vreinterpret_v_u8m2_i8m2(fld);
        vint8m2_t one = __riscv_vmv_v_x_i8m2(1, vl);
        vint8m2_t tern = __riscv_vsub_vv_i8m2(fi, one, vl); // ((qs>>sh)&3) - 1
        vint8m2_t q8v = __riscv_vle8_v_i8m2(q8c + l * 32, vl);
        vint16m4_t prod = __riscv_vwmul_vv_i16m4(tern, q8v, vl);
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t red = __riscv_vwredsum_vs_i16m4_i32m1(prod, zero, vl);
        sumi += __riscv_vmv_x_s_i32m1_i32(red);
      }
    }
    float yd;
    std::memcpy(&yd, yb + 0, 4);
    uint16_t xdh;
    std::memcpy(&xdh, xb + TQ2_D_OFF, 2);
    const float d = yd * fp16_to_fp32(xdh);
    sumf += (float)sumi * d;
  }
  *s = sumf;
}

// ---- random block_tq2_0 / block_q8_K array generators ------------------------
static uint32_t rng = 0x9e3779b9u;
static uint32_t xrand() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static inline void put_fp16(uint8_t *p) {
  _Float16 d = (_Float16)(((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
  uint16_t dh;
  std::memcpy(&dh, &d, 2);
  std::memcpy(p, &dh, 2);
}

static void fill_tq2_0(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + (size_t)i * TQ2_STRIDE;
    for (int j = 0; j < 64; j++)
      b[j] = (uint8_t)(xrand() % 256);             // packed 2-bit weight bytes
    put_fp16(b + TQ2_D_OFF);                        // fp16 scale @ +64
  }
}

static void fill_q8_K(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + (size_t)i * Q8K_STRIDE;
    float d = (((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
    std::memcpy(b + 0, &d, 4);                       // fp32 scale @ +0
    for (int j = 0; j < 256; j++)
      b[Q8K_QS_OFF + j] = (uint8_t)(int8_t)(xrand() % 256); // FULL int8 range
    // bsums @ +260 unused by tq2_0; leave as scratch.
  }
}

// Edge-case fillers. These pin the TERNARY decode at the three realized values
// -1 / 0 / +1 in ISOLATION (the task's named edge set), plus the +2 OOD stress.
//   wmode 0: all qs 0x00 -> every 2-bit field = 0 -> ternary = -1 (all -q8).
//   wmode 1: all qs 0x55 -> every 2-bit field = 1 -> ternary =  0 (all zero -> sumi=0).
//   wmode 2: all qs 0xAA -> every 2-bit field = 2 -> ternary = +1 (all +q8).
//   wmode 3: all qs 0xFF -> every 2-bit field = 3 -> ternary = +2 (out-of-dist stress).
//   wmode 4: marching bytes so every 2-bit field value (0,1,2,3) appears.
static void fill_tq2_0_edge(uint8_t *buf, int nb, int wmode) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + (size_t)i * TQ2_STRIDE;
    for (int j = 0; j < 64; j++) {
      uint8_t v;
      if (wmode == 0) v = 0x00;
      else if (wmode == 1) v = 0x55;
      else if (wmode == 2) v = 0xAA;
      else if (wmode == 3) v = 0xFF;
      else v = (uint8_t)(0x1B ^ (j * 0x4D)); // varied 2-bit fields
      b[j] = v;
    }
    put_fp16(b + TQ2_D_OFF);
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
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * TQ2_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8K_STRIDE);
  fill_tq2_0(vx, nb);
  fill_q8_K(vy, nb);
  int fail = check_buffers(n, vx, vy, "random", generic_delta);
  free(vx);
  free(vy);
  return fail;
}

static int check_edge(int n, int wmode, int ymode, const char *tag,
                      int *generic_delta) {
  int nb = n / QK_K;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * TQ2_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8K_STRIDE);
  fill_tq2_0_edge(vx, nb, wmode);
  fill_q8_K_edge(vy, nb, ymode);
  int fail = check_buffers(n, vx, vy, tag, generic_delta);
  free(vx);
  free(vy);
  return fail;
}

int main() {
  int n_set[] = {256, 512, 768, 1024, 2048, 4096, 8192};
  int total = 0, fails = 0, generic_delta = 0;
  printf("== INC-42 tq2_0 byte-exact (oracle = _generic, the well-defined fp32 "
         "order; quants.c:482) ==\n");
  for (int rep = 0; rep < 400; rep++) {
    for (int n : n_set) {
      total += 1;
      fails += check_random(n, &generic_delta);
    }
  }

  // EDGE: the THREE realized ternary values in ISOLATION -- all-(-1) (qs=0x00 ->
  // -Sum(q8)), all-0 (qs=0x55 -> sumi=0), all-(+1) (qs=0xAA -> +Sum(q8)) -- plus the
  // +2 OOD stress (qs=0xFF) and marching qs, each crossed with q8 saturation (+127 /
  // +/-128 / -127). These pin the TERNARY -1 bias at every realized lane value.
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

  // Negative control 1 (the LOAD-BEARING -1 TERNARY BIAS proof): same input bytes,
  // but the reference uses the WRONG bias (bias=0 -> the lane is the raw 2-bit field
  // ((qs>>sh)&3) in {0,1,2,3} instead of the ternary {-1,0,1,2}). Our kernel embeds
  // the real -1 -> it MUST diverge.
  {
    int n = 2048, nb = n / QK_K;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * TQ2_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8K_STRIDE);
    fill_tq2_0(vx, nb);
    fill_q8_K_edge(vy, nb, 0); // q8 all +127 -> bias-sensitive sum
    float so = 0, s_real = 0, s_wrong = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_p(n, &s_real, vx, vy, 1, 1.0f);
    ggml_generic_p(n, &s_wrong, vx, vy, 0, 1.0f); // wrong: no -1 ternary offset
    if (std::memcmp(&so, &s_real, 4) != 0) {
      printf("NEG-CTRL-1 setup FAILED: ours != real-bias ref (should match)\n");
      fails += 1;
    } else if (std::memcmp(&so, &s_wrong, 4) == 0) {
      printf("NEGATIVE CONTROL 1 FAILED: WRONG (no -1) ternary still matched "
             "(vacuous -- the -1 ternary bias is NOT load-bearing!)\n");
      fails += 1;
    } else {
      printf("negative control 1 OK: WRONG (no -1 -> raw 2-bit field) ternary "
             "diverges (ours=%.9g real=%.9g wrong=%.9g) -> the -1 ternary bias is "
             "load-bearing\n", so, s_real, s_wrong);
    }
    free(vx);
    free(vy);
  }

  // Negative control 2 (the LOAD-BEARING SCALE proof): same input bytes, but the
  // reference scales the fp16 super-block scale by 2. Our kernel uses the real scale
  // -> it MUST diverge.
  {
    int n = 2048, nb = n / QK_K;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * TQ2_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8K_STRIDE);
    fill_tq2_0(vx, nb);
    fill_q8_K_edge(vy, nb, 0);
    float so = 0, s_real = 0, s_wrong = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_p(n, &s_real, vx, vy, 1, 1.0f);
    ggml_generic_p(n, &s_wrong, vx, vy, 1, 2.0f); // wrong d scale (2x)
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

  // Negative control 3: flip ONE weight byte of vx; our kernel must DIFFER from a
  // fresh ggml run on the UNperturbed data (proves the check is non-vacuous: the
  // weight bytes are actually consumed).
  {
    int n = 256, nb = n / QK_K;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * TQ2_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8K_STRIDE);
    fill_tq2_0(vx, nb);
    fill_q8_K_edge(vy, nb, 0);
    float sg = 0;
    ggml_rvv(n, &sg, vx, vy);
    vx[0] ^= 0x03; // flip the low 2-bit field of weight byte 0
    float so = 0;
    our_kernel(n, &so, vx, vy);
    if (std::memcmp(&so, &sg, 4) == 0) {
      printf("NEGATIVE CONTROL 3 FAILED: perturbed weight still matched (vacuous)\n");
      fails += 1;
    } else {
      printf("negative control 3 OK: flipped weight field diverges (the weight "
             "bytes are consumed; check non-vacuous)\n");
    }
    free(vx);
    free(vy);
  }

  if (fails == 0)
    printf("ALL tq2_0 BYTE-EXACT CHECKS PASSED\n");
  else
    printf("tq2_0 had %d FAILURES\n", fails);
  return fails ? 1 : 0;
}
