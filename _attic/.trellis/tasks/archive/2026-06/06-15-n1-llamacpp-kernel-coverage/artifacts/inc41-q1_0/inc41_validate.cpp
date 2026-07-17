// INC-41 q1_0 x q8_0 bit-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.q1_0_q8_0_block_dot op
// (the COMPLETE ggml ggml_vec_dot_q1_0_q8_0 block kernel: super-block QK=128 loop +
// four q8_0 sub-blocks + per-sub-block fp16 scale + the BINARY {-1,+1} sign decode +
// two-level fp32 accumulation + *s store) computes the SAME fp32 result `*s` as
// ggml's q1_0 kernel BIT-FOR-BIT (memcmp of the float bits). There is NO arch/riscv
// q1_0 kernel -- ggml_vec_dot_q1_0_q8_0 is the _generic kernel (quants.c:123), so
// _generic IS the real board kernel AND the oracle. We compare against a faithful
// transcription of _generic AND an INDEPENDENT RVV transcription (a different
// kmask/vmsne/vmerge sign-plane expression than the emitter's), both bit-for-bit,
// over random + edge cases.
//
// q1_0 is the BINARY ({-1,+1}) class -- one of the last three uncommon ggml dot
// kernels. block_q1_0 = { ggml_half d; uint8_t qs[16] } (QK1_0=128, sizeof 18):
// the fp16 scale d0 at +0, then 16 packed BIT bytes (128 element signs) at +2. ONE
// super-block (128 elems) spans FOUR block_q8_0 blocks: sub-block k (k=0..3) reads
// q8 block 4*ib+k, and its 32 bits live at weight bytes +2 + k*4 .. +2 + k*4 + 3.
// A SET bit -> +q8, a CLEAR bit -> -q8; the q8 value is the magnitude. The fold is
// ggml's order: sumf += d0 * sum_k( d1_k * sum_block_k ).
// (ggml _generic quants.c:123-171; ggml-common.h block_q1_0 / block_q8_0.)
//
// CRITICAL (the buffer-sizing landmine): nb = n/128 super-blocks, but the q8 stream
// is n/32 = 4*nb q8_0 blocks (sub-blocks k=1,2,3 read q8 block 4*ib+1..+3). Under-
// sizing the q8 buffer to nb would make OUR kernel AND the reference read the SAME
// OOB addresses -> a FALSE pass that never validates the k>=1 path. We size 4*nb.
//
// The kernel under test is the UNMODIFIED, compiler-emitted q1_0 kernel
// (q1_0_kernel.cpp, #include'd below) -- every line tagged
// source_op=tcrv_rvv.q1_0_q8_0_block_dot. q1_0 uses NO libm (the only sanctioned
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
#include "q1_0_kernel.cpp"

// ---- block formats (ggml-common.h) -------------------------------------------
//   block_q1_0 = { ggml_half d; uint8_t qs[16]; }  sizeof 18, bits @2
//   block_q8_0 = { ggml_half d; int8_t  qs[32]; }  sizeof 34, quants @2
static const int QK = 128;        // QK1_0
static const int Q1_STRIDE = 18;
static const int Q8_STRIDE = 34;
static const int N_SUB = 4;        // q8_0 blocks per q1_0 super-block (QK/32)

// ---- board fp16 -> fp32 (mirrors riscv_compute_fp16_to_fp32, lossless) -------
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_q1_0_q8_0_kernel_ggml_vec_dot_q1_0_q8_0(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's _generic (quants.c:123-171, the byte-exact same math) ------------
// Parameterised by a `polarity` flag so the negative control can flip the bit
// convention (set->-q8 instead of +q8) and a `scale_mul` so it can corrupt the
// scale -- the real kernel uses polarity=+1, scale_mul=1.
static void ggml_generic_p(int n, float *s, const void *vx, const void *vy,
                           int polarity, float scale_mul) {
  const int nb = n / QK;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t *xb = x + (size_t)i * Q1_STRIDE;
    uint16_t d0h;
    std::memcpy(&d0h, xb + 0, 2);
    const float d0 = fp16_to_fp32(d0h) * scale_mul;
    float sumi = 0.0f;
    for (int k = 0; k < N_SUB; ++k) {
      const uint8_t *yb = y + (size_t)(i * 4 + k) * Q8_STRIDE;
      uint16_t d1h;
      std::memcpy(&d1h, yb + 0, 2);
      const float d1 = fp16_to_fp32(d1h);
      int sumi_block = 0;
      const uint8_t *bits = xb + 2 + k * 4;
      const int8_t *qy = (const int8_t *)(yb + 2);
      for (int b = 0; b < 4; ++b, qy += 8) {
        const unsigned mask = bits[b];
        for (int m = 0; m < 8; ++m) {
          int set = (mask & (1u << m)) != 0;
          // polarity +1 = ggml (set -> +q8); -1 = inverted (the negative control).
          int sign = (set ? polarity : -polarity);
          sumi_block += sign * (int)qy[m];
        }
      }
      sumi += d1 * (float)sumi_block;
    }
    sumf += d0 * sumi;
  }
  *s = sumf;
}

static void ggml_generic(int n, float *s, const void *vx, const void *vy) {
  ggml_generic_p(n, s, vx, vy, +1, 1.0f);
}

// ---- an INDEPENDENT RVV transcription (a different sign-plane expression) -----
// The emitter does vmerge(vneg(q8), q8, mask) over 8-lane groups. This reference
// instead builds the signed magnitude as q8 * sign_pm1 (a +1/-1 vector built by
// vmerge of a +1/-1 splat under the same mask) over the WHOLE 32-lane block in a
// single m2 strip -- a genuinely distinct expression of the same semantics.
static void ggml_rvv(int n, float *s, const void *vx, const void *vy) {
  const int nb = n / QK;
  const uint8_t *x = (const uint8_t *)vx;
  const uint8_t *y = (const uint8_t *)vy;
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    const uint8_t *xb = x + (size_t)i * Q1_STRIDE;
    uint16_t d0h;
    std::memcpy(&d0h, xb + 0, 2);
    const float d0 = fp16_to_fp32(d0h);
    float sumi = 0.0f;
    for (int k = 0; k < N_SUB; ++k) {
      const uint8_t *yb = y + (size_t)(i * 4 + k) * Q8_STRIDE;
      uint16_t d1h;
      std::memcpy(&d1h, yb + 0, 2);
      const float d1 = fp16_to_fp32(d1h);
      const uint8_t *bits = xb + 2 + k * 4;
      const int8_t *qy = (const int8_t *)(yb + 2);
      // Expand the 32 bits into a 32-lane uint8 byte vector where lane l holds
      // bits[l/8] (each byte broadcast to its 8 lanes), then & {1,2,4,..} repeated.
      uint8_t bitlane[32], kmask32[32];
      for (int b = 0; b < 4; ++b)
        for (int m = 0; m < 8; ++m) {
          bitlane[b * 8 + m] = bits[b];
          kmask32[b * 8 + m] = (uint8_t)(1u << m);
        }
      size_t vl = 32;
      vuint8m2_t blv = __riscv_vle8_v_u8m2(bitlane, vl);
      vuint8m2_t kmv = __riscv_vle8_v_u8m2(kmask32, vl);
      vbool4_t m = __riscv_vmsne_vx_u8m2_b4(__riscv_vand_vv_u8m2(blv, kmv, vl),
                                            0, vl);
      vint8m2_t q8 = __riscv_vle8_v_i8m2((const int8_t *)qy, vl);
      // sign vector: +1 where bit set, -1 where clear (the distinct expression).
      vint8m2_t plus1 = __riscv_vmv_v_x_i8m2(1, vl);
      vint8m2_t minus1 = __riscv_vmv_v_x_i8m2(-1, vl);
      vint8m2_t sgn = __riscv_vmerge_vvm_i8m2(minus1, plus1, m, vl);
      vint16m4_t prod = __riscv_vwmul_vv_i16m4(sgn, q8, vl);
      vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t red = __riscv_vwredsum_vs_i16m4_i32m1(prod, zero, vl);
      int32_t sumi_block = __riscv_vmv_x_s_i32m1_i32(red);
      sumi += d1 * (float)sumi_block;
    }
    sumf += d0 * sumi;
  }
  *s = sumf;
}

// ---- random block_q1_0 / block_q8_0 array generators -------------------------
static uint32_t rng = 0x9e3779b9u;
static uint32_t xrand() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static inline void put_fp16(uint8_t *p) {
  _Float16 d = (_Float16)(((float)(int)(xrand() % 2001) - 1000.0f) / 256.0f);
  uint16_t dh;
  std::memcpy(&dh, &d, 2);
  std::memcpy(p, &dh, 2);
}

static void fill_q1_0(uint8_t *buf, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + (size_t)i * Q1_STRIDE;
    put_fp16(b + 0);                                 // d0
    for (int j = 0; j < 16; j++)
      b[2 + j] = (uint8_t)(xrand() % 256);           // packed bit bytes
  }
}

// nb = super-blocks; the q8 stream is q8nb = 4*nb q8_0 blocks (the buffer-sizing fix).
static void fill_q8_0(uint8_t *buf, int q8nb) {
  for (int i = 0; i < q8nb; i++) {
    uint8_t *b = buf + (size_t)i * Q8_STRIDE;
    put_fp16(b + 0); // d
    for (int j = 0; j < 32; j++)
      b[2 + j] = (uint8_t)(int8_t)(xrand() % 256);
  }
}

// Edge-case fillers.
//   bmode 0: all bits 0x00 -> every sign is CLEAR -> -q8 for all 128 lanes.
//   bmode 1: all bits 0xFF -> every sign is SET   -> +q8 for all 128 lanes.
//   bmode 2: marching bytes (b | (b<<...)) so every bit position toggles.
static void fill_q1_0_edge(uint8_t *buf, int nb, int bmode) {
  for (int i = 0; i < nb; i++) {
    uint8_t *b = buf + (size_t)i * Q1_STRIDE;
    put_fp16(b + 0);
    for (int j = 0; j < 16; j++) {
      uint8_t v;
      if (bmode == 0) v = 0x00;
      else if (bmode == 1) v = 0xFF;
      else v = (uint8_t)(0xA5 ^ (j * 0x37));
      b[2 + j] = v;
    }
  }
}

static void fill_q8_0_edge(uint8_t *buf, int q8nb, int mode) {
  for (int i = 0; i < q8nb; i++) {
    uint8_t *b = buf + (size_t)i * Q8_STRIDE;
    put_fp16(b + 0);
    for (int j = 0; j < 32; j++)
      b[2 + j] = (uint8_t)(int8_t)((mode == 0) ? 127
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
  int nb = n / QK;
  int q8nb = n / 32; // = 4*nb  (the buffer-sizing fix)
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q1_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)q8nb * Q8_STRIDE);
  fill_q1_0(vx, nb);
  fill_q8_0(vy, q8nb);
  int fail = check_buffers(n, vx, vy, "random", generic_delta);
  free(vx);
  free(vy);
  return fail;
}

static int check_edge(int n, int bmode, int ymode, const char *tag,
                      int *generic_delta) {
  int nb = n / QK;
  int q8nb = n / 32;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q1_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)q8nb * Q8_STRIDE);
  fill_q1_0_edge(vx, nb, bmode);
  fill_q8_0_edge(vy, q8nb, ymode);
  int fail = check_buffers(n, vx, vy, tag, generic_delta);
  free(vx);
  free(vy);
  return fail;
}

int main() {
  int n_set[] = {128, 256, 384, 512, 640, 1024, 4096, 8192};
  int total = 0, fails = 0, generic_delta = 0;
  printf("== INC-41 q1_0 byte-exact (oracle = _generic = the real board kernel, "
         "no arch/riscv q1_0 exists) ==\n");
  for (int rep = 0; rep < 400; rep++) {
    for (int n : n_set) {
      total += 1;
      fails += check_random(n, &generic_delta);
    }
  }

  // EDGE: all-0 bits (-> all -q8), all-1 bits (-> all +q8), marching bytes, crossed
  // with q8 saturation (+127 / +/-128 / -127). These pin the BINARY sign decode at
  // its extremes (the load-bearing polarity proof: the all-0 and all-1 cases must
  // produce -Σq8 and +Σq8 respectively).
  for (int rep = 0; rep < 50; rep++) {
    for (int bmode = 0; bmode < 3; ++bmode)
      for (int ymode = 0; ymode < 3; ++ymode) {
        total += 1;
        char tag[64];
        snprintf(tag, sizeof tag, "bits=%d q8=%d", bmode, ymode);
        fails += check_edge(512, bmode, ymode, tag, &generic_delta);
      }
  }
  printf("checked %d cases, %d failures (vs ggml's RVV transcription)\n",
         total, fails);
  printf("_generic cross-check delta: %d/%d (at =off this is 0; at =fast it is the "
         "references' OWN mutual FMA-formation delta, not a kernel defect)\n",
         generic_delta, total);

  // Negative control 1 (the LOAD-BEARING BIT POLARITY proof): same input bytes, but
  // the reference INVERTS the bit convention (set -> -q8, clear -> +q8). Our kernel
  // embeds the real polarity -> it MUST diverge.
  {
    int n = 2048, nb = n / QK, q8nb = n / 32;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q1_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)q8nb * Q8_STRIDE);
    fill_q1_0(vx, nb);
    fill_q8_0_edge(vy, q8nb, 0); // q8 all +127 -> non-zero, polarity-sensitive sum
    float so = 0, s_real = 0, s_wrong = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_p(n, &s_real, vx, vy, +1, 1.0f);
    ggml_generic_p(n, &s_wrong, vx, vy, -1, 1.0f); // inverted polarity
    if (std::memcmp(&so, &s_real, 4) != 0) {
      printf("NEG-CTRL-1 setup FAILED: ours != real-polarity ref (should match)\n");
      fails += 1;
    } else if (std::memcmp(&so, &s_wrong, 4) == 0) {
      printf("NEGATIVE CONTROL 1 FAILED: WRONG (inverted) bit polarity still matched "
             "(vacuous -- the sign decode is NOT load-bearing!)\n");
      fails += 1;
    } else {
      printf("negative control 1 OK: WRONG (inverted set->-q8) polarity diverges "
             "(ours=%.9g real=%.9g inverted=%.9g) -> the bit polarity is load-bearing\n",
             so, s_real, s_wrong);
    }
    free(vx);
    free(vy);
  }

  // Negative control 2 (the LOAD-BEARING SCALE proof): same input bytes, but the
  // reference scales d0 by 2 (a wrong per-block scale). Our kernel uses the real
  // scale -> it MUST diverge.
  {
    int n = 2048, nb = n / QK, q8nb = n / 32;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q1_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)q8nb * Q8_STRIDE);
    fill_q1_0(vx, nb);
    fill_q8_0_edge(vy, q8nb, 0);
    float so = 0, s_real = 0, s_wrong = 0;
    our_kernel(n, &so, vx, vy);
    ggml_generic_p(n, &s_real, vx, vy, +1, 1.0f);
    ggml_generic_p(n, &s_wrong, vx, vy, +1, 2.0f); // wrong d0 scale (2x)
    if (std::memcmp(&so, &s_real, 4) != 0) {
      printf("NEG-CTRL-2 setup FAILED: ours != real-scale ref (should match)\n");
      fails += 1;
    } else if (std::memcmp(&so, &s_wrong, 4) == 0) {
      printf("NEGATIVE CONTROL 2 FAILED: WRONG (2x d0) scale still matched (vacuous "
             "-- the d0 scale is NOT load-bearing!)\n");
      fails += 1;
    } else {
      printf("negative control 2 OK: WRONG (2x d0) scale diverges "
             "(ours=%.9g real=%.9g wrong=%.9g) -> the d0 scale is load-bearing\n",
             so, s_real, s_wrong);
    }
    free(vx);
    free(vy);
  }

  // Negative control 3: flip ONE bit byte of vx; our kernel must DIFFER from a fresh
  // ggml run on the UNperturbed data (proves the check is non-vacuous: the bits are
  // actually consumed).
  {
    int n = 256, nb = n / QK, q8nb = n / 32;
    uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q1_STRIDE);
    uint8_t *vy = (uint8_t *)malloc((size_t)q8nb * Q8_STRIDE);
    fill_q1_0(vx, nb);
    fill_q8_0_edge(vy, q8nb, 0);
    float sg = 0;
    ggml_rvv(n, &sg, vx, vy);
    vx[2] ^= 0x01; // flip bit 0 of sub-block 0's first bit byte (@+2)
    float so = 0;
    our_kernel(n, &so, vx, vy);
    if (std::memcmp(&so, &sg, 4) == 0) {
      printf("NEGATIVE CONTROL 3 FAILED: perturbed bit still matched (vacuous)\n");
      fails += 1;
    } else {
      printf("negative control 3 OK: flipped bit diverges (the bits are consumed; "
             "check non-vacuous)\n");
    }
    free(vx);
    free(vy);
  }

  if (fails == 0)
    printf("ALL q1_0 BYTE-EXACT CHECKS PASSED\n");
  else
    printf("q1_0 BYTE-EXACT CHECKS FAILED (%d)\n", fails);
  return fails ? 1 : 0;
}
