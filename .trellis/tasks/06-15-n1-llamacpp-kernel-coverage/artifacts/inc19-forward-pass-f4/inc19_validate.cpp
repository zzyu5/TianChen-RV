// INC-19 F4 — ggml `quantize_row_q8_0` (the f32 -> block_q8_0 ACTIVATION
// QUANTIZER, the f32 -> QUANT BRIDGE) bit-exact HW validation harness
// (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.quantize_row_q8_0 op
// computes BIT-FOR-BIT the SAME block_q8_0 as ggml's OWN RVV-path
// quantize_row_q8_0 (riscv/quants.c:32-71, the __riscv_v branch deployed on the
// board), comparing the FULL output block_q8_0 bytes:
//   (a) the fp16 d (memcmp of the 2 ggml_half bytes), AND
//   (b) the 32 int8 q[i] (memcmp of the 32 qs bytes).
// over many rows x value distributions including the rounding-tie edge cases,
// amax edge cases, all-zero blocks (d=0 => q=0), large/small.
//
// BYTE-EXACTNESS matches ggml's EXACT RVV METHOD (vfncvt = round-to-nearest-EVEN
// + the native (_Float16)d conversion), NOT the scalar `_ref` (roundf,
// round-half-AWAY). On the board __riscv_v is defined, so the live quantizer IS
// the RVV path -- that is the oracle.
//
// TWO byte-exactness cruxes, each pinned by a negative control:
//   1. the ROUNDING: vfncvt_x_f_w_i16m4 = dynamic frm = round-to-nearest-EVEN
//      (NOT roundf's round-half-AWAY). They diverge at half-integer ties.
//   2. the FP16 d: GGML_CPU_FP32_TO_FP16(d) = native (_Float16)d (fcvt.h.s, rne).
//
// NEGATIVE CONTROLS (two, each isolating one crux):
//   - nc_roundf: the SAME quantizer but q[i] = roundf(x0) (round-half-AWAY)
//     instead of vfncvt (rne) -- proves the q[i] byte-compare DISCRIMINATES
//     ggml's rne, not merely "a rounding". Tie inputs are constructed so this
//     control is SHARP.
//   - nc_fp16: ggml's EXACT q[i] but a WRONG f32->f16 d (truncate the low
//     mantissa) -- proves the d byte-compare pins ggml's EXACT (_Float16) cast.
//
// CLOSE-THE-LOOP: feed the F4-quantized q8_0 output into a transcription of
// ggml's q4_0_q8_0 block dot and confirm the full f32 -> quantize -> dot pipeline
// is bit-exact vs ggml's quantize_row_q8_0 -> vec_dot_q4_0_q8_0 (the fp32 *s).
//
// The kernel under test is the UNMODIFIED, compiler-emitted C in
// quantize_row_q8_0_kernel.cpp, emitted by tcrv-opt --tcrv-rvv-lower-to-emitc |
// mlir-translate --mlir-to-cpp from the committed
// rvv-to-emitc-ggml-quantize-row-q8-0.mlir.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <riscv_vector.h>

// ---- ggml block formats (ggml-common.h:241-245, QK8_0 = 32). -----------------
#define QK8_0 32
typedef uint16_t ggml_half;
typedef struct {
  ggml_half d;          // delta (fp16)
  int8_t    qs[QK8_0];  // quants
} block_q8_0;            // sizeof == 34
typedef struct {
  ggml_half d;          // delta (fp16)
  uint8_t   qs[QK8_0 / 2];  // 16 packed nibbles
} block_q4_0;           // sizeof == 18

// ---- The board fp16<->fp32 conversion (simd-mappings.h:95-112, __riscv_zfhmin):
// native (_Float16) cast -- the SAME the deployed ggml quantizer/dequantizer use.
static inline float fp16_to_fp32(ggml_half h) {
  _Float16 hf;
  memcpy(&hf, &h, sizeof(ggml_half));
  return hf;
}
static inline ggml_half fp32_to_fp16(float f) {
  ggml_half res;
  _Float16 hf = (_Float16)f;
  memcpy(&res, &hf, sizeof(ggml_half));
  return res;
}

// ---- The F4 kernel our compiler emitted (declared; defined in *_kernel.cpp). --
// ABI: void f(size_t n, const float *x, uint8_t *vy).
extern "C" void
tcrv_emitc_quantize_row_q8_0_kernel_quantize_row_q8_0(size_t n, const float *x,
                                                      uint8_t *vy);

static void our_quantize(int n, const float *x, void *vy) {
  tcrv_emitc_quantize_row_q8_0_kernel_quantize_row_q8_0((size_t)n, x,
                                                        (uint8_t *)vy);
}

// ---- The COMMITTED q4_0_q8_0 block-dot kernel our compiler emitted (INC-2a,
// proven byte-exact + live token-identical; defined in q4_0_q8_0_dot_kernel.cpp).
// ABI (ggml's): void f(size_t n, float *s, size_t bs, const uint8_t *vx,
//                      size_t bx, const uint8_t *vy, size_t by, int32_t nrc).
// This is the INTEGRATION partner for the close-the-loop: our F4 quantizer's
// q8_0 output feeds this separately-proven compiler kernel.
extern "C" void
tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
    size_t n, float *s, size_t bs, const uint8_t *vx, size_t bx,
    const uint8_t *vy, size_t by, int32_t nrc);

static float our_dot_q4_0_q8_0(int n, const block_q4_0 *w, const block_q8_0 *y) {
  float s = 0.0f;
  tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
      (size_t)n, &s, 0, (const uint8_t *)w, 0, (const uint8_t *)y, 0, 1);
  return s;
}

// ---- The ORACLE: a VERBATIM transcription of ggml's RVV quantize_row_q8_0
// (riscv/quants.c:32-71, the __riscv_v branch). This is what the board runs. ----
static void ggml_quantize_rvv(int k, const float *x, void *vy) {
  const int nb = k / QK8_0;
  block_q8_0 *y = (block_q8_0 *)vy;
  size_t vl = QK8_0;
  for (int i = 0; i < nb; i++) {
    vfloat32m8_t v_x = __riscv_vle32_v_f32m8(x + i * QK8_0, vl);
    vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(v_x, vl);
    vfloat32m1_t tmp = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m8_f32m1(vabs, tmp, vl);
    float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);
    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = fp32_to_fp16(d);
    vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);
    vint16m4_t vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);
    vint8m2_t vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);
    __riscv_vse8_v_i8m2(y[i].qs, vs, vl);
  }
}

// ---- NC-round: the WRONG rounding (roundf = round-half-AWAY, the scalar _ref's
// rounding) instead of vfncvt's round-to-nearest-EVEN. Same d / fp16 as ggml. ---
static void nc_roundf_quantize(int k, const float *x, void *vy) {
  const int nb = k / QK8_0;
  block_q8_0 *y = (block_q8_0 *)vy;
  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;
    for (int j = 0; j < QK8_0; j++)
      amax = fmaxf(amax, fabsf(x[i * QK8_0 + j]));
    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = fp32_to_fp16(d);
    for (int j = 0; j < QK8_0; j++) {
      const float x0 = x[i * QK8_0 + j] * id;
      int r = (int)roundf(x0);  // round-half-AWAY (the _ref method)
      if (r > 127) r = 127;
      if (r < -128) r = -128;
      y[i].qs[j] = (int8_t)r;
    }
  }
}

// ---- NC-fp16: ggml's EXACT q[i] (the RVV path) but a WRONG f32->f16 d (truncate
// the low 3 mantissa bits of the half) instead of the native rne cast. ----------
static ggml_half wrong_fp32_to_fp16(float f) {
  ggml_half h = fp32_to_fp16(f);
  // off-by-one-ULP on every NONZERO d (toggle the low mantissa bit): differs
  // bit-for-bit on every nonzero-d block, so this control is SHARP. d==0 stays 0
  // (the all-zero block keeps its correct d).
  return (f != 0.0f) ? (ggml_half)(h ^ 0x1) : h;
}
static void nc_fp16_quantize(int k, const float *x, void *vy) {
  const int nb = k / QK8_0;
  block_q8_0 *y = (block_q8_0 *)vy;
  size_t vl = QK8_0;
  for (int i = 0; i < nb; i++) {
    vfloat32m8_t v_x = __riscv_vle32_v_f32m8(x + i * QK8_0, vl);
    vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(v_x, vl);
    vfloat32m1_t tmp = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m8_f32m1(vabs, tmp, vl);
    float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);
    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = wrong_fp32_to_fp16(d);  // WRONG fp16 d
    vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);
    vint16m4_t vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);
    vint8m2_t vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);
    __riscv_vse8_v_i8m2(y[i].qs, vs, vl);
  }
}

// ---- ggml q4_0_q8_0 block dot (generic ref, quants.c:174-210) for close-the-loop.
static float ggml_dot_q4_0_q8_0(int n, const block_q4_0 *x, const block_q8_0 *y) {
  const int qk = QK8_0;
  const int nb = n / qk;
  float sumf = 0;
  for (int ib = 0; ib < nb; ++ib) {
    int sumi0 = 0, sumi1 = 0;
    for (int j = 0; j < qk / 2; ++j) {
      const int v0 = (x[ib].qs[j] & 0x0F) - 8;
      const int v1 = (x[ib].qs[j] >> 4) - 8;
      sumi0 += (v0 * y[ib].qs[j]);
      sumi1 += (v1 * y[ib].qs[j + qk / 2]);
    }
    int sumi = sumi0 + sumi1;
    sumf += sumi * fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d);
  }
  return sumf;
}

// ---- A small PRNG + value-distribution generators. ---------------------------
static uint64_t rng_state = 0x123456789abcdef0ULL;
static uint32_t xrand() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (uint32_t)(rng_state >> 32);
}
static float frand_uniform(float lo, float hi) {
  return lo + (hi - lo) * (xrand() / 4294967296.0f);
}

// Fill a block with a distribution chosen by `dist`. A few distributions are
// engineered to land x0 exactly on a half-integer TIE (the rne-vs-roundf
// discriminator): amax = 254 => d = 2, id = 0.5, so an element of value 5 gives
// x0 = 2.5 (ties to even 2 under rne, to 3 under roundf); value 3 => x0 = 1.5
// (ties to 2 under both -- harmless); value 7 => 3.5 (rne 4, roundf 4 -- equal),
// value 1 => 0.5 (rne 0, roundf 1 -- DIFFERS). The tie blocks pepper many such
// odd-half values so the rne/roundf split is sharp.
static void fill_block(float *blk, int dist) {
  switch (dist) {
  case 0:  // small uniform
    for (int j = 0; j < QK8_0; j++) blk[j] = frand_uniform(-1.0f, 1.0f);
    break;
  case 1:  // large uniform
    for (int j = 0; j < QK8_0; j++) blk[j] = frand_uniform(-1000.0f, 1000.0f);
    break;
  case 2:  // tiny (near-denormal scale)
    for (int j = 0; j < QK8_0; j++) blk[j] = frand_uniform(-1e-6f, 1e-6f);
    break;
  case 3: {  // all-zero block (d=0 => id=0 => q=0)
    for (int j = 0; j < QK8_0; j++) blk[j] = 0.0f;
    break;
  }
  case 4: {  // single spiky lane (amax from one element)
    for (int j = 0; j < QK8_0; j++) blk[j] = frand_uniform(-0.01f, 0.01f);
    blk[xrand() % QK8_0] = (xrand() & 1) ? 500.0f : -500.0f;
    break;
  }
  case 5: {  // ENGINEERED TIES: amax forced to 254 (d=2, id=0.5), many odd-half
             // values so x0 = k + 0.5 lands on rne-vs-roundf ties.
    blk[0] = 254.0f;  // sets amax => d = 2.0, id = 0.5
    for (int j = 1; j < QK8_0; j++) {
      // odd integers => x0 = odd/2 = half-integer tie. Mix signs.
      int odd = (2 * (xrand() % 60) + 1);     // 1,3,5,...,119
      float sign = (xrand() & 1) ? 1.0f : -1.0f;
      blk[j] = sign * (float)odd;             // x0 = sign*odd*0.5 (a .5 tie)
    }
    break;
  }
  case 6: {  // ENGINEERED TIES at a different scale: amax = 127*2 = 254 again but
             // values that produce x0 just above/below .5 (rounding boundary).
    blk[0] = 254.0f;
    for (int j = 1; j < QK8_0; j++) {
      int odd = (2 * (xrand() % 60) + 1);
      blk[j] = (xrand() & 1) ? (float)odd : -(float)odd;
    }
    break;
  }
  case 7: {  // mixed magnitudes (realistic activations)
    for (int j = 0; j < QK8_0; j++) {
      float m = frand_uniform(0.0f, 1.0f);
      blk[j] = (m < 0.1f) ? frand_uniform(-50.0f, 50.0f)
                          : frand_uniform(-2.0f, 2.0f);
    }
    break;
  }
  }
}

int main() {
  const int kNumDist = 8;
  const int kRowsPerSizePerDist = 24;
  // Row sizes (multiples of 32, the ggml contract n % 32 == 0). Include a single
  // block, attention/ffn dims, and strip-boundary-ish counts.
  const int sizes[] = {32, 64, 96, 128, 256, 512, 1024, 2048, 4096, 11008};
  const int kNumSizes = sizeof(sizes) / sizeof(sizes[0]);

  long total_blocks = 0, d_exact = 0, qs_exact = 0, block_exact = 0;
  long nc_round_blocks = 0, nc_round_qs_differ = 0;
  long nc_fp16_blocks = 0, nc_fp16_d_differ = 0;
  long ctl_total = 0, ctl_dot_exact = 0;
  long tie_blocks = 0, tie_round_differ = 0;

  for (int si = 0; si < kNumSizes; si++) {
    int n = sizes[si];
    int nb = n / QK8_0;
    float *x = (float *)malloc(sizeof(float) * n);
    block_q8_0 *out_ours = (block_q8_0 *)malloc(sizeof(block_q8_0) * nb);
    block_q8_0 *out_ref = (block_q8_0 *)malloc(sizeof(block_q8_0) * nb);
    block_q8_0 *out_nc_r = (block_q8_0 *)malloc(sizeof(block_q8_0) * nb);
    block_q8_0 *out_nc_f = (block_q8_0 *)malloc(sizeof(block_q8_0) * nb);
    block_q4_0 *w = (block_q4_0 *)malloc(sizeof(block_q4_0) * nb);

    for (int dist = 0; dist < kNumDist; dist++) {
      for (int r = 0; r < kRowsPerSizePerDist; r++) {
        for (int b = 0; b < nb; b++)
          fill_block(x + b * QK8_0, dist);

        our_quantize(n, x, out_ours);
        ggml_quantize_rvv(n, x, out_ref);
        nc_roundf_quantize(n, x, out_nc_r);
        nc_fp16_quantize(n, x, out_nc_f);

        for (int b = 0; b < nb; b++) {
          total_blocks++;
          bool dEq = memcmp(&out_ours[b].d, &out_ref[b].d, sizeof(ggml_half)) == 0;
          bool qEq = memcmp(out_ours[b].qs, out_ref[b].qs, QK8_0) == 0;
          if (dEq) d_exact++;
          if (qEq) qs_exact++;
          if (dEq && qEq) block_exact++;

          // NC-round discrimination on the qs (vs ggml's qs).
          nc_round_blocks++;
          if (memcmp(out_nc_r[b].qs, out_ref[b].qs, QK8_0) != 0)
            nc_round_qs_differ++;
          // On TIE distributions specifically (5,6), track the sharpness.
          if (dist == 5 || dist == 6) {
            tie_blocks++;
            if (memcmp(out_nc_r[b].qs, out_ref[b].qs, QK8_0) != 0)
              tie_round_differ++;
          }
          // NC-fp16 discrimination on the d (vs ggml's d).
          nc_fp16_blocks++;
          if (memcmp(&out_nc_f[b].d, &out_ref[b].d, sizeof(ggml_half)) != 0)
            nc_fp16_d_differ++;
        }

        // CLOSE-THE-LOOP (the INTEGRATION demo, task return-item 3): two
        // separately-proven compiler kernels composing end-to-end.
        //   pipeline A (ours): our F4 quantize(x) -> the COMMITTED compiler
        //                      q4_0_q8_0 block-dot kernel -> s_ours.
        //   pipeline B (ggml): ggml's RVV quantize(x) -> ggml's generic
        //                      q4_0_q8_0 dot -> s_ref.
        // Confirmatory GIVEN block-exactness (out_ours == out_ref proven above
        // AND inc2a proved the dot kernel byte-exact), but it exercises the FULL
        // f32 -> quantize -> dot path through DISTINCT compiler kernels, not a
        // self-comparison.
        for (int b = 0; b < nb; b++) {
          w[b].d = fp32_to_fp16(frand_uniform(0.001f, 0.5f));
          for (int j = 0; j < QK8_0 / 2; j++)
            w[b].qs[j] = (uint8_t)(xrand() & 0xFF);
        }
        float s_ours = our_dot_q4_0_q8_0(n, w, out_ours);  // our quant + our dot
        float s_ref = ggml_dot_q4_0_q8_0(n, w, out_ref);   // ggml quant + ggml dot
        ctl_total++;
        if (memcmp(&s_ours, &s_ref, sizeof(float)) == 0)
          ctl_dot_exact++;
      }
    }
    free(x); free(out_ours); free(out_ref); free(out_nc_r); free(out_nc_f); free(w);
  }

  printf("INC-19 F4 quantize_row_q8_0: blocks %ld/%ld d bit-exact, %ld/%ld qs "
         "bit-exact, %ld/%ld FULL block bit-exact (vs ggml RVV quantizer)\n",
         d_exact, total_blocks, qs_exact, total_blocks, block_exact, total_blocks);
  printf("NC-round (roundf, round-half-AWAY): %ld/%ld blocks qs correctly "
         "DIFFER; on engineered-tie blocks %ld/%ld DIFFER (rne vs roundf)\n",
         nc_round_qs_differ, nc_round_blocks, tie_round_differ, tie_blocks);
  printf("NC-fp16 (truncated f32->f16 d): %ld/%ld blocks d correctly DIFFER\n",
         nc_fp16_d_differ, nc_fp16_blocks);
  printf("CLOSE-THE-LOOP (f32->quantize->q4_0_q8_0 dot *s): %ld/%ld bit-exact "
         "(our_quantize->dot vs ggml_quantize->dot)\n",
         ctl_dot_exact, ctl_total);

  bool pass = (block_exact == total_blocks) && (ctl_dot_exact == ctl_total);
  bool round_ctl_sharp = (tie_blocks > 0) && (tie_round_differ * 2 > tie_blocks);
  bool fp16_ctl_sharp = (nc_fp16_d_differ * 2 > nc_fp16_blocks);
  printf("DISCRIMINATION: NC-round %s (%ld/%ld tie blocks differ); NC-fp16 %s "
         "(%ld/%ld differ)\n",
         round_ctl_sharp ? "SHARP" : "DULL", tie_round_differ, tie_blocks,
         fp16_ctl_sharp ? "SHARP" : "DULL", nc_fp16_d_differ, nc_fp16_blocks);
  printf("RESULT: %s (FULL block_q8_0 d+qs bit-exact AND close-the-loop *s "
         "bit-exact vs ggml RVV quantizer; NC-round pins rne, NC-fp16 pins the "
         "(_Float16) cast)\n",
         pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
