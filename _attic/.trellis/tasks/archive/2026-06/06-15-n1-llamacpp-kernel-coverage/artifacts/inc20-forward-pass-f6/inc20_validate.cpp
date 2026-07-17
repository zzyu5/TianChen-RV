// INC-20 F6 — ggml NORMAL rope (ggml_compute_forward_rope_f32, the
// GGML_ROPE_TYPE_NORMAL variant llama-2 / LLM_ARCH_LLAMA uses) byte-exact HW
// validation harness (ssh rvv, -march=rv64gcv, -ffp-contract=off).
//
// Proves: the C our compiler emits for the new tcrv_rvv.ggml_rope_norm_f32 op
// computes BIT-FOR-BIT the SAME rotated f32 output as ggml's OWN NORMAL rope
// (ops.cpp: ggml_rope_cache_init + rotate_pairs<float> with n_offset=1,
// scale=1), over realistic head rows (head_dim 128) x positions (0 .. large) x
// value distributions, the plain path (ext_factor=0, freq_scale=1,
// attn_factor=1, freq_factors=NULL, forward => sin_sign=+1, n_dims=ne0).
//
// TWO byte-exactness axes, both surfaced honestly:
//   1. The angle transcendental cosf/sinf is SCALAR libm. Linking the SAME libm
//      for BOTH our kernel and the ggml reference (this single TU links libm
//      once) makes the angle cache bit-identical -- so the comparison is
//      apples-to-apples and EXACT, not tolerance.  theta_base (= pos as f32) and
//      theta_scale (= powf(freq_base, -2/n_dims)) are precomputed ONCE on the
//      host and passed identically to BOTH paths, so there is no powf divergence
//      either.
//   2. The rotation x0*cos - x1*sin is a*b - c*d, an FP-contraction hazard. Our
//      emitted kernel renders it as ONE C statement (the grouped emitc.expression)
//      TOKEN-IDENTICAL to ggml's source single expression, so clang makes the
//      IDENTICAL contraction decision under EVERY -ffp-contract mode -- byte-exact
//      under default / on / off / fast (this TU is built once per flag, so the
//      reference exercises ggml's REAL behavior at that flag). NOT off-only.
//
// NEGATIVE CONTROLS (proving the compare is a real discriminator, not vacuous):
//   - nc_neox: the SAME math but the NEOX pairing (x0=x[i], x1=x[i+n_dims/2])
//     instead of NORMAL's consecutive pair -- proves the f32 byte-compare
//     DISCRIMINATES the pairing/variant (matching ggml's NORMAL indexing is
//     load-bearing). Flag-ROBUST: differs under every contraction mode.
//   - nc_fma: the rotation FORCED unfused via fmaf with the WRONG sign discipline
//     would not isolate cleanly across flags, so the primary discriminator here is
//     nc_neox; nc_fma is reported for context (it differs when the reference is
//     unfused, i.e. at -ffp-contract=off, confirming the rotation form matters).
//
// The kernel under test is the UNMODIFIED, compiler-emitted C in
// ggml_rope_norm_f32_kernel.cpp (tcrv-opt --tcrv-rvv-lower-to-emitc |
// mlir-translate --mlir-to-cpp from the committed
// rvv-to-emitc-ggml-rope-norm-f32.mlir).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <riscv_vector.h>

// ---- The UNMODIFIED compiler-emitted kernel under test. ----------------------
#include "ggml_rope_norm_f32_kernel.cpp"
// extern "C" void tcrv_emitc_ggml_rope_norm_f32_kernel_ggml_rope_norm_f32(
//     size_t n_dims, const float *x, float *y, float theta_base,
//     float theta_scale);

// ---- ggml's EXACT NORMAL rope reference (verbatim transcription). ------------
// ggml_rope_cache_init (ops.cpp:5707-5721), the PLAIN path (ext_factor=0 =>
// rope_yarn returns cosf(theta)*mscale / sinf(theta)*mscale with mscale=1 and
// theta = freq_scale*theta_extrap = theta with freq_scale=1; freq_factors=NULL
// => ff=1; sin_sign=+1). The recurrence is the iterative f32 `theta *=
// theta_scale`, theta seeded at theta_base (= pos as f32).
static void ggml_rope_cache_init_ref(float theta_base, float theta_scale,
                                     int64_t ne0, float *cache) {
  float theta = theta_base;
  for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
    cache[i0 + 0] = cosf(theta);
    cache[i0 + 1] = sinf(theta);
    theta *= theta_scale;
  }
}

// rotate_pairs<float>(n_dims, n_offset=1, cache, src, dst, scale=1) — the NORMAL
// variant: ic = i0/scale = i0, so x0 = src[i0], x1 = src[i0+1] (CONSECUTIVE),
// cos = cache[i0], sin = cache[i0+1]. The rotation is the two-rounding
// mul;mul;sub / mul;mul;add f32 form (ops.cpp:5808-5809). Compiled in this TU at
// -ffp-contract=off, so it matches the deployed reference.
static void ggml_rope_norm_ref(int64_t n_dims, const float *x, float *y,
                               float theta_base, float theta_scale,
                               float *cache) {
  ggml_rope_cache_init_ref(theta_base, theta_scale, n_dims, cache);
  for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
    const float cos_theta = cache[i0 + 0];
    const float sin_theta = cache[i0 + 1];
    const float x0 = x[i0 + 0];
    const float x1 = x[i0 + 1];
    y[i0 + 0] = x0 * cos_theta - x1 * sin_theta;
    y[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
  }
}

// ---- NEGATIVE CONTROL 1: the FUSED (FMA) rotation. ---------------------------
// One rounding per a*b - c*d via fmaf instead of the two-rounding mul;mul;sub.
// This is what -ffp-contract=fast would produce; it MUST differ from ggml's
// two-rounding form on enough inputs to prove the byte-compare discriminates.
static void nc_fma_rope_ref(int64_t n_dims, const float *x, float *y,
                            float theta_base, float theta_scale, float *cache) {
  ggml_rope_cache_init_ref(theta_base, theta_scale, n_dims, cache);
  for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
    const float cos_theta = cache[i0 + 0];
    const float sin_theta = cache[i0 + 1];
    const float x0 = x[i0 + 0];
    const float x1 = x[i0 + 1];
    // y0 = x0*cos - x1*sin  ==  fmaf(x0, cos, -(x1*sin))  (single rounding).
    y[i0 + 0] = fmaf(x0, cos_theta, -(x1 * sin_theta));
    y[i0 + 1] = fmaf(x0, sin_theta, x1 * cos_theta);
  }
}

// ---- NEGATIVE CONTROL 2: the NEOX pairing (wrong variant). -------------------
// Same theta/cos/sin but x0=x[i], x1=x[i+n_dims/2] (split-half), cache indexed by
// the pair count -- proves matching NORMAL's consecutive pairing is load-bearing.
static void nc_neox_rope_ref(int64_t n_dims, const float *x, float *y,
                             float theta_base, float theta_scale, float *cache) {
  ggml_rope_cache_init_ref(theta_base, theta_scale, n_dims, cache);
  const int64_t half = n_dims / 2;
  for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
    const float cos_theta = cache[i0 + 0];
    const float sin_theta = cache[i0 + 1];
    const int64_t ic = i0 / 2;
    const float x0 = x[ic];
    const float x1 = x[ic + half];
    y[ic]        = x0 * cos_theta - x1 * sin_theta;
    y[ic + half] = x0 * sin_theta + x1 * cos_theta;
  }
}

// ---- Test driver. ------------------------------------------------------------
static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main() {
  const int64_t HEAD_DIM = 128;                 // llama-2 7b head_dim = n_rot.
  const float FREQ_BASE = 10000.0f;             // the llama-2 rope base.
  const float theta_scale = powf(FREQ_BASE, -2.0f / (float)HEAD_DIM);

  // A spread of positions including the required edge cases: pos 0 (theta=0 =>
  // cos=1,sin=0, identity), small, mid, and LARGE pos (libm range reduction).
  const int positions[] = {0, 1, 2, 7, 31, 128, 511, 2047, 4095, 8191, 32767};
  const int n_positions = sizeof(positions) / sizeof(positions[0]);

  // Several value distributions for x[] per (position) — including a tie-prone
  // grid so the nc_fma control has inputs where fused vs unfused diverge.
  const int n_dists = 6;

  long total = 0, ours_exact = 0;
  long nc_fma_total = 0, nc_fma_diff = 0;
  long nc_neox_total = 0, nc_neox_diff = 0;

  float *x = (float *)malloc(sizeof(float) * HEAD_DIM);
  float *y_ours = (float *)malloc(sizeof(float) * HEAD_DIM);
  float *y_ref = (float *)malloc(sizeof(float) * HEAD_DIM);
  float *y_nc = (float *)malloc(sizeof(float) * HEAD_DIM);
  float *cache = (float *)malloc(sizeof(float) * HEAD_DIM);

  unsigned long seed = 0x1234567u;
  auto rnd = [&]() {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((seed >> 11) & 0xFFFFF) / 524288.0f - 1.0f; // [-1,1)
  };

  for (int pi = 0; pi < n_positions; ++pi) {
    const float theta_base = (float)positions[pi];
    for (int d = 0; d < n_dists; ++d) {
      for (int64_t i = 0; i < HEAD_DIM; ++i) {
        switch (d) {
          case 0: x[i] = rnd(); break;                       // small uniform
          case 1: x[i] = rnd() * 100.0f; break;              // large
          case 2: x[i] = rnd() * 0.001f; break;              // tiny
          case 3: x[i] = (i % 2 == 0) ? 1.0f : -1.0f; break; // ±1 grid
          case 4: x[i] = (float)(i) - 64.0f; break;          // integer ramp
          case 5: x[i] = rnd() * (1.0f + (float)i); break;   // mixed magnitude
        }
      }

      // Our compiler-emitted kernel (n_dims = HEAD_DIM).
      tcrv_emitc_ggml_rope_norm_f32_kernel_ggml_rope_norm_f32(
          (size_t)HEAD_DIM, x, y_ours, theta_base, theta_scale);
      // ggml's exact NORMAL rope reference.
      ggml_rope_norm_ref(HEAD_DIM, x, y_ref, theta_base, theta_scale, cache);

      total++;
      if (memcmp(y_ours, y_ref, sizeof(float) * HEAD_DIM) == 0) ours_exact++;
      else {
        // first divergent lane, for diagnosis
        for (int64_t i = 0; i < HEAD_DIM; ++i)
          if (bits(y_ours[i]) != bits(y_ref[i])) {
            fprintf(stderr,
                    "MISMATCH pos=%d dist=%d lane=%lld ours=%08x ref=%08x\n",
                    positions[pi], d, (long long)i, bits(y_ours[i]),
                    bits(y_ref[i]));
            break;
          }
      }

      // NC-fma: fused rotation vs ggml's two-rounding (vs the SAME ref).
      nc_fma_rope_ref(HEAD_DIM, x, y_nc, theta_base, theta_scale, cache);
      nc_fma_total++;
      if (memcmp(y_nc, y_ref, sizeof(float) * HEAD_DIM) != 0) nc_fma_diff++;

      // NC-neox: wrong pairing vs ggml's NORMAL pairing.
      nc_neox_rope_ref(HEAD_DIM, x, y_nc, theta_base, theta_scale, cache);
      nc_neox_total++;
      if (memcmp(y_nc, y_ref, sizeof(float) * HEAD_DIM) != 0) nc_neox_diff++;
    }
  }

  printf("INC-20 F6 ggml_rope_norm_f32 (NORMAL, head_dim=%lld, base=%.1f): "
         "%ld/%ld rows f32 BIT-EXACT (vs ggml NORMAL rope, same-libm; this TU's "
         "-ffp-contract mode)\n",
         (long long)HEAD_DIM, FREQ_BASE, ours_exact, total);
  printf("NC-neox (split-half pairing): %ld/%ld rows correctly DIFFER (pins the "
         "NORMAL consecutive pairing; flag-robust)\n", nc_neox_diff, nc_neox_total);
  printf("NC-fma (forced-fused rotation): %ld/%ld rows DIFFER vs ggml ref "
         "(context: nonzero when the ref is unfused, e.g. -ffp-contract=off)\n",
         nc_fma_diff, nc_fma_total);
  // The kernel verdict gates on the drop-in test (ours == ggml ref over all rows)
  // and the flag-ROBUST discriminator NC-neox being sharp. NC-fma is context.
  bool pass = (ours_exact == total) && (nc_neox_diff > 0);
  printf("DISCRIMINATION: NC-neox %s (flag-robust pairing discriminator)\n",
         nc_neox_diff > 0 ? "SHARP" : "VACUOUS");
  printf("RESULT: %s (f32 output BIT-EXACT vs ggml NORMAL rope at this "
         "contraction mode; NC-neox pins the NORMAL pairing)\n",
         pass ? "PASS" : "FAIL");

  free(x); free(y_ours); free(y_ref); free(y_nc); free(cache);
  return pass ? 0 : 1;
}
