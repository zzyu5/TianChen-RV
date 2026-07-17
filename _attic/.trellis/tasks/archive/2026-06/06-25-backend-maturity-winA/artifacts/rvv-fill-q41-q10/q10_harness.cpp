// q1_0 block-dot single-kernel coverage: OUR emitted q1_0_q8_0 block-dot vs ggml's
// shipped RVV q1_0_q8_0 kernel (_vl128 path, taken on VLEN128). q1_0 is a SUPER-BLOCK
// format: block_q1_0 = {fp16 d; uint8 qs[16]} (18B, QK1_0=128), ONE super-block spans
// FOUR block_q8_0 activation blocks (34B each, QK=32). Binary {-1,+1} sign decode:
// weight bit set -> +q8, clear -> -q8. Identical random super-block bytes fed to both.
// Agreement via relative norm; timing best-of-reps min, taskset-pinned externally.
//
// NOTE (byte-exact discipline): ggml _vl128 negates in the i8 domain (vneg_i8m2), so
// vneg(-128)=-128 (overflow); OUR emit widens to i16 FIRST then negates (correct +128).
// They diverge ONLY when a q8 quant byte == -128, which NEVER occurs in genuine q8_0
// (quant domain is [-127,127]). We therefore constrain the q8 quant fill to [-127,127]
// (the real q8_0 domain) -> both agree byte-exact. The -128 super-boundary is a
// maturity note (we are MORE correct there), not a gate failure.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <riscv_vector.h>

#define QK1_0 128
#define WB 18      // block_q1_0 bytes: d@0, qs[16]@2
#define YB 34      // block_q8_0 bytes: d@0, qs[32]@2
#define YPER 4     // q8_0 blocks per q1_0 super-block

#define OURS tcrv_emitc_ggml_vec_dot_q1_0_q8_0_kernel_ggml_vec_dot_q1_0_q8_0
extern "C" void kern_ggml(size_t, float *, const uint8_t *, const uint8_t *);
extern "C" void OURS(size_t, float *, const uint8_t *, const uint8_t *);

static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t rfp16() { // finite fp16, e<31 (no inf/nan)
  uint16_t s = (xr() & 1) << 15, e = (uint16_t)(xr() % 31), m = (uint16_t)(xr() & 0x3FF);
  return s | (e << 10) | m;
}
// nb = number of q1_0 super-blocks. vx: nb*18B weights. vy: nb*4*34B activations.
// q8 quant bytes constrained to [-127,127] (real q8_0 domain; -128 never occurs).
static void fill(uint8_t *vx, uint8_t *vy, int nb) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * WB;
    for (int i = 0; i < WB; ++i) xb[i] = (uint8_t)(xr() & 0xFF); // random sign bits
    uint16_t d = rfp16(); memcpy(xb + 0, &d, 2);                 // x.d fp16
    for (int k = 0; k < YPER; ++k) {
      uint8_t *yb = vy + ((size_t)ib * YPER + k) * YB;
      d = rfp16(); memcpy(yb + 0, &d, 2);                        // y.d fp16
      for (int j = 0; j < 32; ++j) {
        int q = (int)(xr() % 255) - 127;                         // [-127,127]
        yb[2 + j] = (uint8_t)(int8_t)q;
      }
    }
  }
}
typedef void (*kfn)(size_t, float *, const uint8_t *, const uint8_t *);
static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double tb(kfn fn, size_t n, int it, const uint8_t *vx, const uint8_t *vy) {
  double t0 = now(); float s; for (int i = 0; i < it; i++) fn(n, &s, vx, vy); return (now() - t0) / it;
}
int main() {
  int agn[] = { QK1_0, 2*QK1_0, 4*QK1_0, 8*QK1_0, 16*QK1_0, 64*QK1_0 };
  const int NN = sizeof(agn)/sizeof(agn[0]);
  double max_rel = 0; int any_nonzero = 0, any_nonfinite = 0;
  for (int seed = 1; seed <= 8; ++seed) {
    rng = 0x2468ace0u ^ (seed * 0x9e3779b9u);
    for (int j = 0; j < NN; ++j) {
      int n = agn[j], nb = n / QK1_0;
      uint8_t *vx = (uint8_t *)malloc((size_t)nb * WB);
      uint8_t *vy = (uint8_t *)malloc((size_t)nb * YPER * YB);
      fill(vx, vy, nb);
      float rg = 0, ro = 0;
      kern_ggml(n, &rg, vx, vy);
      OURS(n, &ro, vx, vy);
      if (!isfinite(rg) || !isfinite(ro)) any_nonfinite = 1;
      if (rg != 0.0f) any_nonzero = 1;
      double denom = fabs((double)rg); if (denom < 1e-6) denom = 1e-6;
      double rel = fabs((double)ro - (double)rg) / denom;
      if (rel > max_rel) max_rel = rel;
      free(vx); free(vy);
    }
  }
  printf("AGREEMENT max_rel_norm=%.3e nonzero=%d nonfinite=%d\n", max_rel, any_nonzero, any_nonfinite);

  const int timing_n = 64 * QK1_0, iters = 4000, reps = 200;
  int nb = timing_n / QK1_0;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * WB);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * YPER * YB);
  rng = 0x13572468u; fill(vx, vy, nb);
  for (int i = 0; i < iters; i++) { float s; kern_ggml(timing_n, &s, vx, vy); OURS(timing_n, &s, vx, vy); }
  double ggml_best = 1e18, ours_best = 1e18;
  for (int r = 0; r < reps; r++) {
    double pg = tb(kern_ggml, timing_n, iters, vx, vy); if (pg < ggml_best) ggml_best = pg;
    double po = tb(OURS, timing_n, iters, vx, vy); if (po < ours_best) ours_best = po;
  }
  printf("RESULT ours %.1f\n", ours_best);
  printf("RESULT ggml(real,vl128) %.1f\n", ggml_best);
  printf("RATIO ggml/ours %.3f\n", ggml_best / ours_best);
  free(vx); free(vy);
  return 0;
}
