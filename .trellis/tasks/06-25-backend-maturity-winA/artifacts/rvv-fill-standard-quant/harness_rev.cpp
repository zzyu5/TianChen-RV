// Parameterized block-dot single-kernel coverage harness: OUR emitted
// <quant>_q8_<n> block-dot vs ggml's shipped RVV vec_dot kernel (VLEN128 path).
// Identical random super-block bytes fed to both. Agreement via relative norm
// max|ours-ggml|/|ggml| over seeds x sizes; timing best-of-reps min latency,
// taskset-pinned externally. All quant-specific layout via -D macros:
//   QK  = elements per block (32 for all standard quants)
//   HWB  = weight block bytes  (q4_0=18 q4_1=20 q5_0=22 q5_1=24 q8_0=34)
//   HYB  = activation block bytes (q8_0=34 q8_1=36)
//   XMIN_OFF = byte offset of x 2nd fp16 scale (m), or -1 if none
//   YSUM_OFF = byte offset of y 2nd fp16 scale (s), or -1 if none
//   OURS = exported symbol of our emitted kernel
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <riscv_vector.h>

#ifndef QK
#error "define QK HWB HYB XMIN_OFF YSUM_OFF OURS"
#endif

extern "C" void kern_ggml(size_t, float *, const uint8_t *, const uint8_t *);
extern "C" void OURS(size_t, float *, const uint8_t *, const uint8_t *);

static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t rfp16() { // finite fp16, e<31 (no inf/nan)
  uint16_t s = (xr() & 1) << 15, e = (uint16_t)(xr() % 31), m = (uint16_t)(xr() & 0x3FF);
  return s | (e << 10) | m;
}
// Every byte random, then overlay finite fp16 into the scale fields at known
// offsets. q8_0 quant bytes (the activation int8 quants at +Q8DATA) constrained
// to [-127,127] is unnecessary here (no in-domain negate overflow path used by
// these standard kernels), so we leave all data bytes random. Both kernels read
// identical bytes -> agreement is meaningful regardless of physical consistency.
static void fill(uint8_t *vx, uint8_t *vy, int nb) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * HWB, *yb = vy + (size_t)ib * HYB;
    for (int i = 0; i < HWB; ++i) xb[i] = (uint8_t)(xr() & 0xFF);
    for (int i = 0; i < HYB; ++i) yb[i] = (uint8_t)(xr() & 0xFF);
    uint16_t d;
    d = rfp16(); memcpy(xb + 0, &d, 2);                          // x.d (fp16)
    if (XMIN_OFF >= 0) { d = rfp16(); memcpy(xb + XMIN_OFF, &d, 2); } // x.m
    d = rfp16(); memcpy(yb + 0, &d, 2);                          // y.d (fp16)
    if (YSUM_OFF >= 0) { d = rfp16(); memcpy(yb + YSUM_OFF, &d, 2); } // y.s
  }
}
typedef void (*kfn)(size_t, float *, const uint8_t *, const uint8_t *);
static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double tb(kfn fn, size_t n, int it, const uint8_t *vx, const uint8_t *vy) {
  double t0 = now(); float s; for (int i = 0; i < it; i++) fn(n, &s, vx, vy); return (now() - t0) / it;
}
int main() {
  int agn[] = { QK, 2*QK, 4*QK, 8*QK, 16*QK, 64*QK };
  const int NN = sizeof(agn)/sizeof(agn[0]);
  double max_rel = 0; int any_nonzero = 0, any_nonfinite = 0;
  for (int seed = 1; seed <= 8; ++seed) {
    rng = 0x2468ace0u ^ (seed * 0x9e3779b9u);
    for (int j = 0; j < NN; ++j) {
      int n = agn[j], nb = n / QK;
      uint8_t *vx = (uint8_t *)malloc((size_t)nb * HWB), *vy = (uint8_t *)malloc((size_t)nb * HYB);
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

  const int timing_n = 64 * QK, iters = 4000, reps = 200;
  int nb = timing_n / QK;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * HWB), *vy = (uint8_t *)malloc((size_t)nb * HYB);
  rng = 0x13572468u; fill(vx, vy, nb);
  for (int i = 0; i < iters; i++) { float s; kern_ggml(timing_n, &s, vx, vy); OURS(timing_n, &s, vx, vy); }
  double ggml_best = 1e18, ours_best = 1e18;
  for (int r = 0; r < reps; r++) {
    double po = tb(OURS, timing_n, iters, vx, vy); if (po < ours_best) ours_best = po;
    double pg = tb(kern_ggml, timing_n, iters, vx, vy); if (pg < ggml_best) ggml_best = pg;
  }
  printf("RESULT ours %.1f\n", ours_best);
  printf("RESULT ggml(real,vl128) %.1f\n", ggml_best);
  printf("RATIO ggml/ours %.3f\n", ggml_best / ours_best);
  free(vx); free(vy);
  return 0;
}
