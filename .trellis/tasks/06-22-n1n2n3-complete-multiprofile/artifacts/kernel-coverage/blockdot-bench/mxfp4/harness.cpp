// Coverage harness: OUR emitted mxfp4_q8_0 block-dot vs ggml's shipped RVV _vl128.
// mxfp4 = FP4 e2m1 codebook gather. block_mxfp4=17B (e@0 E8M0-byte, qs[16]@1), q8_0=34B.
// ours.cpp uses split lo/hi 16-lane vrgather_vv_i8m1 over a 16-entry register codebook
// (== the iq4_nl mechanism); ggml _vl128 uses a single 32-lane vrgather_vv_i8m2,
// 2 blocks/iter. We force EVEN nb so ggml's 2-at-a-time covers identical blocks.
// E8M0 exponent byte is constrained to [120,134] so 2^(e-128) stays O(1) and products
// are finite -- an honest fill caveat (the gather/codebook core is exercised in full).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <riscv_vector.h>

#define QK 32
#define WB 17
#define YB 34
#define OURS tcrv_emitc_ggml_vec_dot_mxfp4_q8_0_kernel_ggml_vec_dot_mxfp4_q8_0

extern "C" void kern_ggml(size_t, float *, const uint8_t *, const uint8_t *);
extern "C" void OURS(size_t, float *, const uint8_t *, const uint8_t *);

static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t rfp16() {
  uint16_t s = (xr() & 1) << 15, e = (uint16_t)(xr() % 31), m = (uint16_t)(xr() & 0x3FF);
  return s | (e << 10) | m;
}
// boundary_scales: when true (agreement run), MIX IN the genuinely-new E8M0 decode
// branches the test IR flagged: x<2 denormal patterns (0x00200000u<<x) and small e.
// We avoid LARGE e (overflow->inf->nan rel) but DO cover the x<2 path. When false
// (timing run), keep e in [120,134] so 2^(e-128)~O(1) and products are finite.
static void fill(uint8_t *vx, uint8_t *vy, int nb, int boundary_scales) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * WB, *yb = vy + (size_t)ib * YB;
    for (int i = 0; i < WB; ++i) xb[i] = (uint8_t)(xr() & 0xFF);
    for (int i = 0; i < YB; ++i) yb[i] = (uint8_t)(xr() & 0xFF);
    if (boundary_scales) {
      // hit x<2 denormal branch (e in {0,1}) + a band of moderate finite exponents
      uint32_t r = xr() % 5;
      if (r == 0) xb[0] = 0;                          // x<2 denormal (bits=0x00200000)
      else if (r == 1) xb[0] = 1;                     // x<2 denormal (bits=0x00400000)
      else xb[0] = (uint8_t)(118 + (xr() % 19));      // e in [118,136] -> 2^(e-128) finite
    } else {
      xb[0] = (uint8_t)(120 + (xr() % 15));           // timing: 2^(e-128) ~ O(1)
    }
    uint16_t d = rfp16(); memcpy(yb + 0, &d, 2);      // y.d
  }
}
typedef void (*kfn)(size_t, float *, const uint8_t *, const uint8_t *);
static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double tb(kfn fn, size_t n, int it, const uint8_t *vx, const uint8_t *vy) {
  double t0 = now(); float s; for (int i = 0; i < it; i++) fn(n, &s, vx, vy); return (now() - t0) / it;
}
int main() {
  int agn[] = { 2*QK, 4*QK, 8*QK, 16*QK, 32*QK, 64*QK };
  const int NN = sizeof(agn)/sizeof(agn[0]);
  double max_rel = 0; int any_nonzero = 0, any_nonfinite = 0;
  for (int seed = 1; seed <= 8; ++seed) {
    rng = 0x2468ace0u ^ (seed * 0x9e3779b9u);
    for (int j = 0; j < NN; ++j) {
      int n = agn[j], nb = n / QK;
      uint8_t *vx = (uint8_t *)malloc((size_t)nb * WB), *vy = (uint8_t *)malloc((size_t)nb * YB);
      fill(vx, vy, nb, /*boundary_scales=*/1);
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
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * WB), *vy = (uint8_t *)malloc((size_t)nb * YB);
  rng = 0x13572468u; fill(vx, vy, nb, /*boundary_scales=*/0);
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
