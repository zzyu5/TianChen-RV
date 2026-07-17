// Coverage harness: OUR emitted iq2_s_q8_K block-dot vs ggml's shipped RVV _vl128.
// iq2_s = 2-bit grid-codebook (gather via iq2xxs_grid + sign-grid). One super-block
// at a time (no 2-super-block stride), inner ib32+=2 over 8 sub-blocks -> any multiple
// of QK_K=256 is apples-to-apples (NO tail-drop caveat).
// block_iq2_s=66B (d@0 fp16, qs[32]u16@2). block_q8_K=292B (d@0 float32,
// qs[256]int8@4, bsums[16]int16@260). Both kernels read identical bytes.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define QK_K 256
#define WB 82
#define YB 292
#define OURS tcrv_emitc_ggml_vec_dot_iq2_s_q8_K_kernel_ggml_vec_dot_iq2_s_q8_K

extern "C" void kern_ggml(size_t, float *, const uint8_t *, const uint8_t *);
extern "C" void OURS(size_t, float *, const uint8_t *, const uint8_t *);

static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t rfp16() {
  uint16_t s = (xr() & 1) << 15, e = (uint16_t)(xr() % 31), m = (uint16_t)(xr() & 0x3FF);
  return s | (e << 10) | m;
}
// Weight (iq2_s): qs[64]@2 fully random (grid+sign-packed bits); d@0 finite fp16.
// Activation (q8_K): d@0 modest finite float32; qs[256]@4 int8; bsums[16]@260 consistent.
static void fill(uint8_t *vx, uint8_t *vy, int nb) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * WB, *yb = vy + (size_t)ib * YB;
    for (int i = 0; i < WB; ++i) xb[i] = (uint8_t)(xr() & 0xFF);
    uint16_t d = rfp16(); memcpy(xb + 0, &d, 2);              // x.d fp16
    float yd = 0.01f + (float)(xr() % 1000) / 1000.0f;
    memcpy(yb + 0, &yd, 4);                                    // y.d float32
    int8_t q8[256];
    for (int i = 0; i < 256; ++i) { q8[i] = (int8_t)((xr() & 0xFF) - 128) / 4; yb[4 + i] = (uint8_t)q8[i]; }
    for (int g = 0; g < 16; ++g) {
      int16_t bs = 0; for (int k = 0; k < 16; ++k) bs += q8[g*16 + k];
      memcpy(yb + 260 + g*2, &bs, 2);
    }
  }
}
typedef void (*kfn)(size_t, float *, const uint8_t *, const uint8_t *);
static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double tb(kfn fn, size_t n, int it, const uint8_t *vx, const uint8_t *vy) {
  double t0 = now(); float s; for (int i = 0; i < it; i++) fn(n, &s, vx, vy); return (now() - t0) / it;
}
int main() {
  int agn[] = { QK_K, 2*QK_K, 4*QK_K, 8*QK_K, 16*QK_K, 32*QK_K };
  const int NN = sizeof(agn)/sizeof(agn[0]);
  double max_rel = 0; int any_nonzero = 0, any_nonfinite = 0;
  for (int seed = 1; seed <= 8; ++seed) {
    rng = 0x2468ace0u ^ (seed * 0x9e3779b9u);
    for (int j = 0; j < NN; ++j) {
      int n = agn[j], nb = n / QK_K;
      uint8_t *vx = (uint8_t *)malloc((size_t)nb * WB), *vy = (uint8_t *)malloc((size_t)nb * YB);
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

  const int timing_n = 32 * QK_K, iters = 2000, reps = 200;
  int nb = timing_n / QK_K;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * WB), *vy = (uint8_t *)malloc((size_t)nb * YB);
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
