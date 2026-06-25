// q1_0 VLEN256 m1 board harness (k1, SpacemiT X60, VLEN256).
//   PRIMARY GATE: OUR m1 emit vs an INDEPENDENT SCALAR oracle (kern_scalar) — catches
//     a shared bit->lane / 32-lane-reduce misconception that ggml's own vector ref
//     would also have (and thus falsely agree on).
//   SECONDARY: OUR m1 emit vs ggml's shipped _vl256 (kern_ggml_vl256) — same shape;
//     this is the parity/micro baseline ggml dispatches on a VLEN256 board.
//   MICRO: best-of-reps min, ours vs ggml _vl256. taskset-pinned externally.
// Identical random super-block bytes fed to all three. q8 quant fill [-127,127] (real
// q8_0 domain; -128 never occurs) so the i8-domain negate matches byte-exact.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <riscv_vector.h>

#define QK1_0 128
#define WB 18
#define YB 34
#define YPER 4

#define OURS tcrv_emitc_ggml_vec_dot_q1_0_q8_0_kernel_ggml_vec_dot_q1_0_q8_0
extern "C" void kern_scalar(size_t, float *, const uint8_t *, const uint8_t *);
extern "C" void kern_ggml_vl256(size_t, float *, const uint8_t *, const uint8_t *);
extern "C" void OURS(size_t, float *, const uint8_t *, const uint8_t *);

static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t rfp16() {
  uint16_t s = (xr() & 1) << 15, e = (uint16_t)(xr() % 31), m = (uint16_t)(xr() & 0x3FF);
  return s | (e << 10) | m;
}
// q8 quant bytes constrained to [-127,127]; but ALSO force-hit the ±127 boundary lanes.
static void fill(uint8_t *vx, uint8_t *vy, int nb) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * WB;
    for (int i = 0; i < WB; ++i) xb[i] = (uint8_t)(xr() & 0xFF); // random sign bits
    uint16_t d = rfp16(); memcpy(xb + 0, &d, 2);
    for (int k = 0; k < YPER; ++k) {
      uint8_t *yb = vy + ((size_t)ib * YPER + k) * YB;
      d = rfp16(); memcpy(yb + 0, &d, 2);
      for (int j = 0; j < 32; ++j) {
        int q = (int)(xr() % 255) - 127;                // [-127,127]
        yb[2 + j] = (uint8_t)(int8_t)q;
      }
      // force ±127 boundary lanes into every sub-block (lanes 0 and 31)
      yb[2 + 0]  = (uint8_t)(int8_t)( 127);
      yb[2 + 31] = (uint8_t)(int8_t)(-127);
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
  double max_rel_oracle = 0, max_rel_ggml = 0;
  int any_nonzero = 0, any_nonfinite = 0;
  for (int seed = 1; seed <= 8; ++seed) {
    rng = 0x2468ace0u ^ (seed * 0x9e3779b9u);
    for (int j = 0; j < NN; ++j) {
      int n = agn[j], nb = n / QK1_0;
      uint8_t *vx = (uint8_t *)malloc((size_t)nb * WB);
      uint8_t *vy = (uint8_t *)malloc((size_t)nb * YPER * YB);
      fill(vx, vy, nb);
      float rsc = 0, rg = 0, ro = 0;
      kern_scalar(n, &rsc, vx, vy);
      kern_ggml_vl256(n, &rg, vx, vy);
      OURS(n, &ro, vx, vy);
      if (!isfinite(rsc) || !isfinite(rg) || !isfinite(ro)) any_nonfinite = 1;
      if (rsc != 0.0f) any_nonzero = 1;
      double dsc = fabs((double)rsc); if (dsc < 1e-6) dsc = 1e-6;
      double rel_o = fabs((double)ro - (double)rsc) / dsc;  // OURS vs scalar oracle
      double dg = fabs((double)rg); if (dg < 1e-6) dg = 1e-6;
      double rel_g = fabs((double)ro - (double)rg) / dg;    // OURS vs ggml _vl256
      if (rel_o > max_rel_oracle) max_rel_oracle = rel_o;
      if (rel_g > max_rel_ggml) max_rel_ggml = rel_g;
      free(vx); free(vy);
    }
  }
  printf("AGREEMENT(ours vs scalar oracle) max_rel_norm=%.3e nonzero=%d nonfinite=%d\n",
         max_rel_oracle, any_nonzero, any_nonfinite);
  printf("AGREEMENT(ours vs ggml _vl256)   max_rel_norm=%.3e\n", max_rel_ggml);

  const int timing_n = 64 * QK1_0, iters = 4000, reps = 200;
  int nb = timing_n / QK1_0;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * WB);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * YPER * YB);
  rng = 0x13572468u; fill(vx, vy, nb);
  for (int i = 0; i < iters; i++) { float s; kern_ggml_vl256(timing_n, &s, vx, vy); OURS(timing_n, &s, vx, vy); }
  double ggml_best = 1e18, ours_best = 1e18;
  for (int r = 0; r < reps; r++) {
    double pg = tb(kern_ggml_vl256, timing_n, iters, vx, vy); if (pg < ggml_best) ggml_best = pg;
    double po = tb(OURS, timing_n, iters, vx, vy); if (po < ours_best) ours_best = po;
  }
  printf("RESULT ours %.1f\n", ours_best);
  printf("RESULT ggml(real,vl256) %.1f\n", ggml_best);
  printf("RATIO ggml/ours %.3f\n", ggml_best / ours_best);
  free(vx); free(vy);
  return 0;
}
