// Coverage harness: OUR emitted nvfp4_q8_0 block-dot vs ggml's shipped riscv kernel.
// nvfp4 = FP4 e2m1 codebook gather on a QK=64 SUPER-block with a per-16-elem UE4M3
// sub-block scale. block_nvfp4=36B {d[4]@0 UE4M3, qs[32]@4}; activation = TWO
// block_q8_0(34B) per superblock. ours.cpp gathers each 16-elem sub-block via two
// 8-lane vrgather_vv_i8m1 over the 16-entry register codebook (the FP4-codebook
// mechanism, m1, but a NARROWER 8-lane strip than iq4_nl/mxfp4's 16-lane halves).
// BASELINE NOTE: ggml ships NO RVV nvfp4 kernel on riscv -> the ggml side here is the
// GENERIC SCALAR oracle. Correctness rel-norm is fully valid; the timing ratio is
// vector-vs-scalar (NOT the vector-vs-vector gather-shape contest), so generalization
// of the iq4_nl 1.32x is N/A for nvfp4 (no ggml vector kernel to beat).
// UE4M3 scale bytes constrained to [0x38,0x47] (exp 7..8, nonzero, != 0x7F) so scales
// are O(1) and finite -- honest fill caveat.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <riscv_vector.h>

#define QK 64
#define WB 36
#define YB 34
#define OURS tcrv_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_ggml_vec_dot_nvfp4_q8_0

extern "C" void kern_ggml(size_t, float *, const uint8_t *, const uint8_t *);
extern "C" void OURS(size_t, float *, const uint8_t *, const uint8_t *);

static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t rfp16() {
  uint16_t s = (xr() & 1) << 15, e = (uint16_t)(xr() % 31), m = (uint16_t)(xr() & 0x3FF);
  return s | (e << 10) | m;
}
// nb = number of QK=64 superblocks. weight = nb*36B. activation = (2*nb) q8_0 blocks.
// boundary_scales (agreement run): MIX IN the genuinely-new UE4M3 decode branches the
// test IR flagged: the e==0/e==0x7F specials (->0.0f) and the exp==0 denormal branch
// (d in 0x01..0x07). We avoid LARGE exp (overflow->inf) but DO cover specials+denormal.
// When false (timing): keep d in [0x38,0x47] (exp 7..8) so scales are O(1).
static void fill(uint8_t *vx, uint8_t *vy, int nb, int boundary_scales) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * WB;
    for (int i = 0; i < WB; ++i) xb[i] = (uint8_t)(xr() & 0xFF);
    for (int s = 0; s < 4; ++s) {
      if (boundary_scales) {
        uint32_t r = xr() % 6;
        if (r == 0) xb[s] = 0x00;                       // special -> 0.0f
        else if (r == 1) xb[s] = 0x7F;                  // special -> 0.0f
        else if (r == 2) xb[s] = (uint8_t)(0x01 + (xr() % 7)); // exp==0 denormal branch
        else xb[s] = (uint8_t)(0x28 + (xr() % 40));     // exp 5..9 -> finite O(1)..O(few)
      } else {
        xb[s] = (uint8_t)(0x38 + (xr() % 16));          // timing: exp 7..8, O(1)
      }
    }
  }
  int yblocks = 2 * nb;
  for (int yb = 0; yb < yblocks; ++yb) {
    uint8_t *yp = vy + (size_t)yb * YB;
    for (int i = 0; i < YB; ++i) yp[i] = (uint8_t)(xr() & 0xFF);
    uint16_t d = rfp16(); memcpy(yp + 0, &d, 2);
  }
}
typedef void (*kfn)(size_t, float *, const uint8_t *, const uint8_t *);
static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double tb(kfn fn, size_t n, int it, const uint8_t *vx, const uint8_t *vy) {
  double t0 = now(); float s; for (int i = 0; i < it; i++) fn(n, &s, vx, vy); return (now() - t0) / it;
}
int main() {
  int agn[] = { 1*QK, 2*QK, 4*QK, 8*QK, 16*QK, 32*QK };  // all multiples of 64
  const int NN = sizeof(agn)/sizeof(agn[0]);
  double max_rel = 0; int any_nonzero = 0, any_nonfinite = 0;
  for (int seed = 1; seed <= 8; ++seed) {
    rng = 0x2468ace0u ^ (seed * 0x9e3779b9u);
    for (int j = 0; j < NN; ++j) {
      int n = agn[j], nb = n / QK;
      uint8_t *vx = (uint8_t *)malloc((size_t)nb * WB);
      uint8_t *vy = (uint8_t *)malloc((size_t)(2 * nb) * YB);
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

  const int timing_n = 32 * QK, iters = 4000, reps = 200;
  int nb = timing_n / QK;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * WB);
  uint8_t *vy = (uint8_t *)malloc((size_t)(2 * nb) * YB);
  rng = 0x13572468u; fill(vx, vy, nb, /*boundary_scales=*/0);
  for (int i = 0; i < iters; i++) { float s; kern_ggml(timing_n, &s, vx, vy); OURS(timing_n, &s, vx, vy); }
  double ggml_best = 1e18, ours_best = 1e18;
  for (int r = 0; r < reps; r++) {
    double pg = tb(kern_ggml, timing_n, iters, vx, vy); if (pg < ggml_best) ggml_best = pg;
    double po = tb(OURS, timing_n, iters, vx, vy); if (po < ours_best) ours_best = po;
  }
  printf("RESULT ours %.1f\n", ours_best);
  printf("RESULT ggml(generic-scalar,no-rvv-kernel) %.1f\n", ggml_best);
  printf("RATIO ggml/ours %.3f (vector-vs-SCALAR; not a gather-shape contest)\n", ggml_best / ours_best);
  free(vx); free(vy);
  return 0;
}
