// Win-B·micro harness: OUR emitted q5_K block-dot vs ggml's shipped RVV path.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <riscv_vector.h>

extern "C" void kern_ggml(size_t, float *, const uint8_t *, const uint8_t *);
extern "C" void tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_ggml_vec_dot_q5_K_q8_K(size_t, float *, const uint8_t *, const uint8_t *);

#define QK_K 256
#define WB 176   /* block_q5_K bytes */
#define YB 292   /* block_q8_K bytes */

static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t rfp16() { uint16_t s=(xr()&1)<<15,e=(uint16_t)(xr()%31),m=(uint16_t)(xr()&0x3FF); return s|(e<<10)|m; }
// Weight (q5_K, 176B): dm.d@0(fp16) dm.dmin@2(fp16) scales[12]@4 qh[32]@16 qs[128]@48.
// Activation (q8_K, 292B): d@0(f32) qs[256]@4 bsums[16]@260(int16, consistent).
static void fill(uint8_t *vx, uint8_t *vy, int nb) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * WB, *yb = vy + (size_t)ib * YB;
    uint16_t d;
    d = rfp16(); memcpy(xb + 0, &d, 2);
    d = rfp16(); memcpy(xb + 2, &d, 2);
    for (int i = 0; i < 12;  ++i) xb[4 + i]  = (uint8_t)(xr() & 0xFF); /* scales */
    for (int i = 0; i < 32;  ++i) xb[16 + i] = (uint8_t)(xr() & 0xFF); /* qh */
    for (int i = 0; i < 128; ++i) xb[48 + i] = (uint8_t)(xr() & 0xFF); /* qs */
    float yd = 0.01f + (float)(xr() % 1000) / 1000.0f;
    memcpy(yb + 0, &yd, 4);
    int8_t q8[256];
    for (int i = 0; i < 256; ++i) { q8[i] = (int8_t)((xr() & 0xFF) - 128) / 4; yb[4 + i] = (uint8_t)q8[i]; }
    for (int g = 0; g < 16; ++g) { int16_t bs=0; for (int k=0;k<16;++k) bs+=q8[g*16+k]; memcpy(yb+260+g*2,&bs,2); }
  }
}
typedef void (*kfn)(size_t, float *, const uint8_t *, const uint8_t *);
static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e9 + t.tv_nsec; }
static double tb(kfn fn, size_t n, int it, const uint8_t *vx, const uint8_t *vy) {
  double t0=now(); float s; for (int i=0;i<it;i++) fn(n,&s,vx,vy); return (now()-t0)/it;
}
int main() {
  int agn[] = { QK_K, 2*QK_K, 4*QK_K, 8*QK_K, 16*QK_K };
  const int NN = sizeof(agn)/sizeof(agn[0]);
  double max_rel=0; int any_nonzero=0, any_nonfinite=0;
  for (int seed=1; seed<=8; ++seed) {
    rng = 0x2468ace0u ^ (seed*0x9e3779b9u);
    for (int j=0;j<NN;++j) {
      int n=agn[j], nb=n/QK_K;
      uint8_t *vx=(uint8_t*)malloc((size_t)nb*WB), *vy=(uint8_t*)malloc((size_t)nb*YB);
      fill(vx,vy,nb);
      float rg=0,ro=0;
      kern_ggml(n,&rg,vx,vy);
      tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_ggml_vec_dot_q5_K_q8_K(n,&ro,vx,vy);
      if (!isfinite(rg)||!isfinite(ro)) any_nonfinite=1;
      if (rg!=0.0f) any_nonzero=1;
      double denom=fabs((double)rg); if (denom<1e-6) denom=1e-6;
      double rel=fabs((double)ro-(double)rg)/denom; if (rel>max_rel) max_rel=rel;
      free(vx); free(vy);
    }
  }
  printf("AGREEMENT max_rel_norm=%.3e nonzero=%d nonfinite=%d\n", max_rel, any_nonzero, any_nonfinite);

  const int timing_n=8192, iters=2000, reps=200;
  int nb=timing_n/QK_K;
  uint8_t *vx=(uint8_t*)malloc((size_t)nb*WB), *vy=(uint8_t*)malloc((size_t)nb*YB);
  rng=0x13572468u; fill(vx,vy,nb);
  for (int i=0;i<iters;i++){ float s; kern_ggml(timing_n,&s,vx,vy); tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_ggml_vec_dot_q5_K_q8_K(timing_n,&s,vx,vy); }
  double ggml_best=1e18, ours_best=1e18;
  for (int r=0;r<reps;r++){
    double pg=tb(kern_ggml,timing_n,iters,vx,vy); if(pg<ggml_best) ggml_best=pg;
    double po=tb(tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_ggml_vec_dot_q5_K_q8_K,timing_n,iters,vx,vy); if(po<ours_best) ours_best=po;
  }
  printf("RESULT ours %.1f\n", ours_best);
  printf("RESULT ggml(real,rvv) %.1f\n", ggml_best);
  printf("RATIO ggml/ours %.3f\n", ggml_best/ours_best);
  free(vx); free(vy);
  return 0;
}
