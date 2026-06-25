// q8_0 VLEN-flip BOARD SEAL harness.
// Gates the SHIPPING gearbox-emitted q8_0 block-dot kernel (m1-elided @ VLEN256 /
// m2-elided @ VLEN128) byte-exact vs ggml's REAL hand-written RVV q8_0 kernel
// (a SEPARATE translation unit, kern_ggml, contracted exactly as it ships under
// -ffp-contract=fast), over MANY n x MANY trials. A drift even ~1 ULP = bug ->
// the elided whole-block cover (vsetvl_e8m1(32) on real silicon) is wrong. Then
// times best-of-N (taskset-pinned). The emitted fn has the 8-arg GEMM ABI
// (n,s,bs,vx,bx,vy,by,nrc); only n/s/vx/vy are read, the rest are dummies.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

// Our shipping kernel (8-arg ABI). Name set per-variant at compile time.
extern "C" void OURS(size_t, float*, size_t, const uint8_t*, size_t,
                     const uint8_t*, size_t, int32_t);

// ggml's REAL q8_0 x q8_0 kernel, verbatim recipe (vl=32, i8m2->i16m4->vwredsum).
extern "C" void kern_ggml(size_t, float*, const uint8_t*, const uint8_t*);

static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t rfp16() {
  uint16_t s = (xr() & 1) << 15, e = (uint16_t)(xr() % 31), m = (uint16_t)(xr() & 0x3FF);
  return s | (e << 10) | m;
}
// q8_0 block: 2-byte fp16 d + 32 int8 quants, stride 34 for both x and y.
static void fill(uint8_t* vx, uint8_t* vy, int nb) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t* xb = vx + (size_t)ib * 34, *yb = vy + (size_t)ib * 34;
    uint16_t d;
    d = rfp16(); memcpy(xb, &d, 2);
    d = rfp16(); memcpy(yb, &d, 2);
    for (int i = 0; i < 32; ++i) xb[2 + i] = (uint8_t)(xr() & 0xFF);
    for (int i = 0; i < 32; ++i) yb[2 + i] = (uint8_t)(xr() & 0xFF);
  }
}
static void ours_wrap(size_t n, float* s, const uint8_t* vx, const uint8_t* vy) {
  OURS(n, s, 0, vx, 0, vy, 0, 1);
}
typedef void (*kfn)(size_t, float*, const uint8_t*, const uint8_t*);
static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double tb(kfn fn, size_t n, int it, const uint8_t* vx, const uint8_t* vy) {
  double t0 = now(); float s; for (int i = 0; i < it; i++) fn(n, &s, vx, vy); return (now() - t0) / it;
}
static int beq(float a, float b) { uint32_t x, y; memcpy(&x, &a, 4); memcpy(&y, &b, 4); return x == y; }

int main() {
  // GATE: byte-exact vs ggml over many n (incl. ODD nb to exercise the robust
  // tail AND even nb for the elided main loop) x 200 trials at -ffp-contract=fast.
  int gate_ns[] = {32, 64, 96, 128, 160, 256, 288, 512, 1024, 2048, 4096, 8192};
  const int NN = sizeof(gate_ns) / sizeof(gate_ns[0]);
  int eligible = 1, checks = 0, mism = 0;
  float first_ours = 0, first_ref = 0; int captured = 0;
  for (int t = 0; t < 200 && eligible; ++t) {
    for (int j = 0; j < NN; ++j) {
      int n = gate_ns[j], nb = n / 32;
      uint8_t* vx = (uint8_t*)malloc((size_t)nb * 34), *vy = (uint8_t*)malloc((size_t)nb * 34);
      fill(vx, vy, nb);
      float ref = 0; kern_ggml(n, &ref, vx, vy);
      float g = 0; ours_wrap(n, &g, vx, vy);
      checks++;
      if (!beq(g, ref)) {
        if (!captured) { first_ours = g; first_ref = ref; captured = 1; }
        mism++; eligible = 0;
      }
      free(vx); free(vy);
    }
  }
  printf("GATE ours-vs-ggml-real: checks=%d mismatches=%d -> %s\n",
         checks, mism, eligible ? "BYTE-EXACT" : "DRIFT");
  if (!eligible) {
    printf("FIRST-MISMATCH ours=%.9g (0x%08x) ref=%.9g (0x%08x)\n",
           first_ours, *(uint32_t*)&first_ours, first_ref, *(uint32_t*)&first_ref);
    printf("VERDICT FAIL\n");
    return 1;
  }

  // RANK: time ours vs ggml-real best-of-N min at n=4096.
  int timing_n = 4096, iters = 2000, reps = 200, nb = timing_n / 32;
  uint8_t* vx = (uint8_t*)malloc((size_t)nb * 34), *vy = (uint8_t*)malloc((size_t)nb * 34);
  fill(vx, vy, nb);
  double ours_best = 1e18, ggml_best = 1e18;
  for (int i = 0; i < iters; i++) { float s; ours_wrap(timing_n, &s, vx, vy); kern_ggml(timing_n, &s, vx, vy); }
  for (int r = 0; r < reps; r++) {
    double po = tb(ours_wrap, timing_n, iters, vx, vy); if (po < ours_best) ours_best = po;
    double pg = tb(kern_ggml, timing_n, iters, vx, vy); if (pg < ggml_best) ggml_best = pg;
  }
  printf("RESULT ours %.1f ns\n", ours_best);
  printf("RESULT ggml(real) %.1f ns\n", ggml_best);
  printf("VERDICT PASS\n");
  free(vx); free(vy);
  return 0;
}
