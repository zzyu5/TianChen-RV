// MICRO: ggml_vec_scale_f32 — OUR emit (compiler) vs ggml's REAL RVV path.
// Correctness gate (byte-exact vs ggml RVV) BEFORE perf; then min-of-reps timing.
// Both arms read the SAME freshly re-initialized buffer per rep (in-place op).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <riscv_vector.h>

extern "C" void
tcrv_emitc_ggml_vec_scale_f32_kernel_ggml_vec_scale_f32(size_t n, float *y, float v);

// ggml's REAL ggml_vec_scale_f32 RVV path (vec.h:733-739), verbatim.
static void ggml_rvv(int n, float *y, float v) {
  for (int i = 0, avl; i < n; i += avl) {
    avl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
    vfloat32m8_t ny = __riscv_vfmul_vf_f32m8(ay, v, avl);
    __riscv_vse32_v_f32m8(&y[i], ny, avl);
  }
}

static double now_ns() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main() {
  const int n = 8192;            // row-ish size, hot in L1
  const int REPS = 20000, INNER = 1;
  float *base = (float *)malloc(n * 4);
  float *yo = (float *)malloc(n * 4);
  float *yr = (float *)malloc(n * 4);
  for (int i = 0; i < n; i++) base[i] = ((i * 2654435761u) % 20000) / 100.0f - 100.0f;
  const float v = 3.14159f;

  // ---- correctness gate ----
  memcpy(yo, base, n * 4); memcpy(yr, base, n * 4);
  tcrv_emitc_ggml_vec_scale_f32_kernel_ggml_vec_scale_f32((size_t)n, yo, v);
  ggml_rvv(n, yr, v);
  int ok = memcmp(yo, yr, n * 4) == 0;
  printf("scale GATE byte-exact vs ggml RVV: %s\n", ok ? "PASS" : "FAIL");
  if (!ok) { printf("RESULT: CORRECTNESS FAIL — no perf\n"); return 1; }

  // ---- timing: min-of-reps, re-init per rep, sink output ----
  volatile float sink = 0;
  double best_o = 1e18, best_r = 1e18;
  for (int r = 0; r < REPS; r++) {
    memcpy(yo, base, n * 4);
    double t0 = now_ns();
    for (int k = 0; k < INNER; k++)
      tcrv_emitc_ggml_vec_scale_f32_kernel_ggml_vec_scale_f32((size_t)n, yo, v);
    double t1 = now_ns();
    sink += yo[r % n];
    if (t1 - t0 < best_o) best_o = t1 - t0;
  }
  for (int r = 0; r < REPS; r++) {
    memcpy(yr, base, n * 4);
    double t0 = now_ns();
    for (int k = 0; k < INNER; k++) ggml_rvv(n, yr, v);
    double t1 = now_ns();
    sink += yr[r % n];
    if (t1 - t0 < best_r) best_r = t1 - t0;
  }
  printf("scale OURS=%.1f ns  GGML=%.1f ns  ratio(ggml/ours)=%.3fx  [sink=%.1f]\n",
         best_o, best_r, best_r / best_o, (double)sink);
  return 0;
}
