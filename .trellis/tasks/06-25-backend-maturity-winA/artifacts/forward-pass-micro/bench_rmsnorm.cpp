// MICRO: ggml_rms_norm_f32 — OUR emit (vectorized: scalar-double Sum x^2 fold +
// vectorized m8 normalize) vs ggml's REAL row body (ops.cpp:3791-3817). ggml has
// NO RVV path for rms_norm: the Sum is a SCALAR ggml_float(double) loop; the
// normalize is ggml_vec_scale_f32 (m8 RVV). So this is "we vectorized the parts
// ggml leaves to its scalar source + autovec" — coverage/maturity, NOT a Win-A
// knob and NOT same-algo tuning. Gate byte-exact; min-of-reps.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <riscv_vector.h>

typedef double ggml_float;

extern "C" void
tcrv_emitc_ggml_rms_norm_f32_kernel_ggml_rms_norm_f32(size_t ne00, const float *x,
                                                      float *y, float eps);

// ggml's REAL rms_norm row body (ops.cpp:3791-3817), verbatim. This is ggml's
// ACTUAL code as built (no hand-RVV exists for rms_norm). The double-sum reduction
// does NOT auto-vectorize under -O3 -march=rv64gcv (fp reassoc would change bits);
// the y[i]=x[i]*scale loop MAY auto-vectorize — both forms are "ggml as built".
static void ggml_ref(int ne00, const float *x, float *y, float eps) {
  ggml_float sum = 0.0;
  for (int i00 = 0; i00 < ne00; i00++)
    sum += (ggml_float)(x[i00] * x[i00]);
  const float mean = sum / ne00;
  const float scale = 1.0f / sqrtf(mean + eps);
  for (int i = 0; i < ne00; i++)
    y[i] = x[i] * scale;
}

static double now_ns() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e9+ts.tv_nsec; }

int main() {
  const int n = 4096;            // a real llama hidden dim
  const int REPS = 30000;
  const float eps = 1e-5f;
  float *x = (float *)malloc(n*4), *yo = (float *)malloc(n*4), *yr = (float *)malloc(n*4);
  for (int i = 0; i < n; i++) x[i] = ((i*2654435761u)%20000)/2000.0f - 5.0f;

  tcrv_emitc_ggml_rms_norm_f32_kernel_ggml_rms_norm_f32((size_t)n, x, yo, eps);
  ggml_ref(n, x, yr, eps);
  int ok = memcmp(yo, yr, n*4) == 0;
  printf("rms_norm GATE byte-exact vs ggml (scalar-source): %s\n", ok ? "PASS" : "FAIL");
  if (!ok) {
    for (int i = 0; i < 4; i++) { uint32_t a,b; memcpy(&a,&yo[i],4); memcpy(&b,&yr[i],4);
      printf("  [%d] ours=%08x ggml=%08x\n", i, a, b); }
    printf("RESULT: CORRECTNESS FAIL\n"); return 1;
  }

  volatile float sink = 0; double bo = 1e18, br = 1e18;
  for (int r = 0; r < REPS; r++) {
    double t0 = now_ns();
    tcrv_emitc_ggml_rms_norm_f32_kernel_ggml_rms_norm_f32((size_t)n, x, yo, eps);
    double t1 = now_ns(); sink += yo[r%n]; if (t1-t0 < bo) bo = t1-t0;
  }
  for (int r = 0; r < REPS; r++) {
    double t0 = now_ns(); ggml_ref(n, x, yr, eps);
    double t1 = now_ns(); sink += yr[r%n]; if (t1-t0 < br) br = t1-t0;
  }
  printf("rms_norm OURS=%.1f ns  GGML=%.1f ns  ratio(ggml/ours)=%.3fx  [sink=%.1f]\n",
         bo, br, br/bo, (double)sink);
  return 0;
}
