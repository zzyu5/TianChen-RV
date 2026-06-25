// MICRO: ggml_rope_norm_f32 — OUR emit vs ggml's EXACT NORMAL rope (ops.cpp NORMAL
// path). ggml has NO vectorized RVV rope: the angle cache is SCALAR libm cosf/sinf,
// the rotation is a scalar two-rounding mul;mul;sub/add loop. BOTH our emit and the
// ggml ref link the SAME libm and use -ffp-contract=off, so this is a same-algo,
// scalar-transcendental-dominated comparison → expect TIE. Gate byte-exact; min-reps.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

extern "C" void
tcrv_emitc_ggml_rope_norm_f32_kernel_ggml_rope_norm_f32(size_t n_dims, const float *x,
                                                        float *y, float theta_base,
                                                        float theta_scale);

static void ggml_rope_cache_init_ref(float theta_base, float theta_scale,
                                     int64_t ne0, float *cache) {
  float theta = theta_base;
  for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
    cache[i0 + 0] = cosf(theta);
    cache[i0 + 1] = sinf(theta);
    theta *= theta_scale;
  }
}
static void ggml_rope_norm_ref(int64_t n_dims, const float *x, float *y,
                               float theta_base, float theta_scale, float *cache) {
  ggml_rope_cache_init_ref(theta_base, theta_scale, n_dims, cache);
  for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
    const float c = cache[i0 + 0], s = cache[i0 + 1];
    const float x0 = x[i0 + 0], x1 = x[i0 + 1];
    y[i0 + 0] = x0 * c - x1 * s;
    y[i0 + 1] = x0 * s + x1 * c;
  }
}

static double now_ns() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e9+ts.tv_nsec; }

int main() {
  const int64_t HEAD_DIM = 128;
  const float FREQ_BASE = 10000.0f;
  const float theta_scale = powf(FREQ_BASE, -2.0f / (float)HEAD_DIM);
  const float theta_base = 2047.0f;   // a mid position (range-reduction exercised)
  const int REPS = 60000;
  float *x = (float *)malloc(HEAD_DIM*4), *yo = (float *)malloc(HEAD_DIM*4),
        *yr = (float *)malloc(HEAD_DIM*4), *cache = (float *)malloc(HEAD_DIM*4);
  for (int i = 0; i < HEAD_DIM; i++) x[i] = ((i*2654435761u)%20000)/1000.0f - 10.0f;

  tcrv_emitc_ggml_rope_norm_f32_kernel_ggml_rope_norm_f32((size_t)HEAD_DIM, x, yo, theta_base, theta_scale);
  ggml_rope_norm_ref(HEAD_DIM, x, yr, theta_base, theta_scale, cache);
  int ok = memcmp(yo, yr, HEAD_DIM*4) == 0;
  printf("rope GATE byte-exact vs ggml NORMAL rope (same libm): %s\n", ok ? "PASS" : "FAIL");
  if (!ok) { printf("RESULT: CORRECTNESS FAIL\n"); return 1; }

  // Rotate over many positions per timed unit so the row (128 elems) isn't trivially
  // sub-microsecond; both arms do the SAME 256 positions.
  const int NPOS = 256;
  volatile float sink = 0; double bo = 1e18, br = 1e18;
  for (int r = 0; r < REPS; r++) {
    double t0 = now_ns();
    for (int p = 0; p < NPOS; p++)
      tcrv_emitc_ggml_rope_norm_f32_kernel_ggml_rope_norm_f32((size_t)HEAD_DIM, x, yo, theta_base + p, theta_scale);
    double t1 = now_ns(); sink += yo[r%HEAD_DIM]; if (t1-t0 < bo) bo = t1-t0;
  }
  for (int r = 0; r < REPS; r++) {
    double t0 = now_ns();
    for (int p = 0; p < NPOS; p++)
      ggml_rope_norm_ref(HEAD_DIM, x, yr, theta_base + p, theta_scale, cache);
    double t1 = now_ns(); sink += yr[r%HEAD_DIM]; if (t1-t0 < br) br = t1-t0;
  }
  printf("rope OURS=%.1f ns  GGML=%.1f ns  ratio(ggml/ours)=%.3fx  (%d pos/unit) [sink=%.1f]\n",
         bo, br, br/bo, NPOS, (double)sink);
  return 0;
}
