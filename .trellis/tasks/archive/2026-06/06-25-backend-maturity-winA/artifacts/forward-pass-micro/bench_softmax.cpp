// MICRO: ggml_vec_soft_max_f32 — OUR emit vs ggml's REAL RVV path (vec.cpp:584-592).
// Returns f64 sum. Both at m2, shared exp polynomial. Gate byte-exact (y[] + sum);
// min-of-reps. y is OUT-only (x in), so no per-rep re-init needed for inputs.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <riscv_vector.h>

extern "C" double
tcrv_emitc_ggml_vec_soft_max_f32_kernel_ggml_vec_soft_max_f32(size_t n, float *y,
                                                              const float *x, float max);

// shared ggml_v_expf_m2 verbatim.
static inline vfloat32m2_t ggml_v_expf_m2_ref(vfloat32m2_t x, int vl) {
  const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
  const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
  const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
  const vfloat32m2_t b = __riscv_vfnmsac_vf_f32m2(
      __riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
  const vuint32m2_t e =
      __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
  const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(
      __riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
  const vbool16_t c =
      __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
  const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
  const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
      __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
      __riscv_vfmacc_vv_f32m2(
          __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl),
                                  0x1.555e66p-3f, b, vl),
          __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl),
                                  0x1.0e4020p-7f, b, vl),
          u, vl),
      u, vl);
  if (!__riscv_vcpop_m_b16(c, vl))
    return __riscv_vfmacc_vv_f32m2(k, j, k, vl);
  const vbool16_t dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
  const vuint32m2_t d =
      __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
  const vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(
      __riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
  const vfloat32m2_t s2 =
      __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
  const vfloat32m2_t r1 = __riscv_vmerge_vvm_f32m2(
      __riscv_vfmacc_vv_f32m2(k, k, j, vl),
      __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
  return __riscv_vmerge_vvm_f32m2(
      r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
      __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}
// ggml's REAL soft_max RVV (vec.cpp:584-592): val=exp(x-max), store y, widening f64 reduce.
static double ggml_ref(int n, float *y, const float *x, float max) {
  vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0.0, 1);
  for (int i = 0, vl; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m2(n - i);
    vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
    vfloat32m2_t val = ggml_v_expf_m2_ref(__riscv_vfsub_vf_f32m2(vx, max, vl), vl);
    __riscv_vse32_v_f32m2(&y[i], val, vl);
    vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, vl);
  }
  return (double)__riscv_vfmv_f_s_f64m1_f64(vsum);
}

static double now_ns() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e9+ts.tv_nsec; }

int main() {
  const int n = 8192;
  const int REPS = 20000;
  float *x = (float *)malloc(n*4), *yo = (float *)malloc(n*4), *yr = (float *)malloc(n*4);
  float max = -1e30f;
  for (int i = 0; i < n; i++) { x[i] = ((i*2654435761u)%20000)/400.0f - 25.0f; if (x[i] > max) max = x[i]; }

  double so = tcrv_emitc_ggml_vec_soft_max_f32_kernel_ggml_vec_soft_max_f32((size_t)n, yo, x, max);
  double sr = ggml_ref(n, yr, x, max);
  int ok_y = memcmp(yo, yr, n*4) == 0;
  int ok_s = memcmp(&so, &sr, sizeof(double)) == 0;
  printf("softmax GATE y byte-exact: %s ; sum byte-exact: %s (ours=%.17g ggml=%.17g)\n",
         ok_y?"PASS":"FAIL", ok_s?"PASS":"FAIL", so, sr);
  if (!ok_y || !ok_s) { printf("RESULT: CORRECTNESS FAIL\n"); return 1; }

  volatile double sink = 0; double bo = 1e18, br = 1e18;
  for (int r = 0; r < REPS; r++) {
    double t0 = now_ns();
    sink += tcrv_emitc_ggml_vec_soft_max_f32_kernel_ggml_vec_soft_max_f32((size_t)n, yo, x, max);
    double t1 = now_ns(); if (t1-t0 < bo) bo = t1-t0;
  }
  for (int r = 0; r < REPS; r++) {
    double t0 = now_ns(); sink += ggml_ref(n, yr, x, max);
    double t1 = now_ns(); if (t1-t0 < br) br = t1-t0;
  }
  printf("softmax OURS=%.1f ns  GGML=%.1f ns  ratio(ggml/ours)=%.3fx  [sink=%.1f]\n",
         bo, br, br/bo, (double)sink);
  return 0;
}
