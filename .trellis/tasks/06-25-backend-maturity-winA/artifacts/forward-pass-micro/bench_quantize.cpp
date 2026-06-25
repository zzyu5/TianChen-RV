// MICRO: quantize_row_q8_0 — OUR emit vs ggml's REAL RVV quantizer
// (riscv/quants.c:32-71, __riscv_v branch). Both at m8, same vfncvt (rne) rounding.
// Gate byte-exact (the whole block_q8_0 byte stream); min-of-reps.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <riscv_vector.h>

#define QK8_0 32
typedef uint16_t ggml_half;
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;  // 34 bytes

static inline ggml_half fp32_to_fp16(float f) {
  _Float16 h = (_Float16)f; ggml_half r; memcpy(&r, &h, 2); return r;
}

extern "C" void
tcrv_emitc_quantize_row_q8_0_kernel_quantize_row_q8_0(size_t n, const float *x, uint8_t *y);

// ggml's REAL RVV quantize_row_q8_0 (riscv/quants.c:32-71), verbatim.
static void ggml_quantize_rvv(int k, const float *x, void *vy) {
  const int nb = k / QK8_0;
  block_q8_0 *y = (block_q8_0 *)vy;
  size_t vl = QK8_0;
  for (int i = 0; i < nb; i++) {
    vfloat32m8_t v_x = __riscv_vle32_v_f32m8(x + i * QK8_0, vl);
    vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(v_x, vl);
    vfloat32m1_t tmp = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m8_f32m1(vabs, tmp, vl);
    float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);
    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = fp32_to_fp16(d);
    vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);
    vint16m4_t vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);
    vint8m2_t vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);
    __riscv_vse8_v_i8m2(y[i].qs, vs, vl);
  }
}

static double now_ns() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e9+ts.tv_nsec; }

int main() {
  const int n = 8192;                 // 256 blocks
  const int nb = n / QK8_0;
  const int REPS = 30000;
  float *x = (float *)malloc(n*4);
  uint8_t *yo = (uint8_t *)malloc(nb*34), *yr = (uint8_t *)malloc(nb*34);
  for (int i = 0; i < n; i++) x[i] = ((i*2654435761u)%20000)/100.0f - 100.0f;

  tcrv_emitc_quantize_row_q8_0_kernel_quantize_row_q8_0((size_t)n, x, yo);
  ggml_quantize_rvv(n, x, yr);
  int ok = memcmp(yo, yr, nb*34) == 0;
  printf("quantize_q8_0 GATE byte-exact vs ggml RVV (d+qs): %s\n", ok ? "PASS" : "FAIL");
  if (!ok) { printf("RESULT: CORRECTNESS FAIL\n"); return 1; }

  volatile int sink = 0; double bo = 1e18, br = 1e18;
  for (int r = 0; r < REPS; r++) {
    double t0 = now_ns();
    tcrv_emitc_quantize_row_q8_0_kernel_quantize_row_q8_0((size_t)n, x, yo);
    double t1 = now_ns(); sink += yo[r%(nb*34)]; if (t1-t0 < bo) bo = t1-t0;
  }
  for (int r = 0; r < REPS; r++) {
    double t0 = now_ns(); ggml_quantize_rvv(n, x, yr);
    double t1 = now_ns(); sink += yr[r%(nb*34)]; if (t1-t0 < br) br = t1-t0;
  }
  printf("quantize_q8_0 OURS=%.1f ns  GGML=%.1f ns  ratio(ggml/ours)=%.3fx  [sink=%d]\n",
         bo, br, br/bo, sink);
  return 0;
}
