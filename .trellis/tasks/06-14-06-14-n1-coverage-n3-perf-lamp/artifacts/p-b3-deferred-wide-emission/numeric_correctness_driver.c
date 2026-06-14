/* P-B step 3 ssh-rvv NUMERIC correctness driver for the COMPILER-EMITTED
   deferred-wide low-precision contraction body.

   The function `tcrv_emitc_wide_dequantize_kernel_wide_dequantize` below is the
   VERBATIM output of the TianChen-RV compiler:
     tcrv-opt <wide-lmul test>.mlir --tcrv-rvv-lower-to-emitc
       | mlir-translate --mlir-to-cpp
   (saved at compiler_emitted_deferred_wide.cpp.txt). It is NOT hand-written.

   This driver fills pseudo-random int8 lhs/rhs + an i32 acc seed + an f32 scale,
   runs the compiler-emitted body, and compares its f32 output against a GENUINE
   scalar reference computing exactly acc[0] + sum(lhs[i]*rhs[i]) then *scale.
   PASS iff |emitted - reference| <= 1e-5 * (1 + |reference|) for every n,
   including the prime-tail n=257 (partial last strip into the wide accumulator
   + single trailing vredsum). A fast-wrong body is disqualified (I8). */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <riscv_vector.h>

/* ===== BEGIN compiler-emitted body (verbatim mlir-translate output) ===== */
extern "C" void tcrv_emitc_wide_dequantize_kernel_wide_dequantize(const int8_t* v1, const int8_t* v2, const int32_t* v3, float v4, float* v5, size_t v6) {
  size_t v7 = __riscv_vsetvl_e8m2(v6);
  vint32m8_t v8;
  size_t v9 = __riscv_vsetvlmax_e32m8();
  vint32m8_t v10 = __riscv_vmv_v_x_i32m8(0, v9);
  v8 = v10;
  for (size_t v11 = 0; v11 < v6; v11 += v7) {
    size_t v12 = v6 - v11;
    size_t v13 = __riscv_vsetvl_e8m2(v12);
    const int8_t* v14 = v1 + v11;
    vint8m2_t v15 = __riscv_vle8_v_i8m2(v14, v13);
    const int8_t* v16 = v2 + v11;
    vint8m2_t v17 = __riscv_vle8_v_i8m2(v16, v13);
    vint16m4_t v18 = __riscv_vwmul_vv_i16m4(v15, v17, v13);
    vint32m8_t v19 = v8;
    vint32m8_t v20 = __riscv_vwadd_wv_i32m8(v19, v18, v13);
    v8 = v20;
  }
  size_t v21 = __riscv_vsetvlmax_e32m1();
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(0, v21);
  vint32m8_t v23 = v8;
  vint32m1_t v24 = __riscv_vredsum_vs_i32m8_i32m1(v23, v22, v9);
  int32_t v25 = __riscv_vmv_x_s_i32m1_i32(v24);
  const int32_t v26 = v3[0];
  int32_t v27 = v26 + v25;
  float v28 = (float) v27;
  float v29 = v28 * v4;
  vfloat32m1_t v30 = __riscv_vfmv_v_f_f32m1(v29, 1);
  __riscv_vse32_v_f32m1(v5, v30, 1);
  return;
}
/* ===== END compiler-emitted body ===== */

/* Genuine scalar reference (no vector intrinsics): acc[0] + sum(lhs*rhs), *scale. */
static float scalar_reference(const int8_t *lhs, const int8_t *rhs,
                              const int32_t *acc, float scale, size_t n) {
  int32_t sum = acc[0];
  for (size_t i = 0; i < n; ++i)
    sum += (int32_t)lhs[i] * (int32_t)rhs[i];
  return ((float)sum) * scale;
}

static uint32_t rng_state = 0x12345678u;
static int8_t rand_i8(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return (int8_t)((rng_state >> 17) & 0xFF);
}

int main(void) {
  const size_t sizes[] = {257, 256, 1024, 4096, 16384, 65536};
  const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int all_pass = 1;
  for (size_t s = 0; s < num_sizes; ++s) {
    size_t n = sizes[s];
    int8_t *lhs = (int8_t *)malloc(n);
    int8_t *rhs = (int8_t *)malloc(n);
    int32_t acc[1];
    float scale = 0.013725f;
    float out = 0.0f;
    for (size_t i = 0; i < n; ++i) { lhs[i] = rand_i8(); rhs[i] = rand_i8(); }
    acc[0] = -4096;

    tcrv_emitc_wide_dequantize_kernel_wide_dequantize(lhs, rhs, acc, scale, &out, n);
    float ref = scalar_reference(lhs, rhs, acc, scale, n);
    float tol = 1e-5f * (1.0f + fabsf(ref));
    int pass = fabsf(out - ref) <= tol;
    printf("n=%-6zu emitted=%.6f reference=%.6f abs_err=%.3e tol=%.3e %s\n",
           n, out, ref, fabsf(out - ref), tol, pass ? "PASS" : "FAIL");
    if (!pass) all_pass = 0;
    free(lhs); free(rhs);
  }
  printf("RESULT: %s\n", all_pass ? "ALL_PASS" : "FAIL");
  return all_pass ? 0 : 1;
}
