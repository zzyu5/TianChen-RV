/* P-B7 Stage 2a ssh-rvv NUMERIC correctness driver for the COMPILER-EMITTED
   deferred-wide i16 dot-reduce body (the 2nd kernel family).

   The function tcrv_emitc_wide_dot_reduce_kernel_wide_dot_reduce below is the
   VERBATIM output of the TianChen-RV compiler:
     tcrv-opt test/Conversion/RVV/rvv-to-emitc-widening-dot-reduce-wide-lmul.mlir
       --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
   (saved at compiler_emitted_deferred_wide_dot_reduce.cpp.txt). NOT hand-written.

   This driver fills pseudo-random i16 lhs/rhs + an i32 acc seed, runs the
   compiler-emitted body, and compares its i32 output against a GENUINE scalar
   reference computing exactly acc[0] + sum(lhs[i]*rhs[i]). EXACT integer
   equality at every n incl. the prime-tail n=257 (partial last strip into the
   wide accumulator + single trailing vredsum). A fast-wrong body is
   disqualified (I8). */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <riscv_vector.h>

/* ===== BEGIN compiler-emitted body (verbatim mlir-translate output) ===== */
extern "C" void tcrv_emitc_wide_dot_reduce_kernel_wide_dot_reduce(const int16_t* v1, const int16_t* v2, const int32_t* v3, int32_t* v4, size_t v5) {
  size_t v6 = __riscv_vsetvl_e16m4(v5);
  vint32m8_t v7;
  size_t v8 = __riscv_vsetvlmax_e32m8();
  vint32m8_t v9 = __riscv_vmv_v_x_i32m8(0, v8);
  v7 = v9;
  for (size_t v10 = 0; v10 < v5; v10 += v6) {
    size_t v11 = v5 - v10;
    size_t v12 = __riscv_vsetvl_e16m4(v11);
    const int16_t* v13 = v1 + v10;
    vint16m4_t v14 = __riscv_vle16_v_i16m4(v13, v12);
    const int16_t* v15 = v2 + v10;
    vint16m4_t v16 = __riscv_vle16_v_i16m4(v15, v12);
    vint32m8_t v17 = __riscv_vwmul_vv_i32m8(v14, v16, v12);
    vint32m8_t v18 = v7;
    vint32m8_t v19 = __riscv_vadd_vv_i32m8(v18, v17, v12);
    v7 = v19;
  }
  size_t v20 = __riscv_vsetvlmax_e32m1();
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(0, v20);
  vint32m8_t v22 = v7;
  vint32m1_t v23 = __riscv_vredsum_vs_i32m8_i32m1(v22, v21, v8);
  int32_t v24 = __riscv_vmv_x_s_i32m1_i32(v23);
  const int32_t v25 = v3[0];
  int32_t v26 = v25 + v24;
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v26, 1);
  __riscv_vse32_v_i32m1(v4, v27, 1);
  return;
}
/* ===== END compiler-emitted body ===== */

static int32_t ref_scalar(const int16_t *lhs, const int16_t *rhs,
                          const int32_t *acc, size_t n) {
  int32_t sum = acc[0];
  for (size_t i = 0; i < n; ++i)
    sum += (int32_t)lhs[i] * (int32_t)rhs[i];
  return sum;
}

static int run_case(size_t n) {
  const size_t alloc_n = n == 0 ? 1 : n;
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int16_t *rhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * 4);
  int32_t out = 0;
  if (!lhs || !rhs || !acc) { fprintf(stderr, "alloc n=%zu\n", n); return 11; }
  for (size_t i = 0; i < alloc_n; ++i) {
    lhs[i] = (int16_t)(((i % 6) < 3) ? -((int)(i % 211) + 37)
                                     : ((int)(i % 197) + 41));
    rhs[i] = (int16_t)(((i % 5) == 1 || (i % 5) == 4) ? -((int)(i % 173) + 29)
                                                      : ((int)(i % 181) + 53));
  }
  acc[0] = (int32_t)(-137 + (int32_t)n);
  acc[1] = acc[2] = acc[3] = 0;

  int32_t reference = ref_scalar(lhs, rhs, acc, n);
  tcrv_emitc_wide_dot_reduce_kernel_wide_dot_reduce(lhs, rhs, acc, &out, n);

  int ok = (out == reference);
  printf("n=%zu emitted=%d reference=%d %s\n", n, out, reference,
         ok ? "OK" : "MISMATCH");
  free(lhs); free(rhs); free(acc);
  return ok ? 0 : 12;
}

int main(void) {
  const size_t counts[] = {0, 1, 7, 31, 257, 256, 1024, 4096, 16384, 65536};
  const size_t nc = sizeof(counts) / sizeof(counts[0]);
  int fail = 0;
  for (size_t i = 0; i < nc; ++i)
    if (run_case(counts[i]) != 0) fail = 1;
  if (fail) { printf("DEFERRED_WIDE_DOT_REDUCE_FAIL\n"); return 1; }
  printf("DEFERRED_WIDE_DOT_REDUCE_ALL_PASS\n");
  return 0;
}
