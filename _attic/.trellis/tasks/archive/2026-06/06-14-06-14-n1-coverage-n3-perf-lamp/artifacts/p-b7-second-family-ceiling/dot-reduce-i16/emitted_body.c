/* VERBATIM compiler-emitted deferred-wide i16 dot-reduce body, isolated as its
 * own C TU for the 灯 measure. Source: tcrv-opt
 * test/Conversion/RVV/rvv-to-emitc-widening-dot-reduce-wide-lmul.mlir
 * --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
 * (the C++ 'extern "C"' wrapper dropped; the body is plain C). NOT hand-written. */
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

void tcrv_emitc_wide_dot_reduce_kernel_wide_dot_reduce(const int16_t* v1, const int16_t* v2, const int32_t* v3, int32_t* v4, size_t v5) {
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
