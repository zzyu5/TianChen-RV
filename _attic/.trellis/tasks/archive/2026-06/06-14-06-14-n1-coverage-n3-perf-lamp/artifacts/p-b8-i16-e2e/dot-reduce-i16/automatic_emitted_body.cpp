/* P-B8: the AUTOMATIC e2e compiler output for the i16 dot-reduce kernel.
 * Produced by: tcrv-opt KERNEL --tcrv-rvv-materialize-gearbox-schedules
 *   --tcrv-materialize-selected-lowering-boundaries (the resource-aware
 *   selector picks the wide i32m8 rung at budget 32 and REALIZES the
 *   deferred-wide dot-reduce body) | --tcrv-rvv-lower-to-emitc
 *   | mlir-translate --mlir-to-cpp. NOT hand-written. Byte-identical body
 *   to the P-B7 measured winner (only the symbol name differs). */
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv(const int16_t* v1, const int16_t* v2, const int32_t* v3, int32_t* v4, size_t v5) {
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
