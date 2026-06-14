#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_selected_body_unit_load_strided_store_kernel_explicit_selected_body_rvv_unit_load_strided_store(const int32_t* v1, int32_t* v2, size_t v3, size_t v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v3);
  for (size_t v6 = 0; v6 < v3; v6 += v5) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v7 = v3 - v6;
    size_t v8 = __riscv_vsetvl_e32m1(v7);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v9 = v1 + v6;
    vint32m1_t v10 = __riscv_vle32_v_i32m1(v9, v8);
    // tcrv_emitc.source_op=tcrv_rvv.strided_store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsse32_v_i32m1
    uint8_t* v11 = (uint8_t*) v2;
    size_t v12 = v6 * v4;
    uint8_t* v13 = v11 + v12;
    int32_t* v14 = (int32_t*) v13;
    __riscv_vsse32_v_i32m1(v14, v4, v10, v8);
  }
  return;
}


