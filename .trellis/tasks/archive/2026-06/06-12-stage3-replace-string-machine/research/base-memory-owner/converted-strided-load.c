#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_selected_body_strided_load_unit_store_kernel_explicit_selected_body_rvv_strided_load_unit_store(const int32_t* v1, int32_t* v2, size_t v3, size_t v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v3);
  for (size_t v6 = 0; v6 < v3; v6 += v5) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v7 = v3 - v6;
    size_t v8 = __riscv_vsetvl_e32m1(v7);
    // tcrv_emitc.source_op=tcrv_rvv.strided_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlse32_v_i32m1
    const uint8_t* v9 = (const uint8_t*) v1;
    size_t v10 = v6 * v4;
    const uint8_t* v11 = v9 + v10;
    const int32_t* v12 = (const int32_t*) v11;
    vint32m1_t v13 = __riscv_vlse32_v_i32m1(v12, v4, v8);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    int32_t* v14 = v2 + v6;
    __riscv_vse32_v_i32m1(v14, v13, v8);
  }
  return;
}


