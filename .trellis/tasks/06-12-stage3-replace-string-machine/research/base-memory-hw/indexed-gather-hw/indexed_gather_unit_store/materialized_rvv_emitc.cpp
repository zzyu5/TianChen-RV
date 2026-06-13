#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_selected_body_indexed_gather_unit_store_kernel_explicit_selected_body_rvv_indexed_gather_unit_store(const int32_t* v1, const uint32_t* v2, int32_t* v3, size_t v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v4);
  for (size_t v6 = 0; v6 < v4; v6 += v5) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v7 = v4 - v6;
    size_t v8 = __riscv_vsetvl_e32m1(v7);
    // tcrv_emitc.source_op=tcrv_rvv.index_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_u32m1
    const uint32_t* v9 = v2 + v6;
    vuint32m1_t v10 = __riscv_vle32_v_u32m1(v9, v8);
    // tcrv_emitc.source_op=tcrv_rvv.indexed_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u32m1
    vuint32m1_t v11 = __riscv_vmul_vx_u32m1(v10, 4, v8);
    // tcrv_emitc.source_op=tcrv_rvv.indexed_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vloxei32_v_i32m1
    vint32m1_t v12 = __riscv_vloxei32_v_i32m1(v1, v11, v8);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    int32_t* v13 = v3 + v6;
    __riscv_vse32_v_i32m1(v13, v12, v8);
  }
  return;
}


