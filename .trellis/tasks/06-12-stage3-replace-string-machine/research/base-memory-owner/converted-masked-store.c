#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_pre_realized_body_masked_unit_store_kernel_pre_realized_body_rvv_masked_unit_store(const int32_t* v1, const int32_t* v2, int32_t* v3, size_t v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v4);
  for (size_t v6 = 0; v6 < v4; v6 += v5) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v7 = v4 - v6;
    size_t v8 = __riscv_vsetvl_e32m1(v7);
    // tcrv_emitc.source_op=tcrv_rvv.mask_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v9 = v2 + v6;
    vint32m1_t v10 = __riscv_vle32_v_i32m1(v9, v8);
    // tcrv_emitc.source_op=tcrv_rvv.mask_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_i32m1_b32
    vbool32_t v11 = __riscv_vmsne_vx_i32m1_b32(v10, 0, v8);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v12 = v1 + v6;
    vint32m1_t v13 = __riscv_vle32_v_i32m1(v12, v8);
    // tcrv_emitc.source_op=tcrv_rvv.masked_store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1_m
    int32_t* v14 = v3 + v6;
    __riscv_vse32_v_i32m1_m(v11, v14, v13, v8);
  }
  return;
}


