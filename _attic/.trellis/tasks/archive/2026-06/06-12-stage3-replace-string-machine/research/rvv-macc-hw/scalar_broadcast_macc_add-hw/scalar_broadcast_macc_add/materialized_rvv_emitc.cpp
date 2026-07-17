#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_selected_body_scalar_broadcast_macc_add_kernel_explicit_selected_body_rvv_scalar_broadcast_macc_add(const int32_t* v1, int32_t v2, const int32_t* v3, int32_t* v4, size_t v5) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v5);
  for (size_t v7 = 0; v7 < v5; v7 += v6) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v8 = v5 - v7;
    size_t v9 = __riscv_vsetvl_e32m1(v8);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v10 = v1 + v7;
    vint32m1_t v11 = __riscv_vle32_v_i32m1(v10, v9);
    // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v12 = __riscv_vmv_v_x_i32m1(v2, v9);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v13 = v3 + v7;
    vint32m1_t v14 = __riscv_vle32_v_i32m1(v13, v9);
    // tcrv_emitc.source_op=tcrv_rvv.macc role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m1
    vint32m1_t v15 = __riscv_vmacc_vv_i32m1(v14, v11, v12, v9);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    int32_t* v16 = v4 + v7;
    __riscv_vse32_v_i32m1(v16, v15, v9);
  }
  return;
}


