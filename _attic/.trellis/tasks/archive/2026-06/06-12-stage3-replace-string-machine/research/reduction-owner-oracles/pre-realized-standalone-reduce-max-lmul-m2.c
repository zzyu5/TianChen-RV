#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_pre_realized_body_standalone_reduce_max_lmul_m2_kernel_pre_realized_body_rvv_standalone_reduce_max_lmul_m2(const int32_t* v1, const int32_t* v2, int32_t* v3, size_t v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m2
  size_t v5 = __riscv_vsetvl_e32m2(v4);
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v6 = v2[0];
  vint32m1_t v7 = __riscv_vmv_v_x_i32m1(v6, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
  __riscv_vse32_v_i32m1(v3, v7, 1);
  for (size_t v8 = 0; v8 < v4; v8 += v5) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m2
    size_t v9 = v4 - v8;
    size_t v10 = __riscv_vsetvl_e32m2(v9);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
    const int32_t* v11 = v1 + v8;
    vint32m2_t v12 = __riscv_vle32_v_i32m2(v11, v10);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v13 = v3[0];
    vint32m1_t v14 = __riscv_vmv_v_x_i32m1(v13, 1);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vredmax_vs_i32m2_i32m1
    vint32m1_t v15 = __riscv_vredmax_vs_i32m2_i32m1(v12, v14, v10);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    __riscv_vse32_v_i32m1(v3, v15, 1);
  }
  return;
}


