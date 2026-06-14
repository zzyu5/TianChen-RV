#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_cm_standalone_reduce_max_kernel_rvv_cm_standalone_reduce_max(const int32_t* v1, const int32_t* v2, const int32_t* v3, const int32_t* v4, int32_t* v5, size_t v6) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v7 = __riscv_vsetvl_e32m1(v6);
  // tcrv_emitc.source_op=tcrv_rvv.masked_standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v8 = v4[0];
  vint32m1_t v9 = __riscv_vmv_v_x_i32m1(v8, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
  __riscv_vse32_v_i32m1(v5, v9, 1);
  for (size_t v10 = 0; v10 < v6; v10 += v7) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v11 = v6 - v10;
    size_t v12 = __riscv_vsetvl_e32m1(v11);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v13 = v1 + v10;
    vint32m1_t v14 = __riscv_vle32_v_i32m1(v13, v12);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v15 = v2 + v10;
    vint32m1_t v16 = __riscv_vle32_v_i32m1(v15, v12);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v17 = v3 + v10;
    vint32m1_t v18 = __riscv_vle32_v_i32m1(v17, v12);
    // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsle_vv_i32m1_b32
    vbool32_t v19 = __riscv_vmsle_vv_i32m1_b32(v14, v16, v12);
    // tcrv_emitc.source_op=tcrv_rvv.masked_standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v20 = __riscv_vmv_v_x_i32m1((-2147483647-1), v12);
    // tcrv_emitc.source_op=tcrv_rvv.masked_standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i32m1
    vint32m1_t v21 = __riscv_vmerge_vvm_i32m1(v20, v18, v19, v12);
    // tcrv_emitc.source_op=tcrv_rvv.masked_standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v22 = v5[0];
    vint32m1_t v23 = __riscv_vmv_v_x_i32m1(v22, 1);
    // tcrv_emitc.source_op=tcrv_rvv.masked_standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vredmax_vs_i32m1_i32m1
    vint32m1_t v24 = __riscv_vredmax_vs_i32m1_i32m1(v21, v23, v12);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    __riscv_vse32_v_i32m1(v5, v24, 1);
  }
  return;
}


