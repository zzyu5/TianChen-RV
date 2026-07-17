#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_selected_body_computed_masked_macc_add_kernel_explicit_selected_body_rvv_computed_masked_macc_add(const int32_t* v1, const int32_t* v2, const int32_t* v3, const int32_t* v4, const int32_t* v5, int32_t* v6, size_t v7) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v8 = __riscv_vsetvl_e32m1(v7);
  for (size_t v9 = 0; v9 < v7; v9 += v8) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v10 = v7 - v9;
    size_t v11 = __riscv_vsetvl_e32m1(v10);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v12 = v1 + v9;
    vint32m1_t v13 = __riscv_vle32_v_i32m1(v12, v11);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v14 = v2 + v9;
    vint32m1_t v15 = __riscv_vle32_v_i32m1(v14, v11);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v16 = v3 + v9;
    vint32m1_t v17 = __riscv_vle32_v_i32m1(v16, v11);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v18 = v4 + v9;
    vint32m1_t v19 = __riscv_vle32_v_i32m1(v18, v11);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v20 = v5 + v9;
    vint32m1_t v21 = __riscv_vle32_v_i32m1(v20, v11);
    // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmslt_vv_i32m1_b32
    vbool32_t v22 = __riscv_vmslt_vv_i32m1_b32(v13, v15, v11);
    // tcrv_emitc.source_op=tcrv_rvv.masked_macc role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m1
    vint32m1_t v23 = __riscv_vmacc_vv_i32m1(v21, v17, v19, v11);
    // tcrv_emitc.source_op=tcrv_rvv.masked_macc role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i32m1
    vint32m1_t v24 = __riscv_vmerge_vvm_i32m1(v21, v23, v22, v11);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    int32_t* v25 = v6 + v9;
    __riscv_vse32_v_i32m1(v25, v24, v11);
  }
  return;
}


