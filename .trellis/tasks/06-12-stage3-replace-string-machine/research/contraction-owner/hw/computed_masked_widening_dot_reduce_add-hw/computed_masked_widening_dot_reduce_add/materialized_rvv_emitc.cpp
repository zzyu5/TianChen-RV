#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_masked_wdot_kernel_rvv_explicit_masked_wdot(const int32_t* v1, const int32_t* v2, const int16_t* v3, const int16_t* v4, const int32_t* v5, int32_t* v6, size_t v7) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v8 = __riscv_vsetvl_e32m1(v7);
  // tcrv_emitc.source_op=tcrv_rvv.masked_widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v9 = v5[0];
  vint32m1_t v10 = __riscv_vmv_v_x_i32m1(v9, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
  __riscv_vse32_v_i32m1(v6, v10, 1);
  for (size_t v11 = 0; v11 < v7; v11 += v8) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v12 = v7 - v11;
    size_t v13 = __riscv_vsetvl_e32m1(v12);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v14 = v1 + v11;
    vint32m1_t v15 = __riscv_vle32_v_i32m1(v14, v13);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v16 = v2 + v11;
    vint32m1_t v17 = __riscv_vle32_v_i32m1(v16, v13);
    // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmslt_vv_i32m1_b32
    vbool32_t v18 = __riscv_vmslt_vv_i32m1_b32(v15, v17, v13);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_i16mf2
    const int16_t* v19 = v3 + v11;
    vint16mf2_t v20 = __riscv_vle16_v_i16mf2(v19, v13);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_i16mf2
    const int16_t* v21 = v4 + v11;
    vint16mf2_t v22 = __riscv_vle16_v_i16mf2(v21, v13);
    // tcrv_emitc.source_op=tcrv_rvv.masked_widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v23 = __riscv_vmv_v_x_i32m1(0, v13);
    // tcrv_emitc.source_op=tcrv_rvv.masked_widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i32m1_m
    vint32m1_t v24 = __riscv_vwmul_vv_i32m1_m(v18, v20, v22, v13);
    // tcrv_emitc.source_op=tcrv_rvv.masked_widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i32m1
    vint32m1_t v25 = __riscv_vmerge_vvm_i32m1(v23, v24, v18, v13);
    // tcrv_emitc.source_op=tcrv_rvv.masked_widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v26 = v6[0];
    vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v26, 1);
    // tcrv_emitc.source_op=tcrv_rvv.masked_widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vredsum_vs_i32m1_i32m1
    vint32m1_t v28 = __riscv_vredsum_vs_i32m1_i32m1(v25, v27, v13);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    __riscv_vse32_v_i32m1(v6, v28, 1);
  }
  return;
}


