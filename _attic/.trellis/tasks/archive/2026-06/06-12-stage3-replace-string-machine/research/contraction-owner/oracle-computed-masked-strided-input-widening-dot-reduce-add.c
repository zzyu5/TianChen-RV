#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_masked_strided_dot_kernel_rvv_explicit_masked_strided_dot(const int32_t* v1, const int32_t* v2, const int16_t* v3, const int16_t* v4, const int32_t* v5, int32_t* v6, size_t v7, size_t v8, size_t v9) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v10 = __riscv_vsetvl_e32m1(v7);
  // tcrv_emitc.source_op=tcrv_rvv.masked_widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v11 = v5[0];
  vint32m1_t v12 = __riscv_vmv_v_x_i32m1(v11, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
  __riscv_vse32_v_i32m1(v6, v12, 1);
  for (size_t v13 = 0; v13 < v7; v13 += v10) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v14 = v7 - v13;
    size_t v15 = __riscv_vsetvl_e32m1(v14);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v16 = v1 + v13;
    vint32m1_t v17 = __riscv_vle32_v_i32m1(v16, v15);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v18 = v2 + v13;
    vint32m1_t v19 = __riscv_vle32_v_i32m1(v18, v15);
    // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmslt_vv_i32m1_b32
    vbool32_t v20 = __riscv_vmslt_vv_i32m1_b32(v17, v19, v15);
    // tcrv_emitc.source_op=tcrv_rvv.strided_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlse16_v_i16mf2
    size_t v21 = v13 * v8;
    const int16_t* v22 = v3 + v21;
    ptrdiff_t v23 = (ptrdiff_t) v8;
    ptrdiff_t v24 = (ptrdiff_t) 2;
    ptrdiff_t v25 = v23 * v24;
    vint16mf2_t v26 = __riscv_vlse16_v_i16mf2(v22, v25, v15);
    // tcrv_emitc.source_op=tcrv_rvv.strided_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlse16_v_i16mf2
    size_t v27 = v13 * v9;
    const int16_t* v28 = v4 + v27;
    ptrdiff_t v29 = (ptrdiff_t) v9;
    ptrdiff_t v30 = (ptrdiff_t) 2;
    ptrdiff_t v31 = v29 * v30;
    vint16mf2_t v32 = __riscv_vlse16_v_i16mf2(v28, v31, v15);
    // tcrv_emitc.source_op=tcrv_rvv.masked_widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v33 = __riscv_vmv_v_x_i32m1(0, v15);
    // tcrv_emitc.source_op=tcrv_rvv.masked_widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i32m1_m
    vint32m1_t v34 = __riscv_vwmul_vv_i32m1_m(v20, v26, v32, v15);
    // tcrv_emitc.source_op=tcrv_rvv.masked_widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i32m1
    vint32m1_t v35 = __riscv_vmerge_vvm_i32m1(v33, v34, v20, v15);
    // tcrv_emitc.source_op=tcrv_rvv.masked_widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v36 = v6[0];
    vint32m1_t v37 = __riscv_vmv_v_x_i32m1(v36, 1);
    // tcrv_emitc.source_op=tcrv_rvv.masked_widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vredsum_vs_i32m1_i32m1
    vint32m1_t v38 = __riscv_vredsum_vs_i32m1_i32m1(v35, v37, v15);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    __riscv_vse32_v_i32m1(v6, v38, 1);
  }
  return;
}


