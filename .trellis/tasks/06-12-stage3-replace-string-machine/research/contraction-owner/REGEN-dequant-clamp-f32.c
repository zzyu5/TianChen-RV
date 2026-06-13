#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_pre_realized_body_product_reduce_dequant_clamp_kernel_pre_realized_body_rvv_product_reduce_dequant_clamp(const int8_t* v1, const int8_t* v2, const int32_t* v3, float v4, float v5, float v6, float* v7, size_t v8) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
  vint32m1_t v9 = __riscv_vmv_v_x_i32m1(0, 1);
  // tcrv_emitc.local_variable=grouped_tail_start source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface
  size_t v10 = 0;
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v11 = __riscv_vsetvl_e32m1(v8);
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v12 = v3[0];
  vint32m1_t v13 = __riscv_vmv_v_x_i32m1(v12, 1);
  // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
  v9 = v13;
  size_t v14 = v11 * 2;
  size_t v15 = v8 / v14;
  size_t v16 = v15 * v14;
  // tcrv_emitc.assign target=grouped_tail_start source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface
  v10 = v16;
  size_t v17 = v10;
  size_t v18 = v11 * 2;
  for (size_t v19 = 0; v19 < v17; v19 += v18) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v20 = v8 - v19;
    size_t v21 = __riscv_vsetvl_e32m1(v20);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v22 = v1 + v19;
    vint8mf4_t v23 = __riscv_vle8_v_i8mf4(v22, v21);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v24 = v2 + v19;
    vint8mf4_t v25 = __riscv_vle8_v_i8mf4(v24, v21);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v26 = __riscv_vwmul_vv_i16mf2(v23, v25, v21);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v27 = v9;
    vint32m1_t v28 = __riscv_vwredsum_vs_i16mf2_i32m1(v26, v27, v21);
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v29 = v8 - v19;
    size_t v30 = v29 - v21;
    size_t v31 = __riscv_vsetvl_e32m1(v30);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v32 = v1 + v19;
    const int8_t* v33 = v32 + v21;
    vint8mf4_t v34 = __riscv_vle8_v_i8mf4(v33, v31);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v35 = v2 + v19;
    const int8_t* v36 = v35 + v21;
    vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v31);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v38 = __riscv_vwmul_vv_i16mf2(v34, v37, v31);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v39 = __riscv_vwredsum_vs_i16mf2_i32m1(v38, v28, v31);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v9 = v28;
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v9 = v39;
  }
  size_t v40 = v10;
  for (size_t v41 = v40; v41 < v8; v41 += v11) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v42 = v8 - v41;
    size_t v43 = __riscv_vsetvl_e32m1(v42);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v44 = v1 + v41;
    vint8mf4_t v45 = __riscv_vle8_v_i8mf4(v44, v43);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v46 = v2 + v41;
    vint8mf4_t v47 = __riscv_vle8_v_i8mf4(v46, v43);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v48 = __riscv_vwmul_vv_i16mf2(v45, v47, v43);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v49 = v9;
    vint32m1_t v50 = __riscv_vwredsum_vs_i16mf2_i32m1(v48, v49, v43);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v9 = v50;
  }
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
  vint32m1_t v51 = v9;
  int32_t v52 = __riscv_vmv_x_s_i32m1_i32(v51);
  // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
  float v53 = (float) v52;
  float v54 = v53 * v4;
  vfloat32m1_t v55 = __riscv_vfmv_v_f_f32m1(v54, 1);
  // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
  vfloat32m1_t v56 = __riscv_vfmv_v_f_f32m1(v5, 1);
  // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmflt_vv_f32m1_b32
  vbool32_t v57 = __riscv_vmflt_vv_f32m1_b32(v55, v56, 1);
  // tcrv_emitc.source_op=tcrv_rvv.select role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m1
  vfloat32m1_t v58 = __riscv_vmerge_vvm_f32m1(v55, v56, v57, 1);
  // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
  vfloat32m1_t v59 = __riscv_vfmv_v_f_f32m1(v6, 1);
  // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmflt_vv_f32m1_b32
  vbool32_t v60 = __riscv_vmflt_vv_f32m1_b32(v59, v58, 1);
  // tcrv_emitc.source_op=tcrv_rvv.select role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m1
  vfloat32m1_t v61 = __riscv_vmerge_vvm_f32m1(v58, v59, v60, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
  __riscv_vse32_v_f32m1(v7, v61, 1);
  return;
}


