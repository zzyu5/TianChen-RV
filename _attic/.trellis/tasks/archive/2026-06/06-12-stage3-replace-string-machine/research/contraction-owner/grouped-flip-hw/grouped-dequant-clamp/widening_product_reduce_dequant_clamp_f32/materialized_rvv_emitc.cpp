#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_pre_realized_body_product_reduce_dequant_clamp_kernel_pre_realized_body_rvv_product_reduce_dequant_clamp(const int8_t* v1, const int8_t* v2, const int32_t* v3, float v4, float v5, float v6, float* v7, size_t v8) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v9 = __riscv_vsetvl_e32m1(v8);
  // tcrv_emitc.local_variable=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
  vint32m1_t v10;
  size_t v11 = v9 * 2;
  size_t v12 = v8 / v11;
  size_t v13 = v12 * v11;
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v14 = v3[0];
  vint32m1_t v15 = __riscv_vmv_v_x_i32m1(v14, 1);
  // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
  v10 = v15;
  size_t v16 = v9 * 2;
  for (size_t v17 = 0; v17 < v13; v17 += v16) {
    size_t v18 = v8 - v17;
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v19 = __riscv_vsetvl_e32m1(v18);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v20 = v1 + v17;
    vint8mf4_t v21 = __riscv_vle8_v_i8mf4(v20, v19);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v22 = v2 + v17;
    vint8mf4_t v23 = __riscv_vle8_v_i8mf4(v22, v19);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v24 = __riscv_vwmul_vv_i16mf2(v21, v23, v19);
    vint32m1_t v25 = v10;
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v26 = __riscv_vwredsum_vs_i16mf2_i32m1(v24, v25, v19);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v26;
    size_t v27 = v8 - v17;
    size_t v28 = v17 + v9;
    size_t v29 = v27 - v9;
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v30 = __riscv_vsetvl_e32m1(v29);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v31 = v1 + v28;
    vint8mf4_t v32 = __riscv_vle8_v_i8mf4(v31, v30);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v33 = v2 + v28;
    vint8mf4_t v34 = __riscv_vle8_v_i8mf4(v33, v30);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v35 = __riscv_vwmul_vv_i16mf2(v32, v34, v30);
    vint32m1_t v36 = v10;
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v37 = __riscv_vwredsum_vs_i16mf2_i32m1(v35, v36, v30);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v37;
  }
  for (size_t v38 = v13; v38 < v8; v38 += v9) {
    size_t v39 = v8 - v38;
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v40 = __riscv_vsetvl_e32m1(v39);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v41 = v1 + v38;
    vint8mf4_t v42 = __riscv_vle8_v_i8mf4(v41, v40);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v43 = v2 + v38;
    vint8mf4_t v44 = __riscv_vle8_v_i8mf4(v43, v40);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v45 = __riscv_vwmul_vv_i16mf2(v42, v44, v40);
    vint32m1_t v46 = v10;
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v47 = __riscv_vwredsum_vs_i16mf2_i32m1(v45, v46, v40);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v47;
  }
  vint32m1_t v48 = v10;
  // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
  int32_t v49 = __riscv_vmv_x_s_i32m1_i32(v48);
  float v50 = (float) v49;
  float v51 = v50 * v4;
  // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
  vfloat32m1_t v52 = __riscv_vfmv_v_f_f32m1(v51, 1);
  // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
  vfloat32m1_t v53 = __riscv_vfmv_v_f_f32m1(v5, 1);
  // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
  vfloat32m1_t v54 = __riscv_vfmv_v_f_f32m1(v6, 1);
  // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmflt_vv_f32m1_b32
  vbool32_t v55 = __riscv_vmflt_vv_f32m1_b32(v52, v53, 1);
  // tcrv_emitc.source_op=tcrv_rvv.select role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m1
  vfloat32m1_t v56 = __riscv_vmerge_vvm_f32m1(v52, v53, v55, 1);
  // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmflt_vv_f32m1_b32
  vbool32_t v57 = __riscv_vmflt_vv_f32m1_b32(v54, v56, 1);
  // tcrv_emitc.source_op=tcrv_rvv.select role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m1
  vfloat32m1_t v58 = __riscv_vmerge_vvm_f32m1(v56, v54, v57, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
  __riscv_vse32_v_f32m1(v7, v58, 1);
  return;
}


