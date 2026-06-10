#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_pre_realized_body_product_reduce_dequant_clamp_kernel_pre_realized_body_rvv_product_reduce_dequant_clamp(const int8_t* v1, const int8_t* v2, const int32_t* v3, float v4, float v5, float v6, float* v7, size_t v8) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
  vint32m1_t v9 = __riscv_vmv_v_x_i32m1(0, 1);
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v10 = __riscv_vsetvl_e32m1(v8);
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v11 = v3[0];
  vint32m1_t v12 = __riscv_vmv_v_x_i32m1(v11, 1);
  // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
  v9 = v12;
  for (size_t v13 = 0; v13 < v8; v13 += v10) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v14 = v8 - v13;
    size_t v15 = __riscv_vsetvl_e32m1(v14);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v16 = v1 + v13;
    vint8mf4_t v17 = __riscv_vle8_v_i8mf4(v16, v15);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v18 = v2 + v13;
    vint8mf4_t v19 = __riscv_vle8_v_i8mf4(v18, v15);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8mf4
    vint8mf4_t v20 = __riscv_vsll_vx_i8mf4(v17, 4, v15);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8mf4
    vint8mf4_t v21 = __riscv_vsll_vx_i8mf4(v19, 4, v15);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v22 = __riscv_vwmul_vv_i16mf2(v20, v21, v15);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i16mf2
    vint16mf2_t v23 = __riscv_vsra_vx_i16mf2(v22, 8, v15);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf4
    vint8mf4_t v24 = __riscv_vsra_vx_i8mf4(v17, 4, v15);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf4
    vint8mf4_t v25 = __riscv_vsra_vx_i8mf4(v19, 4, v15);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v26 = __riscv_vwmul_vv_i16mf2(v24, v25, v15);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i16mf2
    vint16mf2_t v27 = __riscv_vadd_vv_i16mf2(v23, v26, v15);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v28 = v9;
    vint32m1_t v29 = __riscv_vwredsum_vs_i16mf2_i32m1(v27, v28, v15);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v9 = v29;
  }
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
  vint32m1_t v30 = v9;
  int32_t v31 = __riscv_vmv_x_s_i32m1_i32(v30);
  // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
  float v32 = (float) v31;
  float v33 = v32 * v4;
  vfloat32m1_t v34 = __riscv_vfmv_v_f_f32m1(v33, 1);
  // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
  vfloat32m1_t v35 = __riscv_vfmv_v_f_f32m1(v5, 1);
  // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmflt_vv_f32m1_b32
  vbool32_t v36 = __riscv_vmflt_vv_f32m1_b32(v34, v35, 1);
  // tcrv_emitc.source_op=tcrv_rvv.select role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m1
  vfloat32m1_t v37 = __riscv_vmerge_vvm_f32m1(v34, v35, v36, 1);
  // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
  vfloat32m1_t v38 = __riscv_vfmv_v_f_f32m1(v6, 1);
  // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmflt_vv_f32m1_b32
  vbool32_t v39 = __riscv_vmflt_vv_f32m1_b32(v38, v37, 1);
  // tcrv_emitc.source_op=tcrv_rvv.select role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m1
  vfloat32m1_t v40 = __riscv_vmerge_vvm_f32m1(v37, v38, v39, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
  __riscv_vse32_v_f32m1(v7, v40, 1);
  return;
}
