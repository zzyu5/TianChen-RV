#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_budget_divergence_kernel_budget_divergence_rvv(const int8_t* v1, const int8_t* v2, const int32_t* v3, float v4, float* v5, size_t v6) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v7 = __riscv_vsetvl_e32m1(v6);
  // tcrv_emitc.local_variable=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
  vint32m1_t v8;
  size_t v9 = v7 * 2;
  size_t v10 = v6 / v9;
  size_t v11 = v10 * v9;
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v12 = v3[0];
  vint32m1_t v13 = __riscv_vmv_v_x_i32m1(v12, 1);
  // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
  v8 = v13;
  size_t v14 = v7 * 2;
  for (size_t v15 = 0; v15 < v11; v15 += v14) {
    size_t v16 = v6 - v15;
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v17 = __riscv_vsetvl_e32m1(v16);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v18 = v1 + v15;
    vint8mf4_t v19 = __riscv_vle8_v_i8mf4(v18, v17);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v20 = v2 + v15;
    vint8mf4_t v21 = __riscv_vle8_v_i8mf4(v20, v17);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v22 = __riscv_vwmul_vv_i16mf2(v19, v21, v17);
    vint32m1_t v23 = v8;
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v24 = __riscv_vwredsum_vs_i16mf2_i32m1(v22, v23, v17);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v8 = v24;
    size_t v25 = v6 - v15;
    size_t v26 = v15 + v7;
    size_t v27 = v25 - v7;
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v28 = __riscv_vsetvl_e32m1(v27);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v29 = v1 + v26;
    vint8mf4_t v30 = __riscv_vle8_v_i8mf4(v29, v28);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v31 = v2 + v26;
    vint8mf4_t v32 = __riscv_vle8_v_i8mf4(v31, v28);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v33 = __riscv_vwmul_vv_i16mf2(v30, v32, v28);
    vint32m1_t v34 = v8;
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v35 = __riscv_vwredsum_vs_i16mf2_i32m1(v33, v34, v28);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v8 = v35;
  }
  for (size_t v36 = v11; v36 < v6; v36 += v7) {
    size_t v37 = v6 - v36;
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v38 = __riscv_vsetvl_e32m1(v37);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v39 = v1 + v36;
    vint8mf4_t v40 = __riscv_vle8_v_i8mf4(v39, v38);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v41 = v2 + v36;
    vint8mf4_t v42 = __riscv_vle8_v_i8mf4(v41, v38);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v43 = __riscv_vwmul_vv_i16mf2(v40, v42, v38);
    vint32m1_t v44 = v8;
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v45 = __riscv_vwredsum_vs_i16mf2_i32m1(v43, v44, v38);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v8 = v45;
  }
  vint32m1_t v46 = v8;
  // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
  int32_t v47 = __riscv_vmv_x_s_i32m1_i32(v46);
  float v48 = (float) v47;
  float v49 = v48 * v4;
  // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
  vfloat32m1_t v50 = __riscv_vfmv_v_f_f32m1(v49, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
  __riscv_vse32_v_f32m1(v5, v50, 1);
  return;
}


