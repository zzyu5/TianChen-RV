#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(const int8_t* v1, const int8_t* v2, const int32_t* v3, float v4, float* v5, size_t v6) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
  vint32m1_t v7 = __riscv_vmv_v_x_i32m1(0, 1);
  // tcrv_emitc.local_variable=grouped_tail_start source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface
  size_t v8 = 0;
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v9 = __riscv_vsetvl_e32m1(v6);
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v10 = v3[0];
  vint32m1_t v11 = __riscv_vmv_v_x_i32m1(v10, 1);
  // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
  v7 = v11;
  size_t v12 = v9 * 2;
  size_t v13 = v6 / v12;
  size_t v14 = v13 * v12;
  // tcrv_emitc.assign target=grouped_tail_start source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface
  v8 = v14;
  size_t v15 = v8;
  size_t v16 = v9 * 2;
  for (size_t v17 = 0; v17 < v15; v17 += v16) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v18 = v6 - v17;
    size_t v19 = __riscv_vsetvl_e32m1(v18);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v20 = v1 + v17;
    vint8mf4_t v21 = __riscv_vle8_v_i8mf4(v20, v19);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v22 = v2 + v17;
    vint8mf4_t v23 = __riscv_vle8_v_i8mf4(v22, v19);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v24 = __riscv_vwmul_vv_i16mf2(v21, v23, v19);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v25 = v7;
    vint32m1_t v26 = __riscv_vwredsum_vs_i16mf2_i32m1(v24, v25, v19);
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v27 = v6 - v17;
    size_t v28 = v27 - v19;
    size_t v29 = __riscv_vsetvl_e32m1(v28);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v30 = v1 + v17;
    const int8_t* v31 = v30 + v19;
    vint8mf4_t v32 = __riscv_vle8_v_i8mf4(v31, v29);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v33 = v2 + v17;
    const int8_t* v34 = v33 + v19;
    vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v29);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v36 = __riscv_vwmul_vv_i16mf2(v32, v35, v29);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v37 = __riscv_vwredsum_vs_i16mf2_i32m1(v36, v26, v29);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v7 = v26;
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v7 = v37;
  }
  size_t v38 = v8;
  for (size_t v39 = v38; v39 < v6; v39 += v9) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v40 = v6 - v39;
    size_t v41 = __riscv_vsetvl_e32m1(v40);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v42 = v1 + v39;
    vint8mf4_t v43 = __riscv_vle8_v_i8mf4(v42, v41);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v44 = v2 + v39;
    vint8mf4_t v45 = __riscv_vle8_v_i8mf4(v44, v41);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v46 = __riscv_vwmul_vv_i16mf2(v43, v45, v41);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v47 = v7;
    vint32m1_t v48 = __riscv_vwredsum_vs_i16mf2_i32m1(v46, v47, v41);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface
    v7 = v48;
  }
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
  vint32m1_t v49 = v7;
  int32_t v50 = __riscv_vmv_x_s_i32m1_i32(v49);
  // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
  float v51 = (float) v50;
  float v52 = v51 * v4;
  vfloat32m1_t v53 = __riscv_vfmv_v_f_f32m1(v52, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
  __riscv_vse32_v_f32m1(v5, v53, 1);
  return;
}


