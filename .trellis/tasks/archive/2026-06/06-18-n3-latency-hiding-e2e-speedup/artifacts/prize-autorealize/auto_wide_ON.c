#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_budget_divergence_kernel_budget_divergence_rvv(const int8_t* v1, const int8_t* v2, const int32_t* v3, float v4, float* v5, size_t v6) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
  size_t v7 = __riscv_vsetvl_e8m2(v6);
  // tcrv_emitc.local_variable=dot_acc_vec source_op=tcrv_rvv.widening_accumulate role=compute op_interface=TCRVEmitCLowerableOpInterface
  vint32m8_t v8;
  // tcrv_emitc.source_op=tcrv_rvv.widening_accumulate role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvlmax_e32m8
  size_t v9 = __riscv_vsetvlmax_e32m8();
  // tcrv_emitc.source_op=tcrv_rvv.widening_accumulate role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m8
  vint32m8_t v10 = __riscv_vmv_v_x_i32m8(0, v9);
  // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.widening_accumulate role=compute op_interface=TCRVEmitCLowerableOpInterface
  v8 = v10;
  for (size_t v11 = 0; v11 < v6; v11 += v7) {
    size_t v12 = v6 - v11;
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v13 = __riscv_vsetvl_e8m2(v12);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    const int8_t* v14 = v1 + v11;
    vint8m2_t v15 = __riscv_vle8_v_i8m2(v14, v13);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    const int8_t* v16 = v2 + v11;
    vint8m2_t v17 = __riscv_vle8_v_i8m2(v16, v13);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v18 = __riscv_vwmul_vv_i16m4(v15, v17, v13);
    vint32m8_t v19 = v8;
    // tcrv_emitc.source_op=tcrv_rvv.widening_accumulate role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m8
    vint32m8_t v20 = __riscv_vwadd_wv_i32m8(v19, v18, v13);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.widening_accumulate role=compute op_interface=TCRVEmitCLowerableOpInterface
    v8 = v20;
  }
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvlmax_e32m1
  size_t v21 = __riscv_vsetvlmax_e32m1();
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(0, v21);
  vint32m8_t v23 = v8;
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vredsum_vs_i32m8_i32m1
  vint32m1_t v24 = __riscv_vredsum_vs_i32m8_i32m1(v23, v22, v9);
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
  int32_t v25 = __riscv_vmv_x_s_i32m1_i32(v24);
  const int32_t v26 = v3[0];
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scalar_acc0_add
  int32_t v27 = v26 + v25;
  float v28 = (float) v27;
  float v29 = v28 * v4;
  // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
  vfloat32m1_t v30 = __riscv_vfmv_v_f_f32m1(v29, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
  __riscv_vse32_v_f32m1(v5, v30, 1);
  return;
}


