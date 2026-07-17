#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_dot_reduce_autotuner_e2e_kernel_dot_reduce_autotuner_e2e_rvv_deferred(const int16_t* v1, const int16_t* v2, const int32_t* v3, int32_t* v4, size_t v5) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e16mf2
  size_t v6 = __riscv_vsetvl_e16mf2(v5);
  // tcrv_emitc.local_variable=dot_acc_vec source_op=tcrv_rvv.deferred_accumulate role=compute op_interface=TCRVEmitCLowerableOpInterface
  vint32m1_t v7;
  // tcrv_emitc.source_op=tcrv_rvv.deferred_accumulate role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvlmax_e32m1
  size_t v8 = __riscv_vsetvlmax_e32m1();
  // tcrv_emitc.source_op=tcrv_rvv.deferred_accumulate role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  vint32m1_t v9 = __riscv_vmv_v_x_i32m1(0, v8);
  // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.deferred_accumulate role=compute op_interface=TCRVEmitCLowerableOpInterface
  v7 = v9;
  for (size_t v10 = 0; v10 < v5; v10 += v6) {
    size_t v11 = v5 - v10;
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e16mf2
    size_t v12 = __riscv_vsetvl_e16mf2(v11);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_i16mf2
    const int16_t* v13 = v1 + v10;
    vint16mf2_t v14 = __riscv_vle16_v_i16mf2(v13, v12);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_i16mf2
    const int16_t* v15 = v2 + v10;
    vint16mf2_t v16 = __riscv_vle16_v_i16mf2(v15, v12);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i32m1
    vint32m1_t v17 = __riscv_vwmul_vv_i32m1(v14, v16, v12);
    vint32m1_t v18 = v7;
    // tcrv_emitc.source_op=tcrv_rvv.deferred_accumulate role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m1
    vint32m1_t v19 = __riscv_vadd_vv_i32m1(v18, v17, v12);
    // tcrv_emitc.assign target=dot_acc_vec source_op=tcrv_rvv.deferred_accumulate role=compute op_interface=TCRVEmitCLowerableOpInterface
    v7 = v19;
  }
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvlmax_e32m1
  size_t v20 = __riscv_vsetvlmax_e32m1();
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  vint32m1_t v21 = __riscv_vmv_v_x_i32m1(0, v20);
  vint32m1_t v22 = v7;
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vredsum_vs_i32m1_i32m1
  vint32m1_t v23 = __riscv_vredsum_vs_i32m1_i32m1(v22, v21, v8);
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
  int32_t v24 = __riscv_vmv_x_s_i32m1_i32(v23);
  const int32_t v25 = v3[0];
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scalar_acc0_add
  int32_t v26 = v25 + v24;
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v26, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
  __riscv_vse32_v_i32m1(v4, v27, 1);
  return;
}


