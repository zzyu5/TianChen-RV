#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_pre_rt_scalar_cm_standalone_reduce_i64_kernel_rvv_pre_rt_scalar_cm_standalone_reduce_i64(const int64_t* v1, int64_t v2, const int64_t* v3, const int64_t* v4, int64_t* v5, size_t v6) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e64m1
  size_t v7 = __riscv_vsetvl_e64m1(v6);
  // tcrv_emitc.source_op=tcrv_rvv.masked_standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i64m1
  const int64_t v8 = v4[0];
  vint64m1_t v9 = __riscv_vmv_v_x_i64m1(v8, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse64_v_i64m1
  __riscv_vse64_v_i64m1(v5, v9, 1);
  for (size_t v10 = 0; v10 < v6; v10 += v7) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e64m1
    size_t v11 = v6 - v10;
    size_t v12 = __riscv_vsetvl_e64m1(v11);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle64_v_i64m1
    const int64_t* v13 = v1 + v10;
    vint64m1_t v14 = __riscv_vle64_v_i64m1(v13, v12);
    // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i64m1
    vint64m1_t v15 = __riscv_vmv_v_x_i64m1(v2, v12);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle64_v_i64m1
    const int64_t* v16 = v3 + v10;
    vint64m1_t v17 = __riscv_vle64_v_i64m1(v16, v12);
    // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsle_vv_i64m1_b64
    vbool64_t v18 = __riscv_vmsle_vv_i64m1_b64(v14, v15, v12);
    // tcrv_emitc.source_op=tcrv_rvv.masked_standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i64m1
    vint64m1_t v19 = __riscv_vmv_v_x_i64m1(0, v12);
    // tcrv_emitc.source_op=tcrv_rvv.masked_standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i64m1
    vint64m1_t v20 = __riscv_vmerge_vvm_i64m1(v19, v17, v18, v12);
    // tcrv_emitc.source_op=tcrv_rvv.masked_standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i64m1
    int64_t v21 = v5[0];
    vint64m1_t v22 = __riscv_vmv_v_x_i64m1(v21, 1);
    // tcrv_emitc.source_op=tcrv_rvv.masked_standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vredsum_vs_i64m1_i64m1
    vint64m1_t v23 = __riscv_vredsum_vs_i64m1_i64m1(v20, v22, v12);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse64_v_i64m1
    __riscv_vse64_v_i64m1(v5, v23, 1);
  }
  return;
}


