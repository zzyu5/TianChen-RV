#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_selected_body_rt_scalar_cmseg_load_kernel_explicit_selected_body_rvv_rt_scalar_cmseg_load(const int32_t* v1, int32_t v2, const int32_t* v3, int32_t* v4, int32_t* v5, size_t v6) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v7 = __riscv_vsetvl_e32m1(v6);
  for (size_t v8 = 0; v8 < v6; v8 += v7) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v9 = v6 - v8;
    size_t v10 = __riscv_vsetvl_e32m1(v9);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v11 = v1 + v8;
    vint32m1_t v12 = __riscv_vle32_v_i32m1(v11, v10);
    // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v13 = __riscv_vmv_v_x_i32m1(v2, v10);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    int32_t* v14 = v4 + v8;
    vint32m1_t v15 = __riscv_vle32_v_i32m1(v14, v10);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    int32_t* v16 = v5 + v8;
    vint32m1_t v17 = __riscv_vle32_v_i32m1(v16, v10);
    // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsle_vv_i32m1_b32
    vbool32_t v18 = __riscv_vmsle_vv_i32m1_b32(v12, v13, v10);
    // tcrv_emitc.source_op=tcrv_rvv.masked_segment2_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vcreate_v_i32m1x2
    vint32m1x2_t v19 = __riscv_vcreate_v_i32m1x2(v15, v17);
    // tcrv_emitc.source_op=tcrv_rvv.masked_segment2_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlseg2e32_v_i32m1x2_tumu
    size_t v20 = v8 * 2;
    const int32_t* v21 = v3 + v20;
    vint32m1x2_t v22 = __riscv_vlseg2e32_v_i32m1x2_tumu(v18, v19, v21, v10);
    // tcrv_emitc.source_op=tcrv_rvv.masked_segment2_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vget_v_i32m1x2_i32m1
    vint32m1_t v23 = __riscv_vget_v_i32m1x2_i32m1(v22, 0);
    // tcrv_emitc.source_op=tcrv_rvv.masked_segment2_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vget_v_i32m1x2_i32m1
    vint32m1_t v24 = __riscv_vget_v_i32m1x2_i32m1(v22, 1);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    int32_t* v25 = v4 + v8;
    __riscv_vse32_v_i32m1(v25, v23, v10);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    int32_t* v26 = v5 + v8;
    __riscv_vse32_v_i32m1(v26, v24, v10);
  }
  return;
}


