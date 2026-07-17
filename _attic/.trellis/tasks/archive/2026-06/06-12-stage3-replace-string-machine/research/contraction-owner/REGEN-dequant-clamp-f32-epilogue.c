#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_pre_realized_dequant_clamp_f32_epilogue_kernel_pre_realized_rvv_dequant_clamp_f32_epilogue(const int32_t* v1, float v2, float v3, float v4, float* v5, size_t v6) {
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
    // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v13 = __riscv_vfcvt_f_x_v_f32m1(v12, v10);
    // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v14 = __riscv_vfmul_vf_f32m1(v13, v2, v10);
    // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
    vfloat32m1_t v15 = __riscv_vfmv_v_f_f32m1(v3, v10);
    // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
    vfloat32m1_t v16 = __riscv_vfmv_v_f_f32m1(v4, v10);
    // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmflt_vv_f32m1_b32
    vbool32_t v17 = __riscv_vmflt_vv_f32m1_b32(v14, v15, v10);
    // tcrv_emitc.source_op=tcrv_rvv.select role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m1
    vfloat32m1_t v18 = __riscv_vmerge_vvm_f32m1(v14, v15, v17, v10);
    // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmflt_vv_f32m1_b32
    vbool32_t v19 = __riscv_vmflt_vv_f32m1_b32(v16, v18, v10);
    // tcrv_emitc.source_op=tcrv_rvv.select role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m1
    vfloat32m1_t v20 = __riscv_vmerge_vvm_f32m1(v18, v16, v19, v10);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    float* v21 = v5 + v8;
    __riscv_vse32_v_f32m1(v21, v20, v10);
  }
  return;
}


