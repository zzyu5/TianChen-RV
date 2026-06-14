#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_selected_body_dequantize_i32_to_f32_kernel_explicit_selected_body_rvv_dequantize_i32_to_f32(const int32_t* v1, float v2, float* v3, size_t v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v4);
  size_t v6 = v5 * 2;
  for (size_t v7 = 0; v7 < v4; v7 += v6) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v8 = v4 - v7;
    size_t v9 = __riscv_vsetvl_e32m1(v8);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v10 = v1 + v7;
    vint32m1_t v11 = __riscv_vle32_v_i32m1(v10, v9);
    // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v12 = __riscv_vfcvt_f_x_v_f32m1(v11, v9);
    // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v13 = __riscv_vfmul_vf_f32m1(v12, v2, v9);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    float* v14 = v3 + v7;
    __riscv_vse32_v_f32m1(v14, v13, v9);
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v15 = v4 - v7;
    size_t v16 = v15 - v9;
    size_t v17 = __riscv_vsetvl_e32m1(v16);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v18 = v1 + v7;
    const int32_t* v19 = v18 + v9;
    vint32m1_t v20 = __riscv_vle32_v_i32m1(v19, v17);
    // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v21 = __riscv_vfcvt_f_x_v_f32m1(v20, v17);
    // tcrv_emitc.source_op=tcrv_rvv.dequantize role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v22 = __riscv_vfmul_vf_f32m1(v21, v2, v17);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    float* v23 = v3 + v7;
    float* v24 = v23 + v9;
    __riscv_vse32_v_f32m1(v24, v22, v17);
  }
  return;
}


