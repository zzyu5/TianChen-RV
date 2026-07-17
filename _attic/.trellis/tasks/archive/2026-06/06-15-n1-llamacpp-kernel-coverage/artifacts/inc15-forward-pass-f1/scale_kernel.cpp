#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_scale_f32_kernel_ggml_vec_scale_f32(size_t v1, float* v2, float v3) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.ggml_vec_scale_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_scale_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m8
  size_t v5 = __riscv_vsetvl_e32m8(v1);
  for (size_t v6 = 0; v6 < v1; v6 += v5) {
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_scale_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m8
    size_t v7 = v1 - v6;
    size_t v8 = __riscv_vsetvl_e32m8(v7);
    float* v9 = v2 + v6;
    float* v10 = (float*) v9;
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_scale_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_f32m8
    vfloat32m8_t v11 = __riscv_vle32_v_f32m8(v10, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_scale_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v12 = __riscv_vfmul_vf_f32m8(v11, v3, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_scale_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v10, v12, v8);
  }
  return;
}


