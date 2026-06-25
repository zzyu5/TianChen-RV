#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
#include <math.h>
extern "C" void tcrv_emitc_ggml_rms_norm_f32_kernel_ggml_rms_norm_f32(size_t v1, const float* v2, float* v3, float v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.ggml_rms_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sum source_op=tcrv_rvv.ggml_rms_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface
  double v6;
  v6 = 0.0;
  // tcrv_emitc.source_op=tcrv_rvv.ggml_rms_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scalar_double_reduce
  for (size_t v7 = 0; v7 < v1; v7 += 1) {
    const float* v8 = v2 + v7;
    const float* v9 = (const float*) v8;
    const float v10 = v9[0];
    double v11 = v6;
    // tcrv_emitc.assign target=sum source_op=tcrv_rvv.ggml_rms_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v11 + (double) (v10 * v10);
  }
  // tcrv_emitc.source_op=tcrv_rvv.ggml_rms_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=mean
  double v12 = v6;
  double v13 = (double) v1;
  double v14 = v12 / v13;
  float v15 = (float) v14;
  // tcrv_emitc.source_op=tcrv_rvv.ggml_rms_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale
  float v16 = v15 + v4;
  float v17 = sqrtf(v16);
  float v18 = 1.0f / v17;
  // tcrv_emitc.source_op=tcrv_rvv.ggml_rms_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m8
  size_t v19 = __riscv_vsetvl_e32m8(v1);
  for (size_t v20 = 0; v20 < v1; v20 += v19) {
    // tcrv_emitc.source_op=tcrv_rvv.ggml_rms_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m8
    size_t v21 = v1 - v20;
    size_t v22 = __riscv_vsetvl_e32m8(v21);
    const float* v23 = v2 + v20;
    const float* v24 = (const float*) v23;
    float* v25 = v3 + v20;
    float* v26 = (float*) v25;
    // tcrv_emitc.source_op=tcrv_rvv.ggml_rms_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_f32m8
    vfloat32m8_t v27 = __riscv_vle32_v_f32m8(v24, v22);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_rms_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v28 = __riscv_vfmul_vf_f32m8(v27, v18, v22);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_rms_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v26, v28, v22);
  }
  return;
}


