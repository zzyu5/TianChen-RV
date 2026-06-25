#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
#include <math.h>
extern "C" void tcrv_emitc_ggml_rope_norm_f32_kernel_ggml_rope_norm_f32(size_t v1, const float* v2, float* v3, float v4, float v5) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.ggml_rope_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=theta source_op=tcrv_rvv.ggml_rope_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v7;
  v7 = v4;
  // tcrv_emitc.source_op=tcrv_rvv.ggml_rope_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=pair_count
  size_t v8 = v1 / 2;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    float v10 = v7;
    // tcrv_emitc.source_op=tcrv_rvv.ggml_rope_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=cosf
    float v11 = cosf(v10);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_rope_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sinf
    float v12 = sinf(v10);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_rope_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=pair_ptr
    size_t v13 = v9 * 2;
    const float* v14 = v2 + v13;
    const float* v15 = (const float*) v14;
    float* v16 = v3 + v13;
    float* v17 = (float*) v16;
    const float v18 = v15[0];
    const float v19 = v15[1];
    // tcrv_emitc.source_op=tcrv_rvv.ggml_rope_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=rotate_lo
    v17[0] = v18 * v11 - v19 * v12;
    // tcrv_emitc.source_op=tcrv_rvv.ggml_rope_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=rotate_hi
    v17[1] = v18 * v12 + v19 * v11;
    // tcrv_emitc.assign target=theta source_op=tcrv_rvv.ggml_rope_norm_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface
    float v20 = v10 * v5;
    v7 = v20;
  }
  return;
}


