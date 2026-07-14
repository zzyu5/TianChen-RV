#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
#include <math.h>
extern "C" void weft_emitc_ggml_rope_norm_f32_kernel_ggml_rope_norm_f32(size_t v1, const float* v2, float* v3, float v4, float v5) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.elementwise_rope_rotate_core role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.local_variable=theta source_op=weft_rvv.elementwise_rope_rotate_core role=compute op_interface=WEFTEmitCLowerableOpInterface
  float v7;
  v7 = v4;
  // weft_emitc.source_op=weft_rvv.elementwise_rope_rotate_core role=compute op_interface=WEFTEmitCLowerableOpInterface callee=pair_count
  size_t v8 = v1 / 2;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    float v10 = v7;
    // weft_emitc.source_op=weft_rvv.elementwise_rope_rotate_core role=compute op_interface=WEFTEmitCLowerableOpInterface callee=cosf
    float v11 = cosf(v10);
    // weft_emitc.source_op=weft_rvv.elementwise_rope_rotate_core role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sinf
    float v12 = sinf(v10);
    // weft_emitc.source_op=weft_rvv.elementwise_rope_rotate_core role=compute op_interface=WEFTEmitCLowerableOpInterface callee=pair_ptr
    size_t v13 = v9 * 2;
    const float* v14 = v2 + v13;
    const float* v15 = (const float*) v14;
    float* v16 = v3 + v13;
    float* v17 = (float*) v16;
    const float v18 = v15[0];
    const float v19 = v15[1];
    // weft_emitc.source_op=weft_rvv.elementwise_rope_rotate_core role=compute op_interface=WEFTEmitCLowerableOpInterface callee=rotate_lo
    v17[0] = v18 * v11 - v19 * v12;
    // weft_emitc.source_op=weft_rvv.elementwise_rope_rotate_core role=compute op_interface=WEFTEmitCLowerableOpInterface callee=rotate_hi
    v17[1] = v18 * v12 + v19 * v11;
    // weft_emitc.assign target=theta source_op=weft_rvv.elementwise_rope_rotate_core role=compute op_interface=WEFTEmitCLowerableOpInterface
    float v20 = v10 * v5;
    v7 = v20;
  }
  return;
}


