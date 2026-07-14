#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
#include <math.h> /* harness-build: emitted gelu uses tanhf; emitter omits this include */
extern "C" void weft_emitc_gelu_f32_kernel_gelu_f32(size_t v1, const float* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.elementwise_gelu_map role=compute op_interface=WEFTEmitCLowerableOpInterface
  for (size_t v5 = 0; v5 < v1; v5 += 1) {
    const float* v6 = v2 + v5;
    const float* v7 = (const float*) v6;
    const float v8 = v7[0];
    float v9 = v8 * v8;
    float v10 = 0.044715f * v9;
    float v11 = 1.0f + v10;
    float v12 = 0.79788456080286535587989211986876f * v8;
    float v13 = v12 * v11;
    // weft_emitc.source_op=weft_rvv.elementwise_gelu_map role=compute op_interface=WEFTEmitCLowerableOpInterface callee=tanhf
    float v14 = tanhf(v13);
    float v15 = 1.0f + v14;
    float v16 = 0.5f * v8;
    float v17 = v16 * v15;
    float* v18 = v3 + v5;
    float* v19 = (float*) v18;
    v19[0] = v17;
  }
  return;
}


