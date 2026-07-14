#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_vec_add_f32_kernel_vec_add_f32(size_t v1, const float* v2, const float* v3, float* v4) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.elementwise_binary_map role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.elementwise_binary_map role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m8
  size_t v6 = __riscv_vsetvl_e32m8(v1);
  for (size_t v7 = 0; v7 < v1; v7 += v6) {
    // weft_emitc.source_op=weft_rvv.elementwise_binary_map role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m8
    size_t v8 = v1 - v7;
    size_t v9 = __riscv_vsetvl_e32m8(v8);
    const float* v10 = v2 + v7;
    const float* v11 = (const float*) v10;
    // weft_emitc.source_op=weft_rvv.elementwise_binary_map role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_f32m8
    vfloat32m8_t v12 = __riscv_vle32_v_f32m8(v11, v9);
    const float* v13 = v3 + v7;
    const float* v14 = (const float*) v13;
    // weft_emitc.source_op=weft_rvv.elementwise_binary_map role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_f32m8
    vfloat32m8_t v15 = __riscv_vle32_v_f32m8(v14, v9);
    // weft_emitc.source_op=weft_rvv.elementwise_binary_map role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m8
    vfloat32m8_t v16 = __riscv_vfadd_vv_f32m8(v12, v15, v9);
    float* v17 = v4 + v7;
    float* v18 = (float*) v17;
    // weft_emitc.source_op=weft_rvv.elementwise_binary_map role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v18, v16, v9);
  }
  return;
}


