#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_vec_cpy_f32_kernel_vec_cpy_f32(size_t v1, const float* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.elementwise_copy_map role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.elementwise_copy_map role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m8
  size_t v5 = __riscv_vsetvl_e32m8(v1);
  for (size_t v6 = 0; v6 < v1; v6 += v5) {
    // weft_emitc.source_op=weft_rvv.elementwise_copy_map role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m8
    size_t v7 = v1 - v6;
    size_t v8 = __riscv_vsetvl_e32m8(v7);
    const float* v9 = v2 + v6;
    const float* v10 = (const float*) v9;
    // weft_emitc.source_op=weft_rvv.elementwise_copy_map role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_f32m8
    vfloat32m8_t v11 = __riscv_vle32_v_f32m8(v10, v8);
    float* v12 = v3 + v6;
    float* v13 = (float*) v12;
    // weft_emitc.source_op=weft_rvv.elementwise_copy_map role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v13, v11, v8);
  }
  return;
}


