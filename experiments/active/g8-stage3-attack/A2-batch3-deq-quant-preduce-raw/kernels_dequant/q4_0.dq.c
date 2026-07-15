#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_q4_0_kernel_dequant_q4_0(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v5 = v1 / 32;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=x_block
    size_t v7 = v6 * 18;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 32;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=d
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v12 = (float)*(const _Float16 *)(v8);
    const uint8_t* v13 = v8 + 2;
    const uint8_t* v14 = (const uint8_t*) v13;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=nibble_decode
    for (size_t v15 = 0; v15 < 16; v15 += 1) {
      const uint8_t* v16 = v14 + v15;
      const uint8_t v17 = v16[0];
      int v18 = (int) v17;
      int v19 = v18 & 15;
      int v20 = v18 >> 4;
      int v21 = v19 - 8;
      int v22 = v20 - 8;
      float v23 = (float) v21;
      float v24 = (float) v22;
      float v25 = v23 * v12;
      float v26 = v24 * v12;
      float* v27 = v11 + v15;
      float* v28 = v11 + v15;
      float* v29 = v28 + 16;
      v27[0] = v25;
      v29[0] = v26;
    }
  }
  return;
}


