#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_iq4_nl_kernel_dequant_iq4_nl(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_dequant_iq4nl_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 32;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v7 = v6 * 18;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 32;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v12 = (float)*(const _Float16 *)(v8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_nibble_decode
    for (size_t v13 = 0; v13 < 16; v13 += 1) {
      size_t v14 = 2 + v13;
      const uint8_t* v15 = v8 + v14;
      const uint8_t v16 = v15[0];
      int v17 = (int) v16;
      int v18 = v17 & 15;
      int v19 = v17 >> 4;
      size_t v20 = (size_t) v18;
      const int8_t* v21 = weft_dequant_iq4nl_kvalues + v20;
      const int8_t v22 = v21[0];
      int v23 = (int) v22;
      float v24 = (float) v23;
      size_t v25 = (size_t) v19;
      const int8_t* v26 = weft_dequant_iq4nl_kvalues + v25;
      const int8_t v27 = v26[0];
      int v28 = (int) v27;
      float v29 = (float) v28;
      float v30 = v24 * v12;
      float* v31 = v11 + v13;
      v31[0] = v30;
      float v32 = v29 * v12;
      size_t v33 = v13 + 16;
      float* v34 = v11 + v33;
      v34[0] = v32;
    }
  }
  return;
}


