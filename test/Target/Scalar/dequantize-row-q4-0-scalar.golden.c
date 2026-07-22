#include <stddef.h>
#include <stdint.h>
extern "C" void weft_emitc_q4_0_dequant_kernel_scalar_fallback_first_slice(int v1, float* v2, const uint8_t* v3) {
  // weft_emitc.route_source_op=weft_scalar.dequantize_row_q4_0 role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_scalar.dequantize_row_q4_0 role=compute step=block_count
  size_t v4 = (size_t) v1;
  size_t v5 = v4 / 32;
  // weft_emitc.source_op=weft_scalar.dequantize_row_q4_0 role=compute step=block_loop
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    size_t v7 = v6 * 18;
    const uint8_t* v8 = v3 + v7;
    // weft_emitc.source_op=weft_scalar.dequantize_row_q4_0 role=compute step=block_scale
    float v9 = (float)*(const _Float16 *)(v8);
    const uint8_t* v10 = v8 + 2;
    size_t v11 = v6 * 32;
    // weft_emitc.source_op=weft_scalar.dequantize_row_q4_0 role=compute step=nibble_loop
    for (size_t v12 = 0; v12 < 16; v12 += 1) {
      // weft_emitc.source_op=weft_scalar.dequantize_row_q4_0 role=compute step=nibble_decode
      const uint8_t v13 = v10[v12];
      int v14 = (int) v13;
      int v15 = v14 & 15;
      int v16 = v15 - 8;
      int v17 = v14 >> 4;
      int v18 = v17 - 8;
      // weft_emitc.source_op=weft_scalar.dequantize_row_q4_0 role=compute step=scatter_low
      size_t v19 = v11 + v12;
      float v20 = (float) v16;
      float v21 = v20 * v9;
      v2[v19] = v21;
      // weft_emitc.source_op=weft_scalar.dequantize_row_q4_0 role=compute step=scatter_high
      size_t v22 = v19 + 16;
      float v23 = (float) v18;
      float v24 = v23 * v9;
      v2[v22] = v24;
    }
  }
  return;
}


