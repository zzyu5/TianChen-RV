#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_mxfp4_kernel_dequant_mxfp4(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_dequant_mxfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 32;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v7 = v6 * 17;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 32;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_exponent_load
    const uint8_t* v12 = (const uint8_t*) v8;
    const uint8_t v13 = v12[0];
    uint32_t v14 = (uint32_t) v13;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_to_fp32_half_bits
    bool v15 = v14 < 2;
    uint32_t v16 = v14 & 0x1F;
    uint32_t v17 = 0x00200000u << v16;
    uint32_t v18 = v14 - 1;
    uint32_t v19 = v18 << 23;
    uint32_t v20 = v15 ? v17 : v19;
    uint32_t v21;
    v21 = v20;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_reinterpret_float
    const uint32_t* v22 = &v21;
    const float* v23 = (const float*) v22;
    const float v24 = v23[0];
    float v25 = (float) v24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_nibble_decode
    for (size_t v26 = 0; v26 < 16; v26 += 1) {
      size_t v27 = 1 + v26;
      const uint8_t* v28 = v8 + v27;
      const uint8_t v29 = v28[0];
      int v30 = (int) v29;
      int v31 = v30 & 15;
      int v32 = v30 >> 4;
      size_t v33 = (size_t) v31;
      const int8_t* v34 = weft_dequant_mxfp4_kvalues + v33;
      const int8_t v35 = v34[0];
      int v36 = (int) v35;
      float v37 = (float) v36;
      size_t v38 = (size_t) v32;
      const int8_t* v39 = weft_dequant_mxfp4_kvalues + v38;
      const int8_t v40 = v39[0];
      int v41 = (int) v40;
      float v42 = (float) v41;
      float v43 = v37 * v25;
      float* v44 = v11 + v26;
      v44[0] = v43;
      float v45 = v42 * v25;
      size_t v46 = v26 + 16;
      float* v47 = v11 + v46;
      v47[0] = v45;
    }
  }
  return;
}


