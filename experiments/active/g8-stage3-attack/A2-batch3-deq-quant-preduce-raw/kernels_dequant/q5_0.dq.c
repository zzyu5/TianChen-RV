#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_q5_0_kernel_dequant_q5_0(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v5 = v1 / 32;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=x_block
    size_t v7 = v6 * 22;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 32;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=d
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v12 = (float)*(const _Float16 *)(v8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh
    const uint8_t* v13 = v8 + 2;
    const uint8_t* v14 = (const uint8_t*) v13;
    const uint8_t v15 = v14[0];
    uint32_t v16 = (uint32_t) v15;
    const uint8_t v17 = v14[1];
    uint32_t v18 = (uint32_t) v17;
    const uint8_t v19 = v14[2];
    uint32_t v20 = (uint32_t) v19;
    const uint8_t v21 = v14[3];
    uint32_t v22 = (uint32_t) v21;
    uint32_t v23 = v18 << 8u;
    uint32_t v24 = v20 << 16u;
    uint32_t v25 = v22 << 24u;
    uint32_t v26 = v16 | v23;
    uint32_t v27 = v26 | v24;
    uint32_t v28 = v27 | v25;
    const uint8_t* v29 = v8 + 6;
    const uint8_t* v30 = (const uint8_t*) v29;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=nibble_decode
    for (size_t v31 = 0; v31 < 16; v31 += 1) {
      const uint8_t* v32 = v30 + v31;
      const uint8_t v33 = v32[0];
      int v34 = (int) v33;
      int v35 = v34 & 15;
      int v36 = v34 >> 4;
      uint32_t v37 = (uint32_t) v31;
      uint32_t v38 = v37 + 12u;
      uint32_t v39 = v28 >> v37;
      uint32_t v40 = v39 << 4u;
      uint32_t v41 = v40 & 16u;
      uint32_t v42 = v28 >> v38;
      uint32_t v43 = v42 & 16u;
      int v44 = (int) v41;
      int v45 = (int) v43;
      int v46 = v35 | v44;
      int v47 = v36 | v45;
      int v48 = v46 - 16;
      int v49 = v47 - 16;
      float v50 = (float) v48;
      float v51 = (float) v49;
      float v52 = v50 * v12;
      float v53 = v51 * v12;
      float* v54 = v11 + v31;
      float* v55 = v11 + v31;
      float* v56 = v55 + 16;
      v54[0] = v52;
      v56[0] = v53;
    }
  }
  return;
}


