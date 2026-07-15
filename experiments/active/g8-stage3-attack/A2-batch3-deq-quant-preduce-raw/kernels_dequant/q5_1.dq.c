#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_q5_1_kernel_dequant_q5_1(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v5 = v1 / 32;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=x_block
    size_t v7 = v6 * 24;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 32;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=d
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v12 = (float)*(const _Float16 *)(v8);
    const uint8_t* v13 = v8 + 2;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v13);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh
    const uint8_t* v15 = v8 + 4;
    const uint8_t* v16 = (const uint8_t*) v15;
    const uint8_t v17 = v16[0];
    uint32_t v18 = (uint32_t) v17;
    const uint8_t v19 = v16[1];
    uint32_t v20 = (uint32_t) v19;
    const uint8_t v21 = v16[2];
    uint32_t v22 = (uint32_t) v21;
    const uint8_t v23 = v16[3];
    uint32_t v24 = (uint32_t) v23;
    uint32_t v25 = v20 << 8u;
    uint32_t v26 = v22 << 16u;
    uint32_t v27 = v24 << 24u;
    uint32_t v28 = v18 | v25;
    uint32_t v29 = v28 | v26;
    uint32_t v30 = v29 | v27;
    const uint8_t* v31 = v8 + 8;
    const uint8_t* v32 = (const uint8_t*) v31;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=nibble_decode
    for (size_t v33 = 0; v33 < 16; v33 += 1) {
      const uint8_t* v34 = v32 + v33;
      const uint8_t v35 = v34[0];
      int v36 = (int) v35;
      int v37 = v36 & 15;
      int v38 = v36 >> 4;
      uint32_t v39 = (uint32_t) v33;
      uint32_t v40 = v39 + 12u;
      uint32_t v41 = v30 >> v39;
      uint32_t v42 = v41 << 4u;
      uint32_t v43 = v42 & 16u;
      uint32_t v44 = v30 >> v40;
      uint32_t v45 = v44 & 16u;
      int v46 = (int) v43;
      int v47 = (int) v45;
      int v48 = v37 | v46;
      int v49 = v38 | v47;
      float v50 = (float) v48;
      float v51 = (float) v49;
      float v52 = v50 * v12;
      float v53 = v51 * v12;
      float v54 = v52 + v14;
      float v55 = v53 + v14;
      float* v56 = v11 + v33;
      float* v57 = v11 + v33;
      float* v58 = v57 + 16;
      v56[0] = v54;
      v58[0] = v55;
    }
  }
  return;
}


