#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_q2_K_kernel_dequant_q2_K(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v7 = v6 * 84;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 256;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    const uint8_t* v12 = v8 + 80;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v12);
    const uint8_t* v14 = v8 + 82;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v14);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=q2_K_decode
    for (size_t v16 = 0; v16 < 2; v16 += 1) {
      size_t v17 = v16 * 8;
      size_t v18 = v16 * 32;
      size_t v19 = 16 + v18;
      size_t v20 = v16 * 128;
      for (size_t v21 = 0; v21 < 4; v21 += 1) {
        int v22 = (int) v21;
        int v23 = v22 << 1;
        size_t v24 = v21 * 2;
        size_t v25 = v17 + v24;
        size_t v26 = v21 * 32;
        size_t v27 = v20 + v26;
        size_t v28 = v25 + 0;
        const uint8_t* v29 = v8 + v28;
        const uint8_t v30 = v29[0];
        int v31 = (int) v30;
        int v32 = v31 & 15;
        float v33 = (float) v32;
        float v34 = v13 * v33;
        int v35 = v31 >> 4;
        float v36 = (float) v35;
        float v37 = v15 * v36;
        size_t v38 = v19 + 0;
        size_t v39 = v27 + 0;
        for (size_t v40 = 0; v40 < 16; v40 += 1) {
          size_t v41 = v38 + v40;
          const uint8_t* v42 = v8 + v41;
          const uint8_t v43 = v42[0];
          int v44 = (int) v43;
          int v45 = v44 >> v23;
          int v46 = v45 & 3;
          float v47 = (float) v46;
          float v48 = v34 * v47;
          float v49 = v48 - v37;
          size_t v50 = v39 + v40;
          float* v51 = v11 + v50;
          v51[0] = v49;
        }
        size_t v52 = v25 + 1;
        const uint8_t* v53 = v8 + v52;
        const uint8_t v54 = v53[0];
        int v55 = (int) v54;
        int v56 = v55 & 15;
        float v57 = (float) v56;
        float v58 = v13 * v57;
        int v59 = v55 >> 4;
        float v60 = (float) v59;
        float v61 = v15 * v60;
        size_t v62 = v19 + 16;
        size_t v63 = v27 + 16;
        for (size_t v64 = 0; v64 < 16; v64 += 1) {
          size_t v65 = v62 + v64;
          const uint8_t* v66 = v8 + v65;
          const uint8_t v67 = v66[0];
          int v68 = (int) v67;
          int v69 = v68 >> v23;
          int v70 = v69 & 3;
          float v71 = (float) v70;
          float v72 = v58 * v71;
          float v73 = v72 - v61;
          size_t v74 = v63 + v64;
          float* v75 = v11 + v74;
          v75[0] = v73;
        }
      }
    }
  }
  return;
}


