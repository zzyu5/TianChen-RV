#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_tq2_0_kernel_dequant_tq2_0(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v7 = v6 * 66;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 256;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    const uint8_t* v12 = v8 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v12);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=tq2_0_decode
    for (size_t v14 = 0; v14 < 32; v14 += 1) {
      size_t v15 = 0 + v14;
      const uint8_t* v16 = v8 + v15;
      const uint8_t v17 = v16[0];
      int v18 = (int) v17;
      int v19 = v18 >> 0;
      int v20 = v19 & 3;
      int v21 = v20 - 1;
      float v22 = (float) v21;
      float v23 = v22 * v13;
      size_t v24 = 0 + v14;
      float* v25 = v11 + v24;
      v25[0] = v23;
    }
    for (size_t v26 = 0; v26 < 32; v26 += 1) {
      size_t v27 = 0 + v26;
      const uint8_t* v28 = v8 + v27;
      const uint8_t v29 = v28[0];
      int v30 = (int) v29;
      int v31 = v30 >> 2;
      int v32 = v31 & 3;
      int v33 = v32 - 1;
      float v34 = (float) v33;
      float v35 = v34 * v13;
      size_t v36 = 32 + v26;
      float* v37 = v11 + v36;
      v37[0] = v35;
    }
    for (size_t v38 = 0; v38 < 32; v38 += 1) {
      size_t v39 = 0 + v38;
      const uint8_t* v40 = v8 + v39;
      const uint8_t v41 = v40[0];
      int v42 = (int) v41;
      int v43 = v42 >> 4;
      int v44 = v43 & 3;
      int v45 = v44 - 1;
      float v46 = (float) v45;
      float v47 = v46 * v13;
      size_t v48 = 64 + v38;
      float* v49 = v11 + v48;
      v49[0] = v47;
    }
    for (size_t v50 = 0; v50 < 32; v50 += 1) {
      size_t v51 = 0 + v50;
      const uint8_t* v52 = v8 + v51;
      const uint8_t v53 = v52[0];
      int v54 = (int) v53;
      int v55 = v54 >> 6;
      int v56 = v55 & 3;
      int v57 = v56 - 1;
      float v58 = (float) v57;
      float v59 = v58 * v13;
      size_t v60 = 96 + v50;
      float* v61 = v11 + v60;
      v61[0] = v59;
    }
    for (size_t v62 = 0; v62 < 32; v62 += 1) {
      size_t v63 = 32 + v62;
      const uint8_t* v64 = v8 + v63;
      const uint8_t v65 = v64[0];
      int v66 = (int) v65;
      int v67 = v66 >> 0;
      int v68 = v67 & 3;
      int v69 = v68 - 1;
      float v70 = (float) v69;
      float v71 = v70 * v13;
      size_t v72 = 128 + v62;
      float* v73 = v11 + v72;
      v73[0] = v71;
    }
    for (size_t v74 = 0; v74 < 32; v74 += 1) {
      size_t v75 = 32 + v74;
      const uint8_t* v76 = v8 + v75;
      const uint8_t v77 = v76[0];
      int v78 = (int) v77;
      int v79 = v78 >> 2;
      int v80 = v79 & 3;
      int v81 = v80 - 1;
      float v82 = (float) v81;
      float v83 = v82 * v13;
      size_t v84 = 160 + v74;
      float* v85 = v11 + v84;
      v85[0] = v83;
    }
    for (size_t v86 = 0; v86 < 32; v86 += 1) {
      size_t v87 = 32 + v86;
      const uint8_t* v88 = v8 + v87;
      const uint8_t v89 = v88[0];
      int v90 = (int) v89;
      int v91 = v90 >> 4;
      int v92 = v91 & 3;
      int v93 = v92 - 1;
      float v94 = (float) v93;
      float v95 = v94 * v13;
      size_t v96 = 192 + v86;
      float* v97 = v11 + v96;
      v97[0] = v95;
    }
    for (size_t v98 = 0; v98 < 32; v98 += 1) {
      size_t v99 = 32 + v98;
      const uint8_t* v100 = v8 + v99;
      const uint8_t v101 = v100[0];
      int v102 = (int) v101;
      int v103 = v102 >> 6;
      int v104 = v103 & 3;
      int v105 = v104 - 1;
      float v106 = (float) v105;
      float v107 = v106 * v13;
      size_t v108 = 224 + v98;
      float* v109 = v11 + v108;
      v109[0] = v107;
    }
  }
  return;
}


