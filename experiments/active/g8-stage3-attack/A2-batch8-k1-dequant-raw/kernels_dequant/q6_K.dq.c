#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_q6_K_kernel_dequant_q6_K(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v7 = v6 * 210;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 256;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    const uint8_t* v12 = v8 + 208;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v12);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=q6_K_decode
    const int8_t* v14 = (const int8_t*) v8;
    for (size_t v15 = 0; v15 < 32; v15 += 1) {
      size_t v16 = v15 / 16;
      size_t v17 = 0 + v15;
      const uint8_t* v18 = v8 + v17;
      const uint8_t v19 = v18[0];
      int v20 = (int) v19;
      size_t v21 = 32 + v15;
      const uint8_t* v22 = v8 + v21;
      const uint8_t v23 = v22[0];
      int v24 = (int) v23;
      size_t v25 = 128 + v15;
      const uint8_t* v26 = v8 + v25;
      const uint8_t v27 = v26[0];
      int v28 = (int) v27;
      int v29 = v28 >> 0;
      int v30 = v29 & 3;
      int v31 = v30 << 4;
      int v32 = v20 & 15;
      int v33 = v32 | v31;
      int v34 = v33 - 32;
      int v35 = v28 >> 2;
      int v36 = v35 & 3;
      int v37 = v36 << 4;
      int v38 = v24 & 15;
      int v39 = v38 | v37;
      int v40 = v39 - 32;
      int v41 = v28 >> 4;
      int v42 = v41 & 3;
      int v43 = v42 << 4;
      int v44 = v20 >> 4;
      int v45 = v44 | v43;
      int v46 = v45 - 32;
      int v47 = v28 >> 6;
      int v48 = v47 & 3;
      int v49 = v48 << 4;
      int v50 = v24 >> 4;
      int v51 = v50 | v49;
      int v52 = v51 - 32;
      size_t v53 = 0 + v15;
      float v54 = (float) v34;
      size_t v55 = 192 + v16;
      const int8_t* v56 = v14 + v55;
      const int8_t v57 = v56[0];
      int v58 = (int) v57;
      float v59 = (float) v58;
      float v60 = v13 * v59;
      float v61 = v60 * v54;
      float* v62 = v11 + v53;
      v62[0] = v61;
      float v63 = (float) v40;
      size_t v64 = 194 + v16;
      const int8_t* v65 = v14 + v64;
      const int8_t v66 = v65[0];
      int v67 = (int) v66;
      float v68 = (float) v67;
      float v69 = v13 * v68;
      float v70 = v69 * v63;
      size_t v71 = v53 + 32;
      float* v72 = v11 + v71;
      v72[0] = v70;
      float v73 = (float) v46;
      size_t v74 = 196 + v16;
      const int8_t* v75 = v14 + v74;
      const int8_t v76 = v75[0];
      int v77 = (int) v76;
      float v78 = (float) v77;
      float v79 = v13 * v78;
      float v80 = v79 * v73;
      size_t v81 = v53 + 64;
      float* v82 = v11 + v81;
      v82[0] = v80;
      float v83 = (float) v52;
      size_t v84 = 198 + v16;
      const int8_t* v85 = v14 + v84;
      const int8_t v86 = v85[0];
      int v87 = (int) v86;
      float v88 = (float) v87;
      float v89 = v13 * v88;
      float v90 = v89 * v83;
      size_t v91 = v53 + 96;
      float* v92 = v11 + v91;
      v92[0] = v90;
    }
    for (size_t v93 = 0; v93 < 32; v93 += 1) {
      size_t v94 = v93 / 16;
      size_t v95 = 64 + v93;
      const uint8_t* v96 = v8 + v95;
      const uint8_t v97 = v96[0];
      int v98 = (int) v97;
      size_t v99 = 96 + v93;
      const uint8_t* v100 = v8 + v99;
      const uint8_t v101 = v100[0];
      int v102 = (int) v101;
      size_t v103 = 160 + v93;
      const uint8_t* v104 = v8 + v103;
      const uint8_t v105 = v104[0];
      int v106 = (int) v105;
      int v107 = v106 >> 0;
      int v108 = v107 & 3;
      int v109 = v108 << 4;
      int v110 = v98 & 15;
      int v111 = v110 | v109;
      int v112 = v111 - 32;
      int v113 = v106 >> 2;
      int v114 = v113 & 3;
      int v115 = v114 << 4;
      int v116 = v102 & 15;
      int v117 = v116 | v115;
      int v118 = v117 - 32;
      int v119 = v106 >> 4;
      int v120 = v119 & 3;
      int v121 = v120 << 4;
      int v122 = v98 >> 4;
      int v123 = v122 | v121;
      int v124 = v123 - 32;
      int v125 = v106 >> 6;
      int v126 = v125 & 3;
      int v127 = v126 << 4;
      int v128 = v102 >> 4;
      int v129 = v128 | v127;
      int v130 = v129 - 32;
      size_t v131 = 128 + v93;
      float v132 = (float) v112;
      size_t v133 = 200 + v94;
      const int8_t* v134 = v14 + v133;
      const int8_t v135 = v134[0];
      int v136 = (int) v135;
      float v137 = (float) v136;
      float v138 = v13 * v137;
      float v139 = v138 * v132;
      float* v140 = v11 + v131;
      v140[0] = v139;
      float v141 = (float) v118;
      size_t v142 = 202 + v94;
      const int8_t* v143 = v14 + v142;
      const int8_t v144 = v143[0];
      int v145 = (int) v144;
      float v146 = (float) v145;
      float v147 = v13 * v146;
      float v148 = v147 * v141;
      size_t v149 = v131 + 32;
      float* v150 = v11 + v149;
      v150[0] = v148;
      float v151 = (float) v124;
      size_t v152 = 204 + v94;
      const int8_t* v153 = v14 + v152;
      const int8_t v154 = v153[0];
      int v155 = (int) v154;
      float v156 = (float) v155;
      float v157 = v13 * v156;
      float v158 = v157 * v151;
      size_t v159 = v131 + 64;
      float* v160 = v11 + v159;
      v160[0] = v158;
      float v161 = (float) v130;
      size_t v162 = 206 + v94;
      const int8_t* v163 = v14 + v162;
      const int8_t v164 = v163[0];
      int v165 = (int) v164;
      float v166 = (float) v165;
      float v167 = v13 * v166;
      float v168 = v167 * v161;
      size_t v169 = v131 + 96;
      float* v170 = v11 + v169;
      v170[0] = v168;
    }
  }
  return;
}


