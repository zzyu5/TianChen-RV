#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_nvfp4_kernel_dequant_nvfp4(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_dequant_nvfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 64;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v7 = v6 * 36;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 64;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=nvfp4_sub_decode
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_scale
    const uint8_t* v12 = v8 + 0;
    const uint8_t v13 = v12[0];
    uint32_t v14 = (uint32_t) v13;
    uint32_t v15 = v14 >> 3;
    uint32_t v16 = v15 & 0xF;
    uint32_t v17 = v14 & 0x7;
    int v18 = (int) v16;
    int v19 = (int) v17;
    float v20 = (float) v19;
    float v21 = ldexpf(v20, -9);
    float v22 = v20 / 8.0f;
    float v23 = 1.0f + v22;
    int v24 = v18 - 7;
    float v25 = ldexpf(v23, v24);
    bool v26 = v16 == 0;
    float v27 = v26 ? v21 : v25;
    float v28 = v27 * 0.5f;
    bool v29 = v14 == 0;
    bool v30 = v14 == 0x7F;
    bool v31 = v29 || v30;
    float v32 = v31 ? 0.0f : v28;
    for (size_t v33 = 0; v33 < 8; v33 += 1) {
      size_t v34 = 4 + v33;
      const uint8_t* v35 = v8 + v34;
      const uint8_t v36 = v35[0];
      int v37 = (int) v36;
      int v38 = v37 & 15;
      size_t v39 = (size_t) v38;
      const int8_t* v40 = weft_dequant_nvfp4_kvalues + v39;
      const int8_t v41 = v40[0];
      int v42 = (int) v41;
      float v43 = (float) v42;
      int v44 = v37 >> 4;
      size_t v45 = (size_t) v44;
      const int8_t* v46 = weft_dequant_nvfp4_kvalues + v45;
      const int8_t v47 = v46[0];
      int v48 = (int) v47;
      float v49 = (float) v48;
      float v50 = v43 * v32;
      size_t v51 = 0 + v33;
      float* v52 = v11 + v51;
      v52[0] = v50;
      float v53 = v49 * v32;
      size_t v54 = 8 + v33;
      float* v55 = v11 + v54;
      v55[0] = v53;
    }
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_scale
    const uint8_t* v56 = v8 + 1;
    const uint8_t v57 = v56[0];
    uint32_t v58 = (uint32_t) v57;
    uint32_t v59 = v58 >> 3;
    uint32_t v60 = v59 & 0xF;
    uint32_t v61 = v58 & 0x7;
    int v62 = (int) v60;
    int v63 = (int) v61;
    float v64 = (float) v63;
    float v65 = ldexpf(v64, -9);
    float v66 = v64 / 8.0f;
    float v67 = 1.0f + v66;
    int v68 = v62 - 7;
    float v69 = ldexpf(v67, v68);
    bool v70 = v60 == 0;
    float v71 = v70 ? v65 : v69;
    float v72 = v71 * 0.5f;
    bool v73 = v58 == 0;
    bool v74 = v58 == 0x7F;
    bool v75 = v73 || v74;
    float v76 = v75 ? 0.0f : v72;
    for (size_t v77 = 0; v77 < 8; v77 += 1) {
      size_t v78 = 12 + v77;
      const uint8_t* v79 = v8 + v78;
      const uint8_t v80 = v79[0];
      int v81 = (int) v80;
      int v82 = v81 & 15;
      size_t v83 = (size_t) v82;
      const int8_t* v84 = weft_dequant_nvfp4_kvalues + v83;
      const int8_t v85 = v84[0];
      int v86 = (int) v85;
      float v87 = (float) v86;
      int v88 = v81 >> 4;
      size_t v89 = (size_t) v88;
      const int8_t* v90 = weft_dequant_nvfp4_kvalues + v89;
      const int8_t v91 = v90[0];
      int v92 = (int) v91;
      float v93 = (float) v92;
      float v94 = v87 * v76;
      size_t v95 = 16 + v77;
      float* v96 = v11 + v95;
      v96[0] = v94;
      float v97 = v93 * v76;
      size_t v98 = 24 + v77;
      float* v99 = v11 + v98;
      v99[0] = v97;
    }
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_scale
    const uint8_t* v100 = v8 + 2;
    const uint8_t v101 = v100[0];
    uint32_t v102 = (uint32_t) v101;
    uint32_t v103 = v102 >> 3;
    uint32_t v104 = v103 & 0xF;
    uint32_t v105 = v102 & 0x7;
    int v106 = (int) v104;
    int v107 = (int) v105;
    float v108 = (float) v107;
    float v109 = ldexpf(v108, -9);
    float v110 = v108 / 8.0f;
    float v111 = 1.0f + v110;
    int v112 = v106 - 7;
    float v113 = ldexpf(v111, v112);
    bool v114 = v104 == 0;
    float v115 = v114 ? v109 : v113;
    float v116 = v115 * 0.5f;
    bool v117 = v102 == 0;
    bool v118 = v102 == 0x7F;
    bool v119 = v117 || v118;
    float v120 = v119 ? 0.0f : v116;
    for (size_t v121 = 0; v121 < 8; v121 += 1) {
      size_t v122 = 20 + v121;
      const uint8_t* v123 = v8 + v122;
      const uint8_t v124 = v123[0];
      int v125 = (int) v124;
      int v126 = v125 & 15;
      size_t v127 = (size_t) v126;
      const int8_t* v128 = weft_dequant_nvfp4_kvalues + v127;
      const int8_t v129 = v128[0];
      int v130 = (int) v129;
      float v131 = (float) v130;
      int v132 = v125 >> 4;
      size_t v133 = (size_t) v132;
      const int8_t* v134 = weft_dequant_nvfp4_kvalues + v133;
      const int8_t v135 = v134[0];
      int v136 = (int) v135;
      float v137 = (float) v136;
      float v138 = v131 * v120;
      size_t v139 = 32 + v121;
      float* v140 = v11 + v139;
      v140[0] = v138;
      float v141 = v137 * v120;
      size_t v142 = 40 + v121;
      float* v143 = v11 + v142;
      v143[0] = v141;
    }
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_scale
    const uint8_t* v144 = v8 + 3;
    const uint8_t v145 = v144[0];
    uint32_t v146 = (uint32_t) v145;
    uint32_t v147 = v146 >> 3;
    uint32_t v148 = v147 & 0xF;
    uint32_t v149 = v146 & 0x7;
    int v150 = (int) v148;
    int v151 = (int) v149;
    float v152 = (float) v151;
    float v153 = ldexpf(v152, -9);
    float v154 = v152 / 8.0f;
    float v155 = 1.0f + v154;
    int v156 = v150 - 7;
    float v157 = ldexpf(v155, v156);
    bool v158 = v148 == 0;
    float v159 = v158 ? v153 : v157;
    float v160 = v159 * 0.5f;
    bool v161 = v146 == 0;
    bool v162 = v146 == 0x7F;
    bool v163 = v161 || v162;
    float v164 = v163 ? 0.0f : v160;
    for (size_t v165 = 0; v165 < 8; v165 += 1) {
      size_t v166 = 28 + v165;
      const uint8_t* v167 = v8 + v166;
      const uint8_t v168 = v167[0];
      int v169 = (int) v168;
      int v170 = v169 & 15;
      size_t v171 = (size_t) v170;
      const int8_t* v172 = weft_dequant_nvfp4_kvalues + v171;
      const int8_t v173 = v172[0];
      int v174 = (int) v173;
      float v175 = (float) v174;
      int v176 = v169 >> 4;
      size_t v177 = (size_t) v176;
      const int8_t* v178 = weft_dequant_nvfp4_kvalues + v177;
      const int8_t v179 = v178[0];
      int v180 = (int) v179;
      float v181 = (float) v180;
      float v182 = v175 * v164;
      size_t v183 = 48 + v165;
      float* v184 = v11 + v183;
      v184[0] = v182;
      float v185 = v181 * v164;
      size_t v186 = 56 + v165;
      float* v187 = v11 + v186;
      v187[0] = v185;
    }
  }
  return;
}


