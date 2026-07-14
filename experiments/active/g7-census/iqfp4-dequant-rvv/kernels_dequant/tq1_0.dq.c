#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_tq1_0_kernel_dequant_tq1_0(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const uint8_t weft_dequant_tq1_0_pow3[6] = {1, 3, 9, 27, 81, 243};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v7 = v6 * 54;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 256;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    const uint8_t* v12 = v8 + 52;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v12);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=tq1_0_qs_full
    const uint8_t* v14 = weft_dequant_tq1_0_pow3 + 0;
    const uint8_t v15 = v14[0];
    int v16 = (int) v15;
    for (size_t v17 = 0; v17 < 32; v17 += 1) {
      const uint8_t* v18 = v8 + v17;
      const uint8_t v19 = v18[0];
      int v20 = (int) v19;
      size_t v21 = 0 + v17;
      int v22 = v20 * v16;
      int v23 = v22 & 255;
      int v24 = v23 * 3;
      int v25 = v24 >> 8;
      int v26 = v25 - 1;
      float v27 = (float) v26;
      float v28 = v27 * v13;
      float* v29 = v11 + v21;
      v29[0] = v28;
    }
    const uint8_t* v30 = weft_dequant_tq1_0_pow3 + 1;
    const uint8_t v31 = v30[0];
    int v32 = (int) v31;
    for (size_t v33 = 0; v33 < 32; v33 += 1) {
      const uint8_t* v34 = v8 + v33;
      const uint8_t v35 = v34[0];
      int v36 = (int) v35;
      size_t v37 = 32 + v33;
      int v38 = v36 * v32;
      int v39 = v38 & 255;
      int v40 = v39 * 3;
      int v41 = v40 >> 8;
      int v42 = v41 - 1;
      float v43 = (float) v42;
      float v44 = v43 * v13;
      float* v45 = v11 + v37;
      v45[0] = v44;
    }
    const uint8_t* v46 = weft_dequant_tq1_0_pow3 + 2;
    const uint8_t v47 = v46[0];
    int v48 = (int) v47;
    for (size_t v49 = 0; v49 < 32; v49 += 1) {
      const uint8_t* v50 = v8 + v49;
      const uint8_t v51 = v50[0];
      int v52 = (int) v51;
      size_t v53 = 64 + v49;
      int v54 = v52 * v48;
      int v55 = v54 & 255;
      int v56 = v55 * 3;
      int v57 = v56 >> 8;
      int v58 = v57 - 1;
      float v59 = (float) v58;
      float v60 = v59 * v13;
      float* v61 = v11 + v53;
      v61[0] = v60;
    }
    const uint8_t* v62 = weft_dequant_tq1_0_pow3 + 3;
    const uint8_t v63 = v62[0];
    int v64 = (int) v63;
    for (size_t v65 = 0; v65 < 32; v65 += 1) {
      const uint8_t* v66 = v8 + v65;
      const uint8_t v67 = v66[0];
      int v68 = (int) v67;
      size_t v69 = 96 + v65;
      int v70 = v68 * v64;
      int v71 = v70 & 255;
      int v72 = v71 * 3;
      int v73 = v72 >> 8;
      int v74 = v73 - 1;
      float v75 = (float) v74;
      float v76 = v75 * v13;
      float* v77 = v11 + v69;
      v77[0] = v76;
    }
    const uint8_t* v78 = weft_dequant_tq1_0_pow3 + 4;
    const uint8_t v79 = v78[0];
    int v80 = (int) v79;
    for (size_t v81 = 0; v81 < 32; v81 += 1) {
      const uint8_t* v82 = v8 + v81;
      const uint8_t v83 = v82[0];
      int v84 = (int) v83;
      size_t v85 = 128 + v81;
      int v86 = v84 * v80;
      int v87 = v86 & 255;
      int v88 = v87 * 3;
      int v89 = v88 >> 8;
      int v90 = v89 - 1;
      float v91 = (float) v90;
      float v92 = v91 * v13;
      float* v93 = v11 + v85;
      v93[0] = v92;
    }
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=tq1_0_qs_tail
    const uint8_t* v94 = weft_dequant_tq1_0_pow3 + 0;
    const uint8_t v95 = v94[0];
    int v96 = (int) v95;
    for (size_t v97 = 0; v97 < 16; v97 += 1) {
      size_t v98 = 32 + v97;
      const uint8_t* v99 = v8 + v98;
      const uint8_t v100 = v99[0];
      int v101 = (int) v100;
      size_t v102 = 160 + v97;
      int v103 = v101 * v96;
      int v104 = v103 & 255;
      int v105 = v104 * 3;
      int v106 = v105 >> 8;
      int v107 = v106 - 1;
      float v108 = (float) v107;
      float v109 = v108 * v13;
      float* v110 = v11 + v102;
      v110[0] = v109;
    }
    const uint8_t* v111 = weft_dequant_tq1_0_pow3 + 1;
    const uint8_t v112 = v111[0];
    int v113 = (int) v112;
    for (size_t v114 = 0; v114 < 16; v114 += 1) {
      size_t v115 = 32 + v114;
      const uint8_t* v116 = v8 + v115;
      const uint8_t v117 = v116[0];
      int v118 = (int) v117;
      size_t v119 = 176 + v114;
      int v120 = v118 * v113;
      int v121 = v120 & 255;
      int v122 = v121 * 3;
      int v123 = v122 >> 8;
      int v124 = v123 - 1;
      float v125 = (float) v124;
      float v126 = v125 * v13;
      float* v127 = v11 + v119;
      v127[0] = v126;
    }
    const uint8_t* v128 = weft_dequant_tq1_0_pow3 + 2;
    const uint8_t v129 = v128[0];
    int v130 = (int) v129;
    for (size_t v131 = 0; v131 < 16; v131 += 1) {
      size_t v132 = 32 + v131;
      const uint8_t* v133 = v8 + v132;
      const uint8_t v134 = v133[0];
      int v135 = (int) v134;
      size_t v136 = 192 + v131;
      int v137 = v135 * v130;
      int v138 = v137 & 255;
      int v139 = v138 * 3;
      int v140 = v139 >> 8;
      int v141 = v140 - 1;
      float v142 = (float) v141;
      float v143 = v142 * v13;
      float* v144 = v11 + v136;
      v144[0] = v143;
    }
    const uint8_t* v145 = weft_dequant_tq1_0_pow3 + 3;
    const uint8_t v146 = v145[0];
    int v147 = (int) v146;
    for (size_t v148 = 0; v148 < 16; v148 += 1) {
      size_t v149 = 32 + v148;
      const uint8_t* v150 = v8 + v149;
      const uint8_t v151 = v150[0];
      int v152 = (int) v151;
      size_t v153 = 208 + v148;
      int v154 = v152 * v147;
      int v155 = v154 & 255;
      int v156 = v155 * 3;
      int v157 = v156 >> 8;
      int v158 = v157 - 1;
      float v159 = (float) v158;
      float v160 = v159 * v13;
      float* v161 = v11 + v153;
      v161[0] = v160;
    }
    const uint8_t* v162 = weft_dequant_tq1_0_pow3 + 4;
    const uint8_t v163 = v162[0];
    int v164 = (int) v163;
    for (size_t v165 = 0; v165 < 16; v165 += 1) {
      size_t v166 = 32 + v165;
      const uint8_t* v167 = v8 + v166;
      const uint8_t v168 = v167[0];
      int v169 = (int) v168;
      size_t v170 = 224 + v165;
      int v171 = v169 * v164;
      int v172 = v171 & 255;
      int v173 = v172 * 3;
      int v174 = v173 >> 8;
      int v175 = v174 - 1;
      float v176 = (float) v175;
      float v177 = v176 * v13;
      float* v178 = v11 + v170;
      v178[0] = v177;
    }
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=tq1_0_qh
    const uint8_t* v179 = weft_dequant_tq1_0_pow3 + 0;
    const uint8_t v180 = v179[0];
    int v181 = (int) v180;
    for (size_t v182 = 0; v182 < 4; v182 += 1) {
      size_t v183 = 48 + v182;
      const uint8_t* v184 = v8 + v183;
      const uint8_t v185 = v184[0];
      int v186 = (int) v185;
      size_t v187 = 240 + v182;
      int v188 = v186 * v181;
      int v189 = v188 & 255;
      int v190 = v189 * 3;
      int v191 = v190 >> 8;
      int v192 = v191 - 1;
      float v193 = (float) v192;
      float v194 = v193 * v13;
      float* v195 = v11 + v187;
      v195[0] = v194;
    }
    const uint8_t* v196 = weft_dequant_tq1_0_pow3 + 1;
    const uint8_t v197 = v196[0];
    int v198 = (int) v197;
    for (size_t v199 = 0; v199 < 4; v199 += 1) {
      size_t v200 = 48 + v199;
      const uint8_t* v201 = v8 + v200;
      const uint8_t v202 = v201[0];
      int v203 = (int) v202;
      size_t v204 = 244 + v199;
      int v205 = v203 * v198;
      int v206 = v205 & 255;
      int v207 = v206 * 3;
      int v208 = v207 >> 8;
      int v209 = v208 - 1;
      float v210 = (float) v209;
      float v211 = v210 * v13;
      float* v212 = v11 + v204;
      v212[0] = v211;
    }
    const uint8_t* v213 = weft_dequant_tq1_0_pow3 + 2;
    const uint8_t v214 = v213[0];
    int v215 = (int) v214;
    for (size_t v216 = 0; v216 < 4; v216 += 1) {
      size_t v217 = 48 + v216;
      const uint8_t* v218 = v8 + v217;
      const uint8_t v219 = v218[0];
      int v220 = (int) v219;
      size_t v221 = 248 + v216;
      int v222 = v220 * v215;
      int v223 = v222 & 255;
      int v224 = v223 * 3;
      int v225 = v224 >> 8;
      int v226 = v225 - 1;
      float v227 = (float) v226;
      float v228 = v227 * v13;
      float* v229 = v11 + v221;
      v229[0] = v228;
    }
    const uint8_t* v230 = weft_dequant_tq1_0_pow3 + 3;
    const uint8_t v231 = v230[0];
    int v232 = (int) v231;
    for (size_t v233 = 0; v233 < 4; v233 += 1) {
      size_t v234 = 48 + v233;
      const uint8_t* v235 = v8 + v234;
      const uint8_t v236 = v235[0];
      int v237 = (int) v236;
      size_t v238 = 252 + v233;
      int v239 = v237 * v232;
      int v240 = v239 & 255;
      int v241 = v240 * 3;
      int v242 = v241 >> 8;
      int v243 = v242 - 1;
      float v244 = (float) v243;
      float v245 = v244 * v13;
      float* v246 = v11 + v238;
      v246[0] = v245;
    }
  }
  return;
}


