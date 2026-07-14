#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_q4_K_kernel_dequant_q4_K(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v7 = v6 * 144;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 256;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v12 = (float)*(const _Float16 *)(v8);
    const uint8_t* v13 = v8 + 2;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v13);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=q4_K_decode
    const uint8_t* v15 = v8 + 4;
    const uint8_t v16 = v15[0];
    int v17 = (int) v16;
    int v18 = v17 & 63;
    const uint8_t* v19 = v8 + 8;
    const uint8_t v20 = v19[0];
    int v21 = (int) v20;
    int v22 = v21 & 63;
    float v23 = (float) v18;
    float v24 = v12 * v23;
    float v25 = (float) v22;
    float v26 = v14 * v25;
    const uint8_t* v27 = v8 + 5;
    const uint8_t v28 = v27[0];
    int v29 = (int) v28;
    int v30 = v29 & 63;
    const uint8_t* v31 = v8 + 9;
    const uint8_t v32 = v31[0];
    int v33 = (int) v32;
    int v34 = v33 & 63;
    float v35 = (float) v30;
    float v36 = v12 * v35;
    float v37 = (float) v34;
    float v38 = v14 * v37;
    for (size_t v39 = 0; v39 < 32; v39 += 1) {
      size_t v40 = 16 + v39;
      const uint8_t* v41 = v8 + v40;
      const uint8_t v42 = v41[0];
      int v43 = (int) v42;
      int v44 = v43 & 15;
      float v45 = (float) v44;
      float v46 = v24 * v45;
      float v47 = v46 - v26;
      size_t v48 = 0 + v39;
      float* v49 = v11 + v48;
      v49[0] = v47;
    }
    for (size_t v50 = 0; v50 < 32; v50 += 1) {
      size_t v51 = 16 + v50;
      const uint8_t* v52 = v8 + v51;
      const uint8_t v53 = v52[0];
      int v54 = (int) v53;
      int v55 = v54 >> 4;
      float v56 = (float) v55;
      float v57 = v36 * v56;
      float v58 = v57 - v38;
      size_t v59 = 32 + v50;
      float* v60 = v11 + v59;
      v60[0] = v58;
    }
    const uint8_t* v61 = v8 + 6;
    const uint8_t v62 = v61[0];
    int v63 = (int) v62;
    int v64 = v63 & 63;
    const uint8_t* v65 = v8 + 10;
    const uint8_t v66 = v65[0];
    int v67 = (int) v66;
    int v68 = v67 & 63;
    float v69 = (float) v64;
    float v70 = v12 * v69;
    float v71 = (float) v68;
    float v72 = v14 * v71;
    const uint8_t* v73 = v8 + 7;
    const uint8_t v74 = v73[0];
    int v75 = (int) v74;
    int v76 = v75 & 63;
    const uint8_t* v77 = v8 + 11;
    const uint8_t v78 = v77[0];
    int v79 = (int) v78;
    int v80 = v79 & 63;
    float v81 = (float) v76;
    float v82 = v12 * v81;
    float v83 = (float) v80;
    float v84 = v14 * v83;
    for (size_t v85 = 0; v85 < 32; v85 += 1) {
      size_t v86 = 48 + v85;
      const uint8_t* v87 = v8 + v86;
      const uint8_t v88 = v87[0];
      int v89 = (int) v88;
      int v90 = v89 & 15;
      float v91 = (float) v90;
      float v92 = v70 * v91;
      float v93 = v92 - v72;
      size_t v94 = 64 + v85;
      float* v95 = v11 + v94;
      v95[0] = v93;
    }
    for (size_t v96 = 0; v96 < 32; v96 += 1) {
      size_t v97 = 48 + v96;
      const uint8_t* v98 = v8 + v97;
      const uint8_t v99 = v98[0];
      int v100 = (int) v99;
      int v101 = v100 >> 4;
      float v102 = (float) v101;
      float v103 = v82 * v102;
      float v104 = v103 - v84;
      size_t v105 = 96 + v96;
      float* v106 = v11 + v105;
      v106[0] = v104;
    }
    const uint8_t* v107 = v8 + 4;
    const uint8_t v108 = v107[0];
    int v109 = (int) v108;
    int v110 = v109 >> 6;
    int v111 = v110 << 4;
    const uint8_t* v112 = v8 + 12;
    const uint8_t v113 = v112[0];
    int v114 = (int) v113;
    int v115 = v114 & 15;
    int v116 = v115 | v111;
    const uint8_t* v117 = v8 + 8;
    const uint8_t v118 = v117[0];
    int v119 = (int) v118;
    int v120 = v119 >> 6;
    int v121 = v120 << 4;
    const uint8_t* v122 = v8 + 12;
    const uint8_t v123 = v122[0];
    int v124 = (int) v123;
    int v125 = v124 >> 4;
    int v126 = v125 | v121;
    float v127 = (float) v116;
    float v128 = v12 * v127;
    float v129 = (float) v126;
    float v130 = v14 * v129;
    const uint8_t* v131 = v8 + 5;
    const uint8_t v132 = v131[0];
    int v133 = (int) v132;
    int v134 = v133 >> 6;
    int v135 = v134 << 4;
    const uint8_t* v136 = v8 + 13;
    const uint8_t v137 = v136[0];
    int v138 = (int) v137;
    int v139 = v138 & 15;
    int v140 = v139 | v135;
    const uint8_t* v141 = v8 + 9;
    const uint8_t v142 = v141[0];
    int v143 = (int) v142;
    int v144 = v143 >> 6;
    int v145 = v144 << 4;
    const uint8_t* v146 = v8 + 13;
    const uint8_t v147 = v146[0];
    int v148 = (int) v147;
    int v149 = v148 >> 4;
    int v150 = v149 | v145;
    float v151 = (float) v140;
    float v152 = v12 * v151;
    float v153 = (float) v150;
    float v154 = v14 * v153;
    for (size_t v155 = 0; v155 < 32; v155 += 1) {
      size_t v156 = 80 + v155;
      const uint8_t* v157 = v8 + v156;
      const uint8_t v158 = v157[0];
      int v159 = (int) v158;
      int v160 = v159 & 15;
      float v161 = (float) v160;
      float v162 = v128 * v161;
      float v163 = v162 - v130;
      size_t v164 = 128 + v155;
      float* v165 = v11 + v164;
      v165[0] = v163;
    }
    for (size_t v166 = 0; v166 < 32; v166 += 1) {
      size_t v167 = 80 + v166;
      const uint8_t* v168 = v8 + v167;
      const uint8_t v169 = v168[0];
      int v170 = (int) v169;
      int v171 = v170 >> 4;
      float v172 = (float) v171;
      float v173 = v152 * v172;
      float v174 = v173 - v154;
      size_t v175 = 160 + v166;
      float* v176 = v11 + v175;
      v176[0] = v174;
    }
    const uint8_t* v177 = v8 + 6;
    const uint8_t v178 = v177[0];
    int v179 = (int) v178;
    int v180 = v179 >> 6;
    int v181 = v180 << 4;
    const uint8_t* v182 = v8 + 14;
    const uint8_t v183 = v182[0];
    int v184 = (int) v183;
    int v185 = v184 & 15;
    int v186 = v185 | v181;
    const uint8_t* v187 = v8 + 10;
    const uint8_t v188 = v187[0];
    int v189 = (int) v188;
    int v190 = v189 >> 6;
    int v191 = v190 << 4;
    const uint8_t* v192 = v8 + 14;
    const uint8_t v193 = v192[0];
    int v194 = (int) v193;
    int v195 = v194 >> 4;
    int v196 = v195 | v191;
    float v197 = (float) v186;
    float v198 = v12 * v197;
    float v199 = (float) v196;
    float v200 = v14 * v199;
    const uint8_t* v201 = v8 + 7;
    const uint8_t v202 = v201[0];
    int v203 = (int) v202;
    int v204 = v203 >> 6;
    int v205 = v204 << 4;
    const uint8_t* v206 = v8 + 15;
    const uint8_t v207 = v206[0];
    int v208 = (int) v207;
    int v209 = v208 & 15;
    int v210 = v209 | v205;
    const uint8_t* v211 = v8 + 11;
    const uint8_t v212 = v211[0];
    int v213 = (int) v212;
    int v214 = v213 >> 6;
    int v215 = v214 << 4;
    const uint8_t* v216 = v8 + 15;
    const uint8_t v217 = v216[0];
    int v218 = (int) v217;
    int v219 = v218 >> 4;
    int v220 = v219 | v215;
    float v221 = (float) v210;
    float v222 = v12 * v221;
    float v223 = (float) v220;
    float v224 = v14 * v223;
    for (size_t v225 = 0; v225 < 32; v225 += 1) {
      size_t v226 = 112 + v225;
      const uint8_t* v227 = v8 + v226;
      const uint8_t v228 = v227[0];
      int v229 = (int) v228;
      int v230 = v229 & 15;
      float v231 = (float) v230;
      float v232 = v198 * v231;
      float v233 = v232 - v200;
      size_t v234 = 192 + v225;
      float* v235 = v11 + v234;
      v235[0] = v233;
    }
    for (size_t v236 = 0; v236 < 32; v236 += 1) {
      size_t v237 = 112 + v236;
      const uint8_t* v238 = v8 + v237;
      const uint8_t v239 = v238[0];
      int v240 = (int) v239;
      int v241 = v240 >> 4;
      float v242 = (float) v241;
      float v243 = v222 * v242;
      float v244 = v243 - v224;
      size_t v245 = 224 + v236;
      float* v246 = v11 + v245;
      v246[0] = v244;
    }
  }
  return;
}


