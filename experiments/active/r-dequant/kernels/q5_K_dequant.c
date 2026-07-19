#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_q5_K_kernel_dequant_q5_K(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v7 = v6 * 176;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 256;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=q5_K_decode
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v12 = (float)*(const _Float16 *)(v8);
    const uint8_t* v13 = v8 + 2;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v13);
    const uint8_t* v15 = v8 + 16;
    const uint8_t* v16 = (const uint8_t*) v15;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v17 = __riscv_vle8_v_u8m2(v16, 32);
    const uint8_t* v18 = v8 + 4;
    const uint8_t* v19 = (const uint8_t*) v18;
    const uint8_t v20 = v19[0];
    int v21 = (int) v20;
    int v22 = v21 & 63;
    const uint8_t* v23 = v8 + 8;
    const uint8_t* v24 = (const uint8_t*) v23;
    const uint8_t v25 = v24[0];
    int v26 = (int) v25;
    int v27 = v26 & 63;
    const uint8_t* v28 = v8 + 5;
    const uint8_t* v29 = (const uint8_t*) v28;
    const uint8_t v30 = v29[0];
    int v31 = (int) v30;
    int v32 = v31 & 63;
    const uint8_t* v33 = v8 + 9;
    const uint8_t* v34 = (const uint8_t*) v33;
    const uint8_t v35 = v34[0];
    int v36 = (int) v35;
    int v37 = v36 & 63;
    float v38 = (float) v22;
    float v39 = v12 * v38;
    float v40 = (float) v27;
    float v41 = v14 * v40;
    float v42 = (float) v32;
    float v43 = v12 * v42;
    float v44 = (float) v37;
    float v45 = v14 * v44;
    const uint8_t* v46 = v8 + 6;
    const uint8_t* v47 = (const uint8_t*) v46;
    const uint8_t v48 = v47[0];
    int v49 = (int) v48;
    int v50 = v49 & 63;
    const uint8_t* v51 = v8 + 10;
    const uint8_t* v52 = (const uint8_t*) v51;
    const uint8_t v53 = v52[0];
    int v54 = (int) v53;
    int v55 = v54 & 63;
    const uint8_t* v56 = v8 + 7;
    const uint8_t* v57 = (const uint8_t*) v56;
    const uint8_t v58 = v57[0];
    int v59 = (int) v58;
    int v60 = v59 & 63;
    const uint8_t* v61 = v8 + 11;
    const uint8_t* v62 = (const uint8_t*) v61;
    const uint8_t v63 = v62[0];
    int v64 = (int) v63;
    int v65 = v64 & 63;
    float v66 = (float) v50;
    float v67 = v12 * v66;
    float v68 = (float) v55;
    float v69 = v14 * v68;
    float v70 = (float) v60;
    float v71 = v12 * v70;
    float v72 = (float) v65;
    float v73 = v14 * v72;
    const uint8_t* v74 = v8 + 4;
    const uint8_t* v75 = (const uint8_t*) v74;
    const uint8_t v76 = v75[0];
    int v77 = (int) v76;
    int v78 = v77 >> 6;
    int v79 = v78 << 4;
    const uint8_t* v80 = v8 + 12;
    const uint8_t* v81 = (const uint8_t*) v80;
    const uint8_t v82 = v81[0];
    int v83 = (int) v82;
    int v84 = v83 & 15;
    int v85 = v84 | v79;
    const uint8_t* v86 = v8 + 8;
    const uint8_t* v87 = (const uint8_t*) v86;
    const uint8_t v88 = v87[0];
    int v89 = (int) v88;
    int v90 = v89 >> 6;
    int v91 = v90 << 4;
    const uint8_t* v92 = v8 + 12;
    const uint8_t* v93 = (const uint8_t*) v92;
    const uint8_t v94 = v93[0];
    int v95 = (int) v94;
    int v96 = v95 >> 4;
    int v97 = v96 | v91;
    const uint8_t* v98 = v8 + 5;
    const uint8_t* v99 = (const uint8_t*) v98;
    const uint8_t v100 = v99[0];
    int v101 = (int) v100;
    int v102 = v101 >> 6;
    int v103 = v102 << 4;
    const uint8_t* v104 = v8 + 13;
    const uint8_t* v105 = (const uint8_t*) v104;
    const uint8_t v106 = v105[0];
    int v107 = (int) v106;
    int v108 = v107 & 15;
    int v109 = v108 | v103;
    const uint8_t* v110 = v8 + 9;
    const uint8_t* v111 = (const uint8_t*) v110;
    const uint8_t v112 = v111[0];
    int v113 = (int) v112;
    int v114 = v113 >> 6;
    int v115 = v114 << 4;
    const uint8_t* v116 = v8 + 13;
    const uint8_t* v117 = (const uint8_t*) v116;
    const uint8_t v118 = v117[0];
    int v119 = (int) v118;
    int v120 = v119 >> 4;
    int v121 = v120 | v115;
    float v122 = (float) v85;
    float v123 = v12 * v122;
    float v124 = (float) v97;
    float v125 = v14 * v124;
    float v126 = (float) v109;
    float v127 = v12 * v126;
    float v128 = (float) v121;
    float v129 = v14 * v128;
    const uint8_t* v130 = v8 + 6;
    const uint8_t* v131 = (const uint8_t*) v130;
    const uint8_t v132 = v131[0];
    int v133 = (int) v132;
    int v134 = v133 >> 6;
    int v135 = v134 << 4;
    const uint8_t* v136 = v8 + 14;
    const uint8_t* v137 = (const uint8_t*) v136;
    const uint8_t v138 = v137[0];
    int v139 = (int) v138;
    int v140 = v139 & 15;
    int v141 = v140 | v135;
    const uint8_t* v142 = v8 + 10;
    const uint8_t* v143 = (const uint8_t*) v142;
    const uint8_t v144 = v143[0];
    int v145 = (int) v144;
    int v146 = v145 >> 6;
    int v147 = v146 << 4;
    const uint8_t* v148 = v8 + 14;
    const uint8_t* v149 = (const uint8_t*) v148;
    const uint8_t v150 = v149[0];
    int v151 = (int) v150;
    int v152 = v151 >> 4;
    int v153 = v152 | v147;
    const uint8_t* v154 = v8 + 7;
    const uint8_t* v155 = (const uint8_t*) v154;
    const uint8_t v156 = v155[0];
    int v157 = (int) v156;
    int v158 = v157 >> 6;
    int v159 = v158 << 4;
    const uint8_t* v160 = v8 + 15;
    const uint8_t* v161 = (const uint8_t*) v160;
    const uint8_t v162 = v161[0];
    int v163 = (int) v162;
    int v164 = v163 & 15;
    int v165 = v164 | v159;
    const uint8_t* v166 = v8 + 11;
    const uint8_t* v167 = (const uint8_t*) v166;
    const uint8_t v168 = v167[0];
    int v169 = (int) v168;
    int v170 = v169 >> 6;
    int v171 = v170 << 4;
    const uint8_t* v172 = v8 + 15;
    const uint8_t* v173 = (const uint8_t*) v172;
    const uint8_t v174 = v173[0];
    int v175 = (int) v174;
    int v176 = v175 >> 4;
    int v177 = v176 | v171;
    float v178 = (float) v141;
    float v179 = v12 * v178;
    float v180 = (float) v153;
    float v181 = v14 * v180;
    float v182 = (float) v165;
    float v183 = v12 * v182;
    float v184 = (float) v177;
    float v185 = v14 * v184;
    const uint8_t* v186 = v8 + 48;
    const uint8_t* v187 = (const uint8_t*) v186;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v188 = __riscv_vle8_v_u8m2(v187, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v189 = __riscv_vand_vx_u8m2(v188, 15, 32);
    float* v190 = v11 + 0;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v191 = __riscv_vsrl_vx_u8m2(v17, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v192 = __riscv_vand_vx_u8m2(v191, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v193 = __riscv_vsll_vx_u8m2(v192, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v194 = __riscv_vor_vv_u8m2(v189, v193, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v195 = __riscv_vzext_vf2_u16m4(v194, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v196 = __riscv_vfwcvt_f_xu_v_f32m8(v195, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v197 = __riscv_vfmv_v_f_f32m8(v41, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v198 = __riscv_vfmsac_vf_f32m8(v197, v39, v196, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v190, v198, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v199 = __riscv_vsrl_vx_u8m2(v188, 4, 32);
    float* v200 = v11 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v201 = __riscv_vsrl_vx_u8m2(v17, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v202 = __riscv_vand_vx_u8m2(v201, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v203 = __riscv_vsll_vx_u8m2(v202, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v204 = __riscv_vor_vv_u8m2(v199, v203, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v205 = __riscv_vzext_vf2_u16m4(v204, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v206 = __riscv_vfwcvt_f_xu_v_f32m8(v205, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v207 = __riscv_vfmv_v_f_f32m8(v45, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v208 = __riscv_vfmsac_vf_f32m8(v207, v43, v206, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v200, v208, 32);
    const uint8_t* v209 = v8 + 80;
    const uint8_t* v210 = (const uint8_t*) v209;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v211 = __riscv_vle8_v_u8m2(v210, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v212 = __riscv_vand_vx_u8m2(v211, 15, 32);
    float* v213 = v11 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v214 = __riscv_vsrl_vx_u8m2(v17, 2, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v215 = __riscv_vand_vx_u8m2(v214, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v216 = __riscv_vsll_vx_u8m2(v215, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v217 = __riscv_vor_vv_u8m2(v212, v216, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v218 = __riscv_vzext_vf2_u16m4(v217, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v219 = __riscv_vfwcvt_f_xu_v_f32m8(v218, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v220 = __riscv_vfmv_v_f_f32m8(v69, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v221 = __riscv_vfmsac_vf_f32m8(v220, v67, v219, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v213, v221, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v222 = __riscv_vsrl_vx_u8m2(v211, 4, 32);
    float* v223 = v11 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v224 = __riscv_vsrl_vx_u8m2(v17, 3, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v225 = __riscv_vand_vx_u8m2(v224, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v226 = __riscv_vsll_vx_u8m2(v225, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v227 = __riscv_vor_vv_u8m2(v222, v226, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v228 = __riscv_vzext_vf2_u16m4(v227, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v229 = __riscv_vfwcvt_f_xu_v_f32m8(v228, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v230 = __riscv_vfmv_v_f_f32m8(v73, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v231 = __riscv_vfmsac_vf_f32m8(v230, v71, v229, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v223, v231, 32);
    const uint8_t* v232 = v8 + 112;
    const uint8_t* v233 = (const uint8_t*) v232;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v234 = __riscv_vle8_v_u8m2(v233, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v235 = __riscv_vand_vx_u8m2(v234, 15, 32);
    float* v236 = v11 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v237 = __riscv_vsrl_vx_u8m2(v17, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v238 = __riscv_vand_vx_u8m2(v237, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v239 = __riscv_vsll_vx_u8m2(v238, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v240 = __riscv_vor_vv_u8m2(v235, v239, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v241 = __riscv_vzext_vf2_u16m4(v240, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v242 = __riscv_vfwcvt_f_xu_v_f32m8(v241, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v243 = __riscv_vfmv_v_f_f32m8(v125, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v244 = __riscv_vfmsac_vf_f32m8(v243, v123, v242, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v236, v244, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v245 = __riscv_vsrl_vx_u8m2(v234, 4, 32);
    float* v246 = v11 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v247 = __riscv_vsrl_vx_u8m2(v17, 5, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v248 = __riscv_vand_vx_u8m2(v247, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v249 = __riscv_vsll_vx_u8m2(v248, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v250 = __riscv_vor_vv_u8m2(v245, v249, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v251 = __riscv_vzext_vf2_u16m4(v250, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v252 = __riscv_vfwcvt_f_xu_v_f32m8(v251, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v253 = __riscv_vfmv_v_f_f32m8(v129, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v254 = __riscv_vfmsac_vf_f32m8(v253, v127, v252, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v246, v254, 32);
    const uint8_t* v255 = v8 + 144;
    const uint8_t* v256 = (const uint8_t*) v255;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v257 = __riscv_vle8_v_u8m2(v256, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v258 = __riscv_vand_vx_u8m2(v257, 15, 32);
    float* v259 = v11 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v260 = __riscv_vsrl_vx_u8m2(v17, 6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v261 = __riscv_vand_vx_u8m2(v260, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v262 = __riscv_vsll_vx_u8m2(v261, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v263 = __riscv_vor_vv_u8m2(v258, v262, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v264 = __riscv_vzext_vf2_u16m4(v263, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v265 = __riscv_vfwcvt_f_xu_v_f32m8(v264, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v266 = __riscv_vfmv_v_f_f32m8(v181, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v267 = __riscv_vfmsac_vf_f32m8(v266, v179, v265, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v259, v267, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v268 = __riscv_vsrl_vx_u8m2(v257, 4, 32);
    float* v269 = v11 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v270 = __riscv_vsrl_vx_u8m2(v17, 7, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v271 = __riscv_vand_vx_u8m2(v270, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v272 = __riscv_vsll_vx_u8m2(v271, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v273 = __riscv_vor_vv_u8m2(v268, v272, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v274 = __riscv_vzext_vf2_u16m4(v273, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v275 = __riscv_vfwcvt_f_xu_v_f32m8(v274, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v276 = __riscv_vfmv_v_f_f32m8(v185, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v277 = __riscv_vfmsac_vf_f32m8(v276, v183, v275, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v269, v277, 32);
  }
  return;
}


