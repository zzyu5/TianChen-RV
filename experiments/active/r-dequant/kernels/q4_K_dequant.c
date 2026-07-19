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
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=q4_K_decode
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v12 = (float)*(const _Float16 *)(v8);
    const uint8_t* v13 = v8 + 2;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v13);
    const uint8_t* v15 = v8 + 4;
    const uint8_t* v16 = (const uint8_t*) v15;
    const uint8_t v17 = v16[0];
    int v18 = (int) v17;
    int v19 = v18 & 63;
    const uint8_t* v20 = v8 + 8;
    const uint8_t* v21 = (const uint8_t*) v20;
    const uint8_t v22 = v21[0];
    int v23 = (int) v22;
    int v24 = v23 & 63;
    const uint8_t* v25 = v8 + 5;
    const uint8_t* v26 = (const uint8_t*) v25;
    const uint8_t v27 = v26[0];
    int v28 = (int) v27;
    int v29 = v28 & 63;
    const uint8_t* v30 = v8 + 9;
    const uint8_t* v31 = (const uint8_t*) v30;
    const uint8_t v32 = v31[0];
    int v33 = (int) v32;
    int v34 = v33 & 63;
    float v35 = (float) v19;
    float v36 = v12 * v35;
    float v37 = (float) v24;
    float v38 = v14 * v37;
    float v39 = (float) v29;
    float v40 = v12 * v39;
    float v41 = (float) v34;
    float v42 = v14 * v41;
    const uint8_t* v43 = v8 + 6;
    const uint8_t* v44 = (const uint8_t*) v43;
    const uint8_t v45 = v44[0];
    int v46 = (int) v45;
    int v47 = v46 & 63;
    const uint8_t* v48 = v8 + 10;
    const uint8_t* v49 = (const uint8_t*) v48;
    const uint8_t v50 = v49[0];
    int v51 = (int) v50;
    int v52 = v51 & 63;
    const uint8_t* v53 = v8 + 7;
    const uint8_t* v54 = (const uint8_t*) v53;
    const uint8_t v55 = v54[0];
    int v56 = (int) v55;
    int v57 = v56 & 63;
    const uint8_t* v58 = v8 + 11;
    const uint8_t* v59 = (const uint8_t*) v58;
    const uint8_t v60 = v59[0];
    int v61 = (int) v60;
    int v62 = v61 & 63;
    float v63 = (float) v47;
    float v64 = v12 * v63;
    float v65 = (float) v52;
    float v66 = v14 * v65;
    float v67 = (float) v57;
    float v68 = v12 * v67;
    float v69 = (float) v62;
    float v70 = v14 * v69;
    const uint8_t* v71 = v8 + 4;
    const uint8_t* v72 = (const uint8_t*) v71;
    const uint8_t v73 = v72[0];
    int v74 = (int) v73;
    int v75 = v74 >> 6;
    int v76 = v75 << 4;
    const uint8_t* v77 = v8 + 12;
    const uint8_t* v78 = (const uint8_t*) v77;
    const uint8_t v79 = v78[0];
    int v80 = (int) v79;
    int v81 = v80 & 15;
    int v82 = v81 | v76;
    const uint8_t* v83 = v8 + 8;
    const uint8_t* v84 = (const uint8_t*) v83;
    const uint8_t v85 = v84[0];
    int v86 = (int) v85;
    int v87 = v86 >> 6;
    int v88 = v87 << 4;
    const uint8_t* v89 = v8 + 12;
    const uint8_t* v90 = (const uint8_t*) v89;
    const uint8_t v91 = v90[0];
    int v92 = (int) v91;
    int v93 = v92 >> 4;
    int v94 = v93 | v88;
    const uint8_t* v95 = v8 + 5;
    const uint8_t* v96 = (const uint8_t*) v95;
    const uint8_t v97 = v96[0];
    int v98 = (int) v97;
    int v99 = v98 >> 6;
    int v100 = v99 << 4;
    const uint8_t* v101 = v8 + 13;
    const uint8_t* v102 = (const uint8_t*) v101;
    const uint8_t v103 = v102[0];
    int v104 = (int) v103;
    int v105 = v104 & 15;
    int v106 = v105 | v100;
    const uint8_t* v107 = v8 + 9;
    const uint8_t* v108 = (const uint8_t*) v107;
    const uint8_t v109 = v108[0];
    int v110 = (int) v109;
    int v111 = v110 >> 6;
    int v112 = v111 << 4;
    const uint8_t* v113 = v8 + 13;
    const uint8_t* v114 = (const uint8_t*) v113;
    const uint8_t v115 = v114[0];
    int v116 = (int) v115;
    int v117 = v116 >> 4;
    int v118 = v117 | v112;
    float v119 = (float) v82;
    float v120 = v12 * v119;
    float v121 = (float) v94;
    float v122 = v14 * v121;
    float v123 = (float) v106;
    float v124 = v12 * v123;
    float v125 = (float) v118;
    float v126 = v14 * v125;
    const uint8_t* v127 = v8 + 6;
    const uint8_t* v128 = (const uint8_t*) v127;
    const uint8_t v129 = v128[0];
    int v130 = (int) v129;
    int v131 = v130 >> 6;
    int v132 = v131 << 4;
    const uint8_t* v133 = v8 + 14;
    const uint8_t* v134 = (const uint8_t*) v133;
    const uint8_t v135 = v134[0];
    int v136 = (int) v135;
    int v137 = v136 & 15;
    int v138 = v137 | v132;
    const uint8_t* v139 = v8 + 10;
    const uint8_t* v140 = (const uint8_t*) v139;
    const uint8_t v141 = v140[0];
    int v142 = (int) v141;
    int v143 = v142 >> 6;
    int v144 = v143 << 4;
    const uint8_t* v145 = v8 + 14;
    const uint8_t* v146 = (const uint8_t*) v145;
    const uint8_t v147 = v146[0];
    int v148 = (int) v147;
    int v149 = v148 >> 4;
    int v150 = v149 | v144;
    const uint8_t* v151 = v8 + 7;
    const uint8_t* v152 = (const uint8_t*) v151;
    const uint8_t v153 = v152[0];
    int v154 = (int) v153;
    int v155 = v154 >> 6;
    int v156 = v155 << 4;
    const uint8_t* v157 = v8 + 15;
    const uint8_t* v158 = (const uint8_t*) v157;
    const uint8_t v159 = v158[0];
    int v160 = (int) v159;
    int v161 = v160 & 15;
    int v162 = v161 | v156;
    const uint8_t* v163 = v8 + 11;
    const uint8_t* v164 = (const uint8_t*) v163;
    const uint8_t v165 = v164[0];
    int v166 = (int) v165;
    int v167 = v166 >> 6;
    int v168 = v167 << 4;
    const uint8_t* v169 = v8 + 15;
    const uint8_t* v170 = (const uint8_t*) v169;
    const uint8_t v171 = v170[0];
    int v172 = (int) v171;
    int v173 = v172 >> 4;
    int v174 = v173 | v168;
    float v175 = (float) v138;
    float v176 = v12 * v175;
    float v177 = (float) v150;
    float v178 = v14 * v177;
    float v179 = (float) v162;
    float v180 = v12 * v179;
    float v181 = (float) v174;
    float v182 = v14 * v181;
    const uint8_t* v183 = v8 + 16;
    const uint8_t* v184 = (const uint8_t*) v183;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v185 = __riscv_vle8_v_u8m2(v184, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v186 = __riscv_vand_vx_u8m2(v185, 15, 32);
    float* v187 = v11 + 0;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v188 = __riscv_vzext_vf2_u16m4(v186, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v189 = __riscv_vfwcvt_f_xu_v_f32m8(v188, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v190 = __riscv_vfmv_v_f_f32m8(v38, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v191 = __riscv_vfmsac_vf_f32m8(v190, v36, v189, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v187, v191, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v192 = __riscv_vsrl_vx_u8m2(v185, 4, 32);
    float* v193 = v11 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v194 = __riscv_vzext_vf2_u16m4(v192, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v195 = __riscv_vfwcvt_f_xu_v_f32m8(v194, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v196 = __riscv_vfmv_v_f_f32m8(v42, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v197 = __riscv_vfmsac_vf_f32m8(v196, v40, v195, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v193, v197, 32);
    const uint8_t* v198 = v8 + 48;
    const uint8_t* v199 = (const uint8_t*) v198;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v200 = __riscv_vle8_v_u8m2(v199, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v201 = __riscv_vand_vx_u8m2(v200, 15, 32);
    float* v202 = v11 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v203 = __riscv_vzext_vf2_u16m4(v201, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v204 = __riscv_vfwcvt_f_xu_v_f32m8(v203, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v205 = __riscv_vfmv_v_f_f32m8(v66, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v206 = __riscv_vfmsac_vf_f32m8(v205, v64, v204, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v202, v206, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v207 = __riscv_vsrl_vx_u8m2(v200, 4, 32);
    float* v208 = v11 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v209 = __riscv_vzext_vf2_u16m4(v207, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v210 = __riscv_vfwcvt_f_xu_v_f32m8(v209, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v211 = __riscv_vfmv_v_f_f32m8(v70, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v212 = __riscv_vfmsac_vf_f32m8(v211, v68, v210, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v208, v212, 32);
    const uint8_t* v213 = v8 + 80;
    const uint8_t* v214 = (const uint8_t*) v213;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v215 = __riscv_vle8_v_u8m2(v214, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v216 = __riscv_vand_vx_u8m2(v215, 15, 32);
    float* v217 = v11 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v218 = __riscv_vzext_vf2_u16m4(v216, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v219 = __riscv_vfwcvt_f_xu_v_f32m8(v218, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v220 = __riscv_vfmv_v_f_f32m8(v122, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v221 = __riscv_vfmsac_vf_f32m8(v220, v120, v219, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v217, v221, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v222 = __riscv_vsrl_vx_u8m2(v215, 4, 32);
    float* v223 = v11 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v224 = __riscv_vzext_vf2_u16m4(v222, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v225 = __riscv_vfwcvt_f_xu_v_f32m8(v224, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v226 = __riscv_vfmv_v_f_f32m8(v126, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v227 = __riscv_vfmsac_vf_f32m8(v226, v124, v225, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v223, v227, 32);
    const uint8_t* v228 = v8 + 112;
    const uint8_t* v229 = (const uint8_t*) v228;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v230 = __riscv_vle8_v_u8m2(v229, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v231 = __riscv_vand_vx_u8m2(v230, 15, 32);
    float* v232 = v11 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v233 = __riscv_vzext_vf2_u16m4(v231, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v234 = __riscv_vfwcvt_f_xu_v_f32m8(v233, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v235 = __riscv_vfmv_v_f_f32m8(v178, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v236 = __riscv_vfmsac_vf_f32m8(v235, v176, v234, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v232, v236, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v237 = __riscv_vsrl_vx_u8m2(v230, 4, 32);
    float* v238 = v11 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m4
    vuint16m4_t v239 = __riscv_vzext_vf2_u16m4(v237, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_xu_v_f32m8
    vfloat32m8_t v240 = __riscv_vfwcvt_f_xu_v_f32m8(v239, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v241 = __riscv_vfmv_v_f_f32m8(v182, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v242 = __riscv_vfmsac_vf_f32m8(v241, v180, v240, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v238, v242, 32);
  }
  return;
}


