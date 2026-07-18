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
    const uint8_t* v43 = v8 + 16;
    const uint8_t* v44 = (const uint8_t*) v43;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v45 = __riscv_vle8_v_u8m2(v44, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v46 = __riscv_vand_vx_u8m2(v45, 15, 32);
    float* v47 = v11 + 0;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v48 = __riscv_vzext_vf4_u32m8(v46, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v49 = __riscv_vreinterpret_v_u32m8_i32m8(v48);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v50 = __riscv_vfcvt_f_x_v_f32m8(v49, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v51 = __riscv_vfmv_v_f_f32m8(v38, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v52 = __riscv_vfmsac_vf_f32m8(v51, v36, v50, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v47, v52, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v53 = __riscv_vsrl_vx_u8m2(v45, 4, 32);
    float* v54 = v11 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v55 = __riscv_vzext_vf4_u32m8(v53, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v56 = __riscv_vreinterpret_v_u32m8_i32m8(v55);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v57 = __riscv_vfcvt_f_x_v_f32m8(v56, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v58 = __riscv_vfmv_v_f_f32m8(v42, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v59 = __riscv_vfmsac_vf_f32m8(v58, v40, v57, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v54, v59, 32);
    const uint8_t* v60 = v8 + 6;
    const uint8_t* v61 = (const uint8_t*) v60;
    const uint8_t v62 = v61[0];
    int v63 = (int) v62;
    int v64 = v63 & 63;
    const uint8_t* v65 = v8 + 10;
    const uint8_t* v66 = (const uint8_t*) v65;
    const uint8_t v67 = v66[0];
    int v68 = (int) v67;
    int v69 = v68 & 63;
    const uint8_t* v70 = v8 + 7;
    const uint8_t* v71 = (const uint8_t*) v70;
    const uint8_t v72 = v71[0];
    int v73 = (int) v72;
    int v74 = v73 & 63;
    const uint8_t* v75 = v8 + 11;
    const uint8_t* v76 = (const uint8_t*) v75;
    const uint8_t v77 = v76[0];
    int v78 = (int) v77;
    int v79 = v78 & 63;
    float v80 = (float) v64;
    float v81 = v12 * v80;
    float v82 = (float) v69;
    float v83 = v14 * v82;
    float v84 = (float) v74;
    float v85 = v12 * v84;
    float v86 = (float) v79;
    float v87 = v14 * v86;
    const uint8_t* v88 = v8 + 48;
    const uint8_t* v89 = (const uint8_t*) v88;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v90 = __riscv_vle8_v_u8m2(v89, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v91 = __riscv_vand_vx_u8m2(v90, 15, 32);
    float* v92 = v11 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v93 = __riscv_vzext_vf4_u32m8(v91, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v94 = __riscv_vreinterpret_v_u32m8_i32m8(v93);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v95 = __riscv_vfcvt_f_x_v_f32m8(v94, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v96 = __riscv_vfmv_v_f_f32m8(v83, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v97 = __riscv_vfmsac_vf_f32m8(v96, v81, v95, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v92, v97, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v98 = __riscv_vsrl_vx_u8m2(v90, 4, 32);
    float* v99 = v11 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v100 = __riscv_vzext_vf4_u32m8(v98, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v101 = __riscv_vreinterpret_v_u32m8_i32m8(v100);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v102 = __riscv_vfcvt_f_x_v_f32m8(v101, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v103 = __riscv_vfmv_v_f_f32m8(v87, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v104 = __riscv_vfmsac_vf_f32m8(v103, v85, v102, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v99, v104, 32);
    const uint8_t* v105 = v8 + 4;
    const uint8_t* v106 = (const uint8_t*) v105;
    const uint8_t v107 = v106[0];
    int v108 = (int) v107;
    int v109 = v108 >> 6;
    int v110 = v109 << 4;
    const uint8_t* v111 = v8 + 12;
    const uint8_t* v112 = (const uint8_t*) v111;
    const uint8_t v113 = v112[0];
    int v114 = (int) v113;
    int v115 = v114 & 15;
    int v116 = v115 | v110;
    const uint8_t* v117 = v8 + 8;
    const uint8_t* v118 = (const uint8_t*) v117;
    const uint8_t v119 = v118[0];
    int v120 = (int) v119;
    int v121 = v120 >> 6;
    int v122 = v121 << 4;
    const uint8_t* v123 = v8 + 12;
    const uint8_t* v124 = (const uint8_t*) v123;
    const uint8_t v125 = v124[0];
    int v126 = (int) v125;
    int v127 = v126 >> 4;
    int v128 = v127 | v122;
    const uint8_t* v129 = v8 + 5;
    const uint8_t* v130 = (const uint8_t*) v129;
    const uint8_t v131 = v130[0];
    int v132 = (int) v131;
    int v133 = v132 >> 6;
    int v134 = v133 << 4;
    const uint8_t* v135 = v8 + 13;
    const uint8_t* v136 = (const uint8_t*) v135;
    const uint8_t v137 = v136[0];
    int v138 = (int) v137;
    int v139 = v138 & 15;
    int v140 = v139 | v134;
    const uint8_t* v141 = v8 + 9;
    const uint8_t* v142 = (const uint8_t*) v141;
    const uint8_t v143 = v142[0];
    int v144 = (int) v143;
    int v145 = v144 >> 6;
    int v146 = v145 << 4;
    const uint8_t* v147 = v8 + 13;
    const uint8_t* v148 = (const uint8_t*) v147;
    const uint8_t v149 = v148[0];
    int v150 = (int) v149;
    int v151 = v150 >> 4;
    int v152 = v151 | v146;
    float v153 = (float) v116;
    float v154 = v12 * v153;
    float v155 = (float) v128;
    float v156 = v14 * v155;
    float v157 = (float) v140;
    float v158 = v12 * v157;
    float v159 = (float) v152;
    float v160 = v14 * v159;
    const uint8_t* v161 = v8 + 80;
    const uint8_t* v162 = (const uint8_t*) v161;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v163 = __riscv_vle8_v_u8m2(v162, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v164 = __riscv_vand_vx_u8m2(v163, 15, 32);
    float* v165 = v11 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v166 = __riscv_vzext_vf4_u32m8(v164, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v167 = __riscv_vreinterpret_v_u32m8_i32m8(v166);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v168 = __riscv_vfcvt_f_x_v_f32m8(v167, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v169 = __riscv_vfmv_v_f_f32m8(v156, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v170 = __riscv_vfmsac_vf_f32m8(v169, v154, v168, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v165, v170, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v171 = __riscv_vsrl_vx_u8m2(v163, 4, 32);
    float* v172 = v11 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v173 = __riscv_vzext_vf4_u32m8(v171, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v174 = __riscv_vreinterpret_v_u32m8_i32m8(v173);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v175 = __riscv_vfcvt_f_x_v_f32m8(v174, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v176 = __riscv_vfmv_v_f_f32m8(v160, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v177 = __riscv_vfmsac_vf_f32m8(v176, v158, v175, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v172, v177, 32);
    const uint8_t* v178 = v8 + 6;
    const uint8_t* v179 = (const uint8_t*) v178;
    const uint8_t v180 = v179[0];
    int v181 = (int) v180;
    int v182 = v181 >> 6;
    int v183 = v182 << 4;
    const uint8_t* v184 = v8 + 14;
    const uint8_t* v185 = (const uint8_t*) v184;
    const uint8_t v186 = v185[0];
    int v187 = (int) v186;
    int v188 = v187 & 15;
    int v189 = v188 | v183;
    const uint8_t* v190 = v8 + 10;
    const uint8_t* v191 = (const uint8_t*) v190;
    const uint8_t v192 = v191[0];
    int v193 = (int) v192;
    int v194 = v193 >> 6;
    int v195 = v194 << 4;
    const uint8_t* v196 = v8 + 14;
    const uint8_t* v197 = (const uint8_t*) v196;
    const uint8_t v198 = v197[0];
    int v199 = (int) v198;
    int v200 = v199 >> 4;
    int v201 = v200 | v195;
    const uint8_t* v202 = v8 + 7;
    const uint8_t* v203 = (const uint8_t*) v202;
    const uint8_t v204 = v203[0];
    int v205 = (int) v204;
    int v206 = v205 >> 6;
    int v207 = v206 << 4;
    const uint8_t* v208 = v8 + 15;
    const uint8_t* v209 = (const uint8_t*) v208;
    const uint8_t v210 = v209[0];
    int v211 = (int) v210;
    int v212 = v211 & 15;
    int v213 = v212 | v207;
    const uint8_t* v214 = v8 + 11;
    const uint8_t* v215 = (const uint8_t*) v214;
    const uint8_t v216 = v215[0];
    int v217 = (int) v216;
    int v218 = v217 >> 6;
    int v219 = v218 << 4;
    const uint8_t* v220 = v8 + 15;
    const uint8_t* v221 = (const uint8_t*) v220;
    const uint8_t v222 = v221[0];
    int v223 = (int) v222;
    int v224 = v223 >> 4;
    int v225 = v224 | v219;
    float v226 = (float) v189;
    float v227 = v12 * v226;
    float v228 = (float) v201;
    float v229 = v14 * v228;
    float v230 = (float) v213;
    float v231 = v12 * v230;
    float v232 = (float) v225;
    float v233 = v14 * v232;
    const uint8_t* v234 = v8 + 112;
    const uint8_t* v235 = (const uint8_t*) v234;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v236 = __riscv_vle8_v_u8m2(v235, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v237 = __riscv_vand_vx_u8m2(v236, 15, 32);
    float* v238 = v11 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v239 = __riscv_vzext_vf4_u32m8(v237, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v240 = __riscv_vreinterpret_v_u32m8_i32m8(v239);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v241 = __riscv_vfcvt_f_x_v_f32m8(v240, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v242 = __riscv_vfmv_v_f_f32m8(v229, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v243 = __riscv_vfmsac_vf_f32m8(v242, v227, v241, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v238, v243, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v244 = __riscv_vsrl_vx_u8m2(v236, 4, 32);
    float* v245 = v11 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v246 = __riscv_vzext_vf4_u32m8(v244, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v247 = __riscv_vreinterpret_v_u32m8_i32m8(v246);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v248 = __riscv_vfcvt_f_x_v_f32m8(v247, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v249 = __riscv_vfmv_v_f_f32m8(v233, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v250 = __riscv_vfmsac_vf_f32m8(v249, v231, v248, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v245, v250, 32);
  }
  return;
}


