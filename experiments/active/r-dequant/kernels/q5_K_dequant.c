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
    const uint8_t* v46 = v8 + 48;
    const uint8_t* v47 = (const uint8_t*) v46;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v48 = __riscv_vle8_v_u8m2(v47, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v49 = __riscv_vand_vx_u8m2(v48, 15, 32);
    float* v50 = v11 + 0;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v51 = __riscv_vsrl_vx_u8m2(v17, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v52 = __riscv_vand_vx_u8m2(v51, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v53 = __riscv_vsll_vx_u8m2(v52, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v54 = __riscv_vor_vv_u8m2(v49, v53, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v55 = __riscv_vzext_vf4_u32m8(v54, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v56 = __riscv_vreinterpret_v_u32m8_i32m8(v55);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v57 = __riscv_vfcvt_f_x_v_f32m8(v56, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v58 = __riscv_vfmv_v_f_f32m8(v41, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v59 = __riscv_vfmsac_vf_f32m8(v58, v39, v57, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v50, v59, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v60 = __riscv_vsrl_vx_u8m2(v48, 4, 32);
    float* v61 = v11 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v62 = __riscv_vsrl_vx_u8m2(v17, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v63 = __riscv_vand_vx_u8m2(v62, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v64 = __riscv_vsll_vx_u8m2(v63, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v65 = __riscv_vor_vv_u8m2(v60, v64, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v66 = __riscv_vzext_vf4_u32m8(v65, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v67 = __riscv_vreinterpret_v_u32m8_i32m8(v66);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v68 = __riscv_vfcvt_f_x_v_f32m8(v67, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v69 = __riscv_vfmv_v_f_f32m8(v45, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v70 = __riscv_vfmsac_vf_f32m8(v69, v43, v68, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v61, v70, 32);
    const uint8_t* v71 = v8 + 6;
    const uint8_t* v72 = (const uint8_t*) v71;
    const uint8_t v73 = v72[0];
    int v74 = (int) v73;
    int v75 = v74 & 63;
    const uint8_t* v76 = v8 + 10;
    const uint8_t* v77 = (const uint8_t*) v76;
    const uint8_t v78 = v77[0];
    int v79 = (int) v78;
    int v80 = v79 & 63;
    const uint8_t* v81 = v8 + 7;
    const uint8_t* v82 = (const uint8_t*) v81;
    const uint8_t v83 = v82[0];
    int v84 = (int) v83;
    int v85 = v84 & 63;
    const uint8_t* v86 = v8 + 11;
    const uint8_t* v87 = (const uint8_t*) v86;
    const uint8_t v88 = v87[0];
    int v89 = (int) v88;
    int v90 = v89 & 63;
    float v91 = (float) v75;
    float v92 = v12 * v91;
    float v93 = (float) v80;
    float v94 = v14 * v93;
    float v95 = (float) v85;
    float v96 = v12 * v95;
    float v97 = (float) v90;
    float v98 = v14 * v97;
    const uint8_t* v99 = v8 + 80;
    const uint8_t* v100 = (const uint8_t*) v99;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v101 = __riscv_vle8_v_u8m2(v100, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v102 = __riscv_vand_vx_u8m2(v101, 15, 32);
    float* v103 = v11 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v104 = __riscv_vsrl_vx_u8m2(v17, 2, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v105 = __riscv_vand_vx_u8m2(v104, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v106 = __riscv_vsll_vx_u8m2(v105, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v107 = __riscv_vor_vv_u8m2(v102, v106, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v108 = __riscv_vzext_vf4_u32m8(v107, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v109 = __riscv_vreinterpret_v_u32m8_i32m8(v108);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v110 = __riscv_vfcvt_f_x_v_f32m8(v109, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v111 = __riscv_vfmv_v_f_f32m8(v94, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v112 = __riscv_vfmsac_vf_f32m8(v111, v92, v110, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v103, v112, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v113 = __riscv_vsrl_vx_u8m2(v101, 4, 32);
    float* v114 = v11 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v115 = __riscv_vsrl_vx_u8m2(v17, 3, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v116 = __riscv_vand_vx_u8m2(v115, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v117 = __riscv_vsll_vx_u8m2(v116, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v118 = __riscv_vor_vv_u8m2(v113, v117, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v119 = __riscv_vzext_vf4_u32m8(v118, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v120 = __riscv_vreinterpret_v_u32m8_i32m8(v119);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v121 = __riscv_vfcvt_f_x_v_f32m8(v120, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v122 = __riscv_vfmv_v_f_f32m8(v98, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v123 = __riscv_vfmsac_vf_f32m8(v122, v96, v121, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v114, v123, 32);
    const uint8_t* v124 = v8 + 4;
    const uint8_t* v125 = (const uint8_t*) v124;
    const uint8_t v126 = v125[0];
    int v127 = (int) v126;
    int v128 = v127 >> 6;
    int v129 = v128 << 4;
    const uint8_t* v130 = v8 + 12;
    const uint8_t* v131 = (const uint8_t*) v130;
    const uint8_t v132 = v131[0];
    int v133 = (int) v132;
    int v134 = v133 & 15;
    int v135 = v134 | v129;
    const uint8_t* v136 = v8 + 8;
    const uint8_t* v137 = (const uint8_t*) v136;
    const uint8_t v138 = v137[0];
    int v139 = (int) v138;
    int v140 = v139 >> 6;
    int v141 = v140 << 4;
    const uint8_t* v142 = v8 + 12;
    const uint8_t* v143 = (const uint8_t*) v142;
    const uint8_t v144 = v143[0];
    int v145 = (int) v144;
    int v146 = v145 >> 4;
    int v147 = v146 | v141;
    const uint8_t* v148 = v8 + 5;
    const uint8_t* v149 = (const uint8_t*) v148;
    const uint8_t v150 = v149[0];
    int v151 = (int) v150;
    int v152 = v151 >> 6;
    int v153 = v152 << 4;
    const uint8_t* v154 = v8 + 13;
    const uint8_t* v155 = (const uint8_t*) v154;
    const uint8_t v156 = v155[0];
    int v157 = (int) v156;
    int v158 = v157 & 15;
    int v159 = v158 | v153;
    const uint8_t* v160 = v8 + 9;
    const uint8_t* v161 = (const uint8_t*) v160;
    const uint8_t v162 = v161[0];
    int v163 = (int) v162;
    int v164 = v163 >> 6;
    int v165 = v164 << 4;
    const uint8_t* v166 = v8 + 13;
    const uint8_t* v167 = (const uint8_t*) v166;
    const uint8_t v168 = v167[0];
    int v169 = (int) v168;
    int v170 = v169 >> 4;
    int v171 = v170 | v165;
    float v172 = (float) v135;
    float v173 = v12 * v172;
    float v174 = (float) v147;
    float v175 = v14 * v174;
    float v176 = (float) v159;
    float v177 = v12 * v176;
    float v178 = (float) v171;
    float v179 = v14 * v178;
    const uint8_t* v180 = v8 + 112;
    const uint8_t* v181 = (const uint8_t*) v180;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v182 = __riscv_vle8_v_u8m2(v181, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v183 = __riscv_vand_vx_u8m2(v182, 15, 32);
    float* v184 = v11 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v185 = __riscv_vsrl_vx_u8m2(v17, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v186 = __riscv_vand_vx_u8m2(v185, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v187 = __riscv_vsll_vx_u8m2(v186, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v188 = __riscv_vor_vv_u8m2(v183, v187, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v189 = __riscv_vzext_vf4_u32m8(v188, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v190 = __riscv_vreinterpret_v_u32m8_i32m8(v189);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v191 = __riscv_vfcvt_f_x_v_f32m8(v190, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v192 = __riscv_vfmv_v_f_f32m8(v175, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v193 = __riscv_vfmsac_vf_f32m8(v192, v173, v191, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v184, v193, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v194 = __riscv_vsrl_vx_u8m2(v182, 4, 32);
    float* v195 = v11 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v196 = __riscv_vsrl_vx_u8m2(v17, 5, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v197 = __riscv_vand_vx_u8m2(v196, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v198 = __riscv_vsll_vx_u8m2(v197, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v199 = __riscv_vor_vv_u8m2(v194, v198, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v200 = __riscv_vzext_vf4_u32m8(v199, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v201 = __riscv_vreinterpret_v_u32m8_i32m8(v200);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v202 = __riscv_vfcvt_f_x_v_f32m8(v201, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v203 = __riscv_vfmv_v_f_f32m8(v179, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v204 = __riscv_vfmsac_vf_f32m8(v203, v177, v202, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v195, v204, 32);
    const uint8_t* v205 = v8 + 6;
    const uint8_t* v206 = (const uint8_t*) v205;
    const uint8_t v207 = v206[0];
    int v208 = (int) v207;
    int v209 = v208 >> 6;
    int v210 = v209 << 4;
    const uint8_t* v211 = v8 + 14;
    const uint8_t* v212 = (const uint8_t*) v211;
    const uint8_t v213 = v212[0];
    int v214 = (int) v213;
    int v215 = v214 & 15;
    int v216 = v215 | v210;
    const uint8_t* v217 = v8 + 10;
    const uint8_t* v218 = (const uint8_t*) v217;
    const uint8_t v219 = v218[0];
    int v220 = (int) v219;
    int v221 = v220 >> 6;
    int v222 = v221 << 4;
    const uint8_t* v223 = v8 + 14;
    const uint8_t* v224 = (const uint8_t*) v223;
    const uint8_t v225 = v224[0];
    int v226 = (int) v225;
    int v227 = v226 >> 4;
    int v228 = v227 | v222;
    const uint8_t* v229 = v8 + 7;
    const uint8_t* v230 = (const uint8_t*) v229;
    const uint8_t v231 = v230[0];
    int v232 = (int) v231;
    int v233 = v232 >> 6;
    int v234 = v233 << 4;
    const uint8_t* v235 = v8 + 15;
    const uint8_t* v236 = (const uint8_t*) v235;
    const uint8_t v237 = v236[0];
    int v238 = (int) v237;
    int v239 = v238 & 15;
    int v240 = v239 | v234;
    const uint8_t* v241 = v8 + 11;
    const uint8_t* v242 = (const uint8_t*) v241;
    const uint8_t v243 = v242[0];
    int v244 = (int) v243;
    int v245 = v244 >> 6;
    int v246 = v245 << 4;
    const uint8_t* v247 = v8 + 15;
    const uint8_t* v248 = (const uint8_t*) v247;
    const uint8_t v249 = v248[0];
    int v250 = (int) v249;
    int v251 = v250 >> 4;
    int v252 = v251 | v246;
    float v253 = (float) v216;
    float v254 = v12 * v253;
    float v255 = (float) v228;
    float v256 = v14 * v255;
    float v257 = (float) v240;
    float v258 = v12 * v257;
    float v259 = (float) v252;
    float v260 = v14 * v259;
    const uint8_t* v261 = v8 + 144;
    const uint8_t* v262 = (const uint8_t*) v261;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v263 = __riscv_vle8_v_u8m2(v262, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v264 = __riscv_vand_vx_u8m2(v263, 15, 32);
    float* v265 = v11 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v266 = __riscv_vsrl_vx_u8m2(v17, 6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v267 = __riscv_vand_vx_u8m2(v266, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v268 = __riscv_vsll_vx_u8m2(v267, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v269 = __riscv_vor_vv_u8m2(v264, v268, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v270 = __riscv_vzext_vf4_u32m8(v269, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v271 = __riscv_vreinterpret_v_u32m8_i32m8(v270);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v272 = __riscv_vfcvt_f_x_v_f32m8(v271, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v273 = __riscv_vfmv_v_f_f32m8(v256, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v274 = __riscv_vfmsac_vf_f32m8(v273, v254, v272, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v265, v274, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v275 = __riscv_vsrl_vx_u8m2(v263, 4, 32);
    float* v276 = v11 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v277 = __riscv_vsrl_vx_u8m2(v17, 7, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v278 = __riscv_vand_vx_u8m2(v277, 1, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v279 = __riscv_vsll_vx_u8m2(v278, 4, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v280 = __riscv_vor_vv_u8m2(v275, v279, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m8
    vuint32m8_t v281 = __riscv_vzext_vf4_u32m8(v280, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m8_i32m8
    vint32m8_t v282 = __riscv_vreinterpret_v_u32m8_i32m8(v281);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v283 = __riscv_vfcvt_f_x_v_f32m8(v282, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v284 = __riscv_vfmv_v_f_f32m8(v260, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m8
    vfloat32m8_t v285 = __riscv_vfmsac_vf_f32m8(v284, v258, v283, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v276, v285, 32);
  }
  return;
}


