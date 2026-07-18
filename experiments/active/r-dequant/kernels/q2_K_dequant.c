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
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=q2_K_decode
    const uint8_t* v12 = v8 + 80;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v12);
    const uint8_t* v14 = v8 + 82;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v14);
    const uint8_t* v16 = v8 + 0;
    const uint8_t* v17 = (const uint8_t*) v16;
    const uint8_t v18 = v17[0];
    int v19 = (int) v18;
    int v20 = v19 & 15;
    float v21 = (float) v20;
    float v22 = v13 * v21;
    int v23 = v19 >> 4;
    float v24 = (float) v23;
    float v25 = v15 * v24;
    const uint8_t* v26 = v8 + 16;
    const uint8_t* v27 = (const uint8_t*) v26;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v28 = __riscv_vle8_v_u8m1(v27, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v29 = __riscv_vsrl_vx_u8m1(v28, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v30 = __riscv_vand_vx_u8m1(v29, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v31 = __riscv_vzext_vf4_u32m4(v30, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v32 = __riscv_vreinterpret_v_u32m4_i32m4(v31);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v33 = __riscv_vfcvt_f_x_v_f32m4(v32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v34 = __riscv_vfmv_v_f_f32m4(v25, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v35 = __riscv_vfmsac_vf_f32m4(v34, v22, v33, 16);
    float* v36 = v11 + 0;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v36, v35, 16);
    const uint8_t* v37 = v8 + 1;
    const uint8_t* v38 = (const uint8_t*) v37;
    const uint8_t v39 = v38[0];
    int v40 = (int) v39;
    int v41 = v40 & 15;
    float v42 = (float) v41;
    float v43 = v13 * v42;
    int v44 = v40 >> 4;
    float v45 = (float) v44;
    float v46 = v15 * v45;
    const uint8_t* v47 = v8 + 32;
    const uint8_t* v48 = (const uint8_t*) v47;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v49 = __riscv_vle8_v_u8m1(v48, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v50 = __riscv_vsrl_vx_u8m1(v49, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v51 = __riscv_vand_vx_u8m1(v50, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v52 = __riscv_vzext_vf4_u32m4(v51, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v53 = __riscv_vreinterpret_v_u32m4_i32m4(v52);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v54 = __riscv_vfcvt_f_x_v_f32m4(v53, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v55 = __riscv_vfmv_v_f_f32m4(v46, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v56 = __riscv_vfmsac_vf_f32m4(v55, v43, v54, 16);
    float* v57 = v11 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v57, v56, 16);
    const uint8_t* v58 = v8 + 2;
    const uint8_t* v59 = (const uint8_t*) v58;
    const uint8_t v60 = v59[0];
    int v61 = (int) v60;
    int v62 = v61 & 15;
    float v63 = (float) v62;
    float v64 = v13 * v63;
    int v65 = v61 >> 4;
    float v66 = (float) v65;
    float v67 = v15 * v66;
    const uint8_t* v68 = v8 + 16;
    const uint8_t* v69 = (const uint8_t*) v68;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v70 = __riscv_vle8_v_u8m1(v69, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v71 = __riscv_vsrl_vx_u8m1(v70, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v72 = __riscv_vand_vx_u8m1(v71, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v73 = __riscv_vzext_vf4_u32m4(v72, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v74 = __riscv_vreinterpret_v_u32m4_i32m4(v73);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v75 = __riscv_vfcvt_f_x_v_f32m4(v74, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v76 = __riscv_vfmv_v_f_f32m4(v67, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v77 = __riscv_vfmsac_vf_f32m4(v76, v64, v75, 16);
    float* v78 = v11 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v78, v77, 16);
    const uint8_t* v79 = v8 + 3;
    const uint8_t* v80 = (const uint8_t*) v79;
    const uint8_t v81 = v80[0];
    int v82 = (int) v81;
    int v83 = v82 & 15;
    float v84 = (float) v83;
    float v85 = v13 * v84;
    int v86 = v82 >> 4;
    float v87 = (float) v86;
    float v88 = v15 * v87;
    const uint8_t* v89 = v8 + 32;
    const uint8_t* v90 = (const uint8_t*) v89;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v91 = __riscv_vle8_v_u8m1(v90, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v92 = __riscv_vsrl_vx_u8m1(v91, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v93 = __riscv_vand_vx_u8m1(v92, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v94 = __riscv_vzext_vf4_u32m4(v93, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v95 = __riscv_vreinterpret_v_u32m4_i32m4(v94);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v96 = __riscv_vfcvt_f_x_v_f32m4(v95, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v97 = __riscv_vfmv_v_f_f32m4(v88, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v98 = __riscv_vfmsac_vf_f32m4(v97, v85, v96, 16);
    float* v99 = v11 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v99, v98, 16);
    const uint8_t* v100 = v8 + 4;
    const uint8_t* v101 = (const uint8_t*) v100;
    const uint8_t v102 = v101[0];
    int v103 = (int) v102;
    int v104 = v103 & 15;
    float v105 = (float) v104;
    float v106 = v13 * v105;
    int v107 = v103 >> 4;
    float v108 = (float) v107;
    float v109 = v15 * v108;
    const uint8_t* v110 = v8 + 16;
    const uint8_t* v111 = (const uint8_t*) v110;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v112 = __riscv_vle8_v_u8m1(v111, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v113 = __riscv_vsrl_vx_u8m1(v112, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v114 = __riscv_vand_vx_u8m1(v113, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v115 = __riscv_vzext_vf4_u32m4(v114, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v116 = __riscv_vreinterpret_v_u32m4_i32m4(v115);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v117 = __riscv_vfcvt_f_x_v_f32m4(v116, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v118 = __riscv_vfmv_v_f_f32m4(v109, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v119 = __riscv_vfmsac_vf_f32m4(v118, v106, v117, 16);
    float* v120 = v11 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v120, v119, 16);
    const uint8_t* v121 = v8 + 5;
    const uint8_t* v122 = (const uint8_t*) v121;
    const uint8_t v123 = v122[0];
    int v124 = (int) v123;
    int v125 = v124 & 15;
    float v126 = (float) v125;
    float v127 = v13 * v126;
    int v128 = v124 >> 4;
    float v129 = (float) v128;
    float v130 = v15 * v129;
    const uint8_t* v131 = v8 + 32;
    const uint8_t* v132 = (const uint8_t*) v131;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v133 = __riscv_vle8_v_u8m1(v132, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v134 = __riscv_vsrl_vx_u8m1(v133, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v135 = __riscv_vand_vx_u8m1(v134, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v136 = __riscv_vzext_vf4_u32m4(v135, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v137 = __riscv_vreinterpret_v_u32m4_i32m4(v136);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v138 = __riscv_vfcvt_f_x_v_f32m4(v137, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v139 = __riscv_vfmv_v_f_f32m4(v130, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v140 = __riscv_vfmsac_vf_f32m4(v139, v127, v138, 16);
    float* v141 = v11 + 80;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v141, v140, 16);
    const uint8_t* v142 = v8 + 6;
    const uint8_t* v143 = (const uint8_t*) v142;
    const uint8_t v144 = v143[0];
    int v145 = (int) v144;
    int v146 = v145 & 15;
    float v147 = (float) v146;
    float v148 = v13 * v147;
    int v149 = v145 >> 4;
    float v150 = (float) v149;
    float v151 = v15 * v150;
    const uint8_t* v152 = v8 + 16;
    const uint8_t* v153 = (const uint8_t*) v152;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v154 = __riscv_vle8_v_u8m1(v153, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v155 = __riscv_vsrl_vx_u8m1(v154, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v156 = __riscv_vand_vx_u8m1(v155, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v157 = __riscv_vzext_vf4_u32m4(v156, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v158 = __riscv_vreinterpret_v_u32m4_i32m4(v157);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v159 = __riscv_vfcvt_f_x_v_f32m4(v158, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v160 = __riscv_vfmv_v_f_f32m4(v151, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v161 = __riscv_vfmsac_vf_f32m4(v160, v148, v159, 16);
    float* v162 = v11 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v162, v161, 16);
    const uint8_t* v163 = v8 + 7;
    const uint8_t* v164 = (const uint8_t*) v163;
    const uint8_t v165 = v164[0];
    int v166 = (int) v165;
    int v167 = v166 & 15;
    float v168 = (float) v167;
    float v169 = v13 * v168;
    int v170 = v166 >> 4;
    float v171 = (float) v170;
    float v172 = v15 * v171;
    const uint8_t* v173 = v8 + 32;
    const uint8_t* v174 = (const uint8_t*) v173;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v175 = __riscv_vle8_v_u8m1(v174, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v176 = __riscv_vsrl_vx_u8m1(v175, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v177 = __riscv_vand_vx_u8m1(v176, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v178 = __riscv_vzext_vf4_u32m4(v177, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v179 = __riscv_vreinterpret_v_u32m4_i32m4(v178);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v180 = __riscv_vfcvt_f_x_v_f32m4(v179, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v181 = __riscv_vfmv_v_f_f32m4(v172, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v182 = __riscv_vfmsac_vf_f32m4(v181, v169, v180, 16);
    float* v183 = v11 + 112;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v183, v182, 16);
    const uint8_t* v184 = v8 + 8;
    const uint8_t* v185 = (const uint8_t*) v184;
    const uint8_t v186 = v185[0];
    int v187 = (int) v186;
    int v188 = v187 & 15;
    float v189 = (float) v188;
    float v190 = v13 * v189;
    int v191 = v187 >> 4;
    float v192 = (float) v191;
    float v193 = v15 * v192;
    const uint8_t* v194 = v8 + 48;
    const uint8_t* v195 = (const uint8_t*) v194;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v196 = __riscv_vle8_v_u8m1(v195, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v197 = __riscv_vsrl_vx_u8m1(v196, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v198 = __riscv_vand_vx_u8m1(v197, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v199 = __riscv_vzext_vf4_u32m4(v198, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v200 = __riscv_vreinterpret_v_u32m4_i32m4(v199);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v201 = __riscv_vfcvt_f_x_v_f32m4(v200, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v202 = __riscv_vfmv_v_f_f32m4(v193, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v203 = __riscv_vfmsac_vf_f32m4(v202, v190, v201, 16);
    float* v204 = v11 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v204, v203, 16);
    const uint8_t* v205 = v8 + 9;
    const uint8_t* v206 = (const uint8_t*) v205;
    const uint8_t v207 = v206[0];
    int v208 = (int) v207;
    int v209 = v208 & 15;
    float v210 = (float) v209;
    float v211 = v13 * v210;
    int v212 = v208 >> 4;
    float v213 = (float) v212;
    float v214 = v15 * v213;
    const uint8_t* v215 = v8 + 64;
    const uint8_t* v216 = (const uint8_t*) v215;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v217 = __riscv_vle8_v_u8m1(v216, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v218 = __riscv_vsrl_vx_u8m1(v217, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v219 = __riscv_vand_vx_u8m1(v218, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v220 = __riscv_vzext_vf4_u32m4(v219, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v221 = __riscv_vreinterpret_v_u32m4_i32m4(v220);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v222 = __riscv_vfcvt_f_x_v_f32m4(v221, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v223 = __riscv_vfmv_v_f_f32m4(v214, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v224 = __riscv_vfmsac_vf_f32m4(v223, v211, v222, 16);
    float* v225 = v11 + 144;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v225, v224, 16);
    const uint8_t* v226 = v8 + 10;
    const uint8_t* v227 = (const uint8_t*) v226;
    const uint8_t v228 = v227[0];
    int v229 = (int) v228;
    int v230 = v229 & 15;
    float v231 = (float) v230;
    float v232 = v13 * v231;
    int v233 = v229 >> 4;
    float v234 = (float) v233;
    float v235 = v15 * v234;
    const uint8_t* v236 = v8 + 48;
    const uint8_t* v237 = (const uint8_t*) v236;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v238 = __riscv_vle8_v_u8m1(v237, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v239 = __riscv_vsrl_vx_u8m1(v238, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v240 = __riscv_vand_vx_u8m1(v239, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v241 = __riscv_vzext_vf4_u32m4(v240, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v242 = __riscv_vreinterpret_v_u32m4_i32m4(v241);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v243 = __riscv_vfcvt_f_x_v_f32m4(v242, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v244 = __riscv_vfmv_v_f_f32m4(v235, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v245 = __riscv_vfmsac_vf_f32m4(v244, v232, v243, 16);
    float* v246 = v11 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v246, v245, 16);
    const uint8_t* v247 = v8 + 11;
    const uint8_t* v248 = (const uint8_t*) v247;
    const uint8_t v249 = v248[0];
    int v250 = (int) v249;
    int v251 = v250 & 15;
    float v252 = (float) v251;
    float v253 = v13 * v252;
    int v254 = v250 >> 4;
    float v255 = (float) v254;
    float v256 = v15 * v255;
    const uint8_t* v257 = v8 + 64;
    const uint8_t* v258 = (const uint8_t*) v257;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v259 = __riscv_vle8_v_u8m1(v258, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v260 = __riscv_vsrl_vx_u8m1(v259, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v261 = __riscv_vand_vx_u8m1(v260, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v262 = __riscv_vzext_vf4_u32m4(v261, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v263 = __riscv_vreinterpret_v_u32m4_i32m4(v262);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v264 = __riscv_vfcvt_f_x_v_f32m4(v263, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v265 = __riscv_vfmv_v_f_f32m4(v256, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v266 = __riscv_vfmsac_vf_f32m4(v265, v253, v264, 16);
    float* v267 = v11 + 176;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v267, v266, 16);
    const uint8_t* v268 = v8 + 12;
    const uint8_t* v269 = (const uint8_t*) v268;
    const uint8_t v270 = v269[0];
    int v271 = (int) v270;
    int v272 = v271 & 15;
    float v273 = (float) v272;
    float v274 = v13 * v273;
    int v275 = v271 >> 4;
    float v276 = (float) v275;
    float v277 = v15 * v276;
    const uint8_t* v278 = v8 + 48;
    const uint8_t* v279 = (const uint8_t*) v278;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v280 = __riscv_vle8_v_u8m1(v279, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v281 = __riscv_vsrl_vx_u8m1(v280, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v282 = __riscv_vand_vx_u8m1(v281, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v283 = __riscv_vzext_vf4_u32m4(v282, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v284 = __riscv_vreinterpret_v_u32m4_i32m4(v283);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v285 = __riscv_vfcvt_f_x_v_f32m4(v284, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v286 = __riscv_vfmv_v_f_f32m4(v277, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v287 = __riscv_vfmsac_vf_f32m4(v286, v274, v285, 16);
    float* v288 = v11 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v288, v287, 16);
    const uint8_t* v289 = v8 + 13;
    const uint8_t* v290 = (const uint8_t*) v289;
    const uint8_t v291 = v290[0];
    int v292 = (int) v291;
    int v293 = v292 & 15;
    float v294 = (float) v293;
    float v295 = v13 * v294;
    int v296 = v292 >> 4;
    float v297 = (float) v296;
    float v298 = v15 * v297;
    const uint8_t* v299 = v8 + 64;
    const uint8_t* v300 = (const uint8_t*) v299;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v301 = __riscv_vle8_v_u8m1(v300, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v302 = __riscv_vsrl_vx_u8m1(v301, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v303 = __riscv_vand_vx_u8m1(v302, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v304 = __riscv_vzext_vf4_u32m4(v303, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v305 = __riscv_vreinterpret_v_u32m4_i32m4(v304);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v306 = __riscv_vfcvt_f_x_v_f32m4(v305, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v307 = __riscv_vfmv_v_f_f32m4(v298, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v308 = __riscv_vfmsac_vf_f32m4(v307, v295, v306, 16);
    float* v309 = v11 + 208;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v309, v308, 16);
    const uint8_t* v310 = v8 + 14;
    const uint8_t* v311 = (const uint8_t*) v310;
    const uint8_t v312 = v311[0];
    int v313 = (int) v312;
    int v314 = v313 & 15;
    float v315 = (float) v314;
    float v316 = v13 * v315;
    int v317 = v313 >> 4;
    float v318 = (float) v317;
    float v319 = v15 * v318;
    const uint8_t* v320 = v8 + 48;
    const uint8_t* v321 = (const uint8_t*) v320;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v322 = __riscv_vle8_v_u8m1(v321, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v323 = __riscv_vsrl_vx_u8m1(v322, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v324 = __riscv_vand_vx_u8m1(v323, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v325 = __riscv_vzext_vf4_u32m4(v324, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v326 = __riscv_vreinterpret_v_u32m4_i32m4(v325);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v327 = __riscv_vfcvt_f_x_v_f32m4(v326, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v328 = __riscv_vfmv_v_f_f32m4(v319, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v329 = __riscv_vfmsac_vf_f32m4(v328, v316, v327, 16);
    float* v330 = v11 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v330, v329, 16);
    const uint8_t* v331 = v8 + 15;
    const uint8_t* v332 = (const uint8_t*) v331;
    const uint8_t v333 = v332[0];
    int v334 = (int) v333;
    int v335 = v334 & 15;
    float v336 = (float) v335;
    float v337 = v13 * v336;
    int v338 = v334 >> 4;
    float v339 = (float) v338;
    float v340 = v15 * v339;
    const uint8_t* v341 = v8 + 64;
    const uint8_t* v342 = (const uint8_t*) v341;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v343 = __riscv_vle8_v_u8m1(v342, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v344 = __riscv_vsrl_vx_u8m1(v343, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v345 = __riscv_vand_vx_u8m1(v344, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v346 = __riscv_vzext_vf4_u32m4(v345, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v347 = __riscv_vreinterpret_v_u32m4_i32m4(v346);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v348 = __riscv_vfcvt_f_x_v_f32m4(v347, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v349 = __riscv_vfmv_v_f_f32m4(v340, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmsac_vf_f32m4
    vfloat32m4_t v350 = __riscv_vfmsac_vf_f32m4(v349, v337, v348, 16);
    float* v351 = v11 + 240;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v351, v350, 16);
  }
  return;
}


