#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
#include <math.h>
extern "C" void weft_emitc_dequant_nvfp4_kernel_dequant_nvfp4(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_dequant_mxfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_table_load
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
  vint8m1_t v5 = __riscv_vle8_v_i8m1(weft_dequant_mxfp4_kvalues, 16);
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 64;
  for (size_t v7 = 0; v7 < v6; v7 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v8 = v7 * 36;
    const uint8_t* v9 = v2 + v8;
    size_t v10 = v7 * 64;
    float* v11 = v3 + v10;
    float* v12 = (float*) v11;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=nvfp4_sub_decode
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_scale
    const uint8_t* v13 = v9 + 0;
    const uint8_t* v14 = (const uint8_t*) v13;
    const uint8_t v15 = v14[0];
    uint32_t v16 = (uint32_t) v15;
    uint32_t v17 = v16 >> 3u;
    uint32_t v18 = v17 & 15u;
    uint32_t v19 = v16 & 7u;
    int v20 = (int) v18;
    int v21 = (int) v19;
    float v22 = (float) v21;
    float v23 = ldexpf(v22, -9);
    float v24 = v22 / 8.0f;
    float v25 = 1.0f + v24;
    int v26 = v20 - 7;
    float v27 = ldexpf(v25, v26);
    bool v28 = v18 == 0u;
    float v29 = v28 ? v23 : v27;
    float v30 = v29 * 0.5f;
    bool v31 = v16 == 0u;
    bool v32 = v16 == 127u;
    bool v33 = v31 || v32;
    float v34 = v33 ? 0.0f : v30;
    const uint8_t* v35 = v9 + 4;
    const uint8_t* v36 = (const uint8_t*) v35;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v37 = __riscv_vle8_v_u8m1(v36, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v38 = __riscv_vand_vx_u8m1(v37, 15, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v39 = __riscv_vsrl_vx_u8m1(v37, 4, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v40 = __riscv_vrgather_vv_i8m1(v5, v38, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v41 = __riscv_vsext_vf4_i32m4(v40, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v42 = __riscv_vfcvt_f_x_v_f32m4(v41, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v43 = __riscv_vfmul_vf_f32m4(v42, v34, 8);
    float* v44 = v12 + 0;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v44, v43, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v45 = __riscv_vrgather_vv_i8m1(v5, v39, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v46 = __riscv_vsext_vf4_i32m4(v45, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v47 = __riscv_vfcvt_f_x_v_f32m4(v46, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v48 = __riscv_vfmul_vf_f32m4(v47, v34, 8);
    float* v49 = v12 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v49, v48, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_scale
    const uint8_t* v50 = v9 + 1;
    const uint8_t* v51 = (const uint8_t*) v50;
    const uint8_t v52 = v51[0];
    uint32_t v53 = (uint32_t) v52;
    uint32_t v54 = v53 >> 3u;
    uint32_t v55 = v54 & 15u;
    uint32_t v56 = v53 & 7u;
    int v57 = (int) v55;
    int v58 = (int) v56;
    float v59 = (float) v58;
    float v60 = ldexpf(v59, -9);
    float v61 = v59 / 8.0f;
    float v62 = 1.0f + v61;
    int v63 = v57 - 7;
    float v64 = ldexpf(v62, v63);
    bool v65 = v55 == 0u;
    float v66 = v65 ? v60 : v64;
    float v67 = v66 * 0.5f;
    bool v68 = v53 == 0u;
    bool v69 = v53 == 127u;
    bool v70 = v68 || v69;
    float v71 = v70 ? 0.0f : v67;
    const uint8_t* v72 = v9 + 12;
    const uint8_t* v73 = (const uint8_t*) v72;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v74 = __riscv_vle8_v_u8m1(v73, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v75 = __riscv_vand_vx_u8m1(v74, 15, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v76 = __riscv_vsrl_vx_u8m1(v74, 4, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v77 = __riscv_vrgather_vv_i8m1(v5, v75, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v78 = __riscv_vsext_vf4_i32m4(v77, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v79 = __riscv_vfcvt_f_x_v_f32m4(v78, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v80 = __riscv_vfmul_vf_f32m4(v79, v71, 8);
    float* v81 = v12 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v81, v80, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v82 = __riscv_vrgather_vv_i8m1(v5, v76, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v83 = __riscv_vsext_vf4_i32m4(v82, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v84 = __riscv_vfcvt_f_x_v_f32m4(v83, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v85 = __riscv_vfmul_vf_f32m4(v84, v71, 8);
    float* v86 = v12 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v86, v85, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_scale
    const uint8_t* v87 = v9 + 2;
    const uint8_t* v88 = (const uint8_t*) v87;
    const uint8_t v89 = v88[0];
    uint32_t v90 = (uint32_t) v89;
    uint32_t v91 = v90 >> 3u;
    uint32_t v92 = v91 & 15u;
    uint32_t v93 = v90 & 7u;
    int v94 = (int) v92;
    int v95 = (int) v93;
    float v96 = (float) v95;
    float v97 = ldexpf(v96, -9);
    float v98 = v96 / 8.0f;
    float v99 = 1.0f + v98;
    int v100 = v94 - 7;
    float v101 = ldexpf(v99, v100);
    bool v102 = v92 == 0u;
    float v103 = v102 ? v97 : v101;
    float v104 = v103 * 0.5f;
    bool v105 = v90 == 0u;
    bool v106 = v90 == 127u;
    bool v107 = v105 || v106;
    float v108 = v107 ? 0.0f : v104;
    const uint8_t* v109 = v9 + 20;
    const uint8_t* v110 = (const uint8_t*) v109;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v111 = __riscv_vle8_v_u8m1(v110, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v112 = __riscv_vand_vx_u8m1(v111, 15, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v113 = __riscv_vsrl_vx_u8m1(v111, 4, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v114 = __riscv_vrgather_vv_i8m1(v5, v112, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v115 = __riscv_vsext_vf4_i32m4(v114, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v116 = __riscv_vfcvt_f_x_v_f32m4(v115, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v117 = __riscv_vfmul_vf_f32m4(v116, v108, 8);
    float* v118 = v12 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v118, v117, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v119 = __riscv_vrgather_vv_i8m1(v5, v113, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v120 = __riscv_vsext_vf4_i32m4(v119, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v121 = __riscv_vfcvt_f_x_v_f32m4(v120, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v122 = __riscv_vfmul_vf_f32m4(v121, v108, 8);
    float* v123 = v12 + 40;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v123, v122, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_scale
    const uint8_t* v124 = v9 + 3;
    const uint8_t* v125 = (const uint8_t*) v124;
    const uint8_t v126 = v125[0];
    uint32_t v127 = (uint32_t) v126;
    uint32_t v128 = v127 >> 3u;
    uint32_t v129 = v128 & 15u;
    uint32_t v130 = v127 & 7u;
    int v131 = (int) v129;
    int v132 = (int) v130;
    float v133 = (float) v132;
    float v134 = ldexpf(v133, -9);
    float v135 = v133 / 8.0f;
    float v136 = 1.0f + v135;
    int v137 = v131 - 7;
    float v138 = ldexpf(v136, v137);
    bool v139 = v129 == 0u;
    float v140 = v139 ? v134 : v138;
    float v141 = v140 * 0.5f;
    bool v142 = v127 == 0u;
    bool v143 = v127 == 127u;
    bool v144 = v142 || v143;
    float v145 = v144 ? 0.0f : v141;
    const uint8_t* v146 = v9 + 28;
    const uint8_t* v147 = (const uint8_t*) v146;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v148 = __riscv_vle8_v_u8m1(v147, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v149 = __riscv_vand_vx_u8m1(v148, 15, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v150 = __riscv_vsrl_vx_u8m1(v148, 4, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v151 = __riscv_vrgather_vv_i8m1(v5, v149, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v152 = __riscv_vsext_vf4_i32m4(v151, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v153 = __riscv_vfcvt_f_x_v_f32m4(v152, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v154 = __riscv_vfmul_vf_f32m4(v153, v145, 8);
    float* v155 = v12 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v155, v154, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v156 = __riscv_vrgather_vv_i8m1(v5, v150, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v157 = __riscv_vsext_vf4_i32m4(v156, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v158 = __riscv_vfcvt_f_x_v_f32m4(v157, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v159 = __riscv_vfmul_vf_f32m4(v158, v145, 8);
    float* v160 = v12 + 56;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v160, v159, 8);
  }
  return;
}


