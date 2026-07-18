#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_iq4_xs_kernel_dequant_iq4_xs(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_dequant_iq4nl_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_table_load
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
  vint8m1_t v5 = __riscv_vle8_v_i8m1(weft_dequant_iq4nl_kvalues, 16);
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 256;
  for (size_t v7 = 0; v7 < v6; v7 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v8 = v7 * 136;
    const uint8_t* v9 = v2 + v8;
    size_t v10 = v7 * 256;
    float* v11 = v3 + v10;
    float* v12 = (float*) v11;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=iq4_xs_decode
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v9);
    const uint8_t* v14 = v9 + 3;
    const uint8_t* v15 = (const uint8_t*) v14;
    const uint8_t v16 = v15[0];
    int v17 = (int) v16;
    int v18 = v17 << 8;
    const uint8_t* v19 = v9 + 2;
    const uint8_t* v20 = (const uint8_t*) v19;
    const uint8_t v21 = v20[0];
    int v22 = (int) v21;
    int v23 = v22 | v18;
    const uint8_t* v24 = v9 + 4;
    const uint8_t* v25 = (const uint8_t*) v24;
    const uint8_t v26 = v25[0];
    int v27 = (int) v26;
    int v28 = v27 >> 0;
    int v29 = v28 & 15;
    int v30 = v23 >> 0;
    int v31 = v30 & 3;
    int v32 = v31 << 4;
    int v33 = v29 | v32;
    int v34 = v33 - 32;
    float v35 = (float) v34;
    float v36 = v13 * v35;
    const uint8_t* v37 = v9 + 8;
    const uint8_t* v38 = (const uint8_t*) v37;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v39 = __riscv_vle8_v_u8m1(v38, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v40 = __riscv_vand_vx_u8m1(v39, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v41 = __riscv_vsrl_vx_u8m1(v39, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v42 = __riscv_vrgather_vv_i8m1(v5, v40, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v43 = __riscv_vsext_vf4_i32m4(v42, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v44 = __riscv_vfcvt_f_x_v_f32m4(v43, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v45 = __riscv_vfmul_vf_f32m4(v44, v36, 16);
    float* v46 = v12 + 0;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v46, v45, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v47 = __riscv_vrgather_vv_i8m1(v5, v41, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v48 = __riscv_vsext_vf4_i32m4(v47, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v49 = __riscv_vfcvt_f_x_v_f32m4(v48, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v50 = __riscv_vfmul_vf_f32m4(v49, v36, 16);
    float* v51 = v12 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v51, v50, 16);
    const uint8_t* v52 = v9 + 4;
    const uint8_t* v53 = (const uint8_t*) v52;
    const uint8_t v54 = v53[0];
    int v55 = (int) v54;
    int v56 = v55 >> 4;
    int v57 = v56 & 15;
    int v58 = v23 >> 2;
    int v59 = v58 & 3;
    int v60 = v59 << 4;
    int v61 = v57 | v60;
    int v62 = v61 - 32;
    float v63 = (float) v62;
    float v64 = v13 * v63;
    const uint8_t* v65 = v9 + 24;
    const uint8_t* v66 = (const uint8_t*) v65;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v67 = __riscv_vle8_v_u8m1(v66, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v68 = __riscv_vand_vx_u8m1(v67, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v69 = __riscv_vsrl_vx_u8m1(v67, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v70 = __riscv_vrgather_vv_i8m1(v5, v68, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v71 = __riscv_vsext_vf4_i32m4(v70, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v72 = __riscv_vfcvt_f_x_v_f32m4(v71, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v73 = __riscv_vfmul_vf_f32m4(v72, v64, 16);
    float* v74 = v12 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v74, v73, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v75 = __riscv_vrgather_vv_i8m1(v5, v69, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v76 = __riscv_vsext_vf4_i32m4(v75, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v77 = __riscv_vfcvt_f_x_v_f32m4(v76, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v78 = __riscv_vfmul_vf_f32m4(v77, v64, 16);
    float* v79 = v12 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v79, v78, 16);
    const uint8_t* v80 = v9 + 5;
    const uint8_t* v81 = (const uint8_t*) v80;
    const uint8_t v82 = v81[0];
    int v83 = (int) v82;
    int v84 = v83 >> 0;
    int v85 = v84 & 15;
    int v86 = v23 >> 4;
    int v87 = v86 & 3;
    int v88 = v87 << 4;
    int v89 = v85 | v88;
    int v90 = v89 - 32;
    float v91 = (float) v90;
    float v92 = v13 * v91;
    const uint8_t* v93 = v9 + 40;
    const uint8_t* v94 = (const uint8_t*) v93;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v95 = __riscv_vle8_v_u8m1(v94, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v96 = __riscv_vand_vx_u8m1(v95, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v97 = __riscv_vsrl_vx_u8m1(v95, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v98 = __riscv_vrgather_vv_i8m1(v5, v96, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v99 = __riscv_vsext_vf4_i32m4(v98, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v100 = __riscv_vfcvt_f_x_v_f32m4(v99, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v101 = __riscv_vfmul_vf_f32m4(v100, v92, 16);
    float* v102 = v12 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v102, v101, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v103 = __riscv_vrgather_vv_i8m1(v5, v97, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v104 = __riscv_vsext_vf4_i32m4(v103, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v105 = __riscv_vfcvt_f_x_v_f32m4(v104, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v106 = __riscv_vfmul_vf_f32m4(v105, v92, 16);
    float* v107 = v12 + 80;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v107, v106, 16);
    const uint8_t* v108 = v9 + 5;
    const uint8_t* v109 = (const uint8_t*) v108;
    const uint8_t v110 = v109[0];
    int v111 = (int) v110;
    int v112 = v111 >> 4;
    int v113 = v112 & 15;
    int v114 = v23 >> 6;
    int v115 = v114 & 3;
    int v116 = v115 << 4;
    int v117 = v113 | v116;
    int v118 = v117 - 32;
    float v119 = (float) v118;
    float v120 = v13 * v119;
    const uint8_t* v121 = v9 + 56;
    const uint8_t* v122 = (const uint8_t*) v121;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v123 = __riscv_vle8_v_u8m1(v122, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v124 = __riscv_vand_vx_u8m1(v123, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v125 = __riscv_vsrl_vx_u8m1(v123, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v126 = __riscv_vrgather_vv_i8m1(v5, v124, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v127 = __riscv_vsext_vf4_i32m4(v126, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v128 = __riscv_vfcvt_f_x_v_f32m4(v127, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v129 = __riscv_vfmul_vf_f32m4(v128, v120, 16);
    float* v130 = v12 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v130, v129, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v131 = __riscv_vrgather_vv_i8m1(v5, v125, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v132 = __riscv_vsext_vf4_i32m4(v131, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v133 = __riscv_vfcvt_f_x_v_f32m4(v132, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v134 = __riscv_vfmul_vf_f32m4(v133, v120, 16);
    float* v135 = v12 + 112;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v135, v134, 16);
    const uint8_t* v136 = v9 + 6;
    const uint8_t* v137 = (const uint8_t*) v136;
    const uint8_t v138 = v137[0];
    int v139 = (int) v138;
    int v140 = v139 >> 0;
    int v141 = v140 & 15;
    int v142 = v23 >> 8;
    int v143 = v142 & 3;
    int v144 = v143 << 4;
    int v145 = v141 | v144;
    int v146 = v145 - 32;
    float v147 = (float) v146;
    float v148 = v13 * v147;
    const uint8_t* v149 = v9 + 72;
    const uint8_t* v150 = (const uint8_t*) v149;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v151 = __riscv_vle8_v_u8m1(v150, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v152 = __riscv_vand_vx_u8m1(v151, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v153 = __riscv_vsrl_vx_u8m1(v151, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v154 = __riscv_vrgather_vv_i8m1(v5, v152, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v155 = __riscv_vsext_vf4_i32m4(v154, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v156 = __riscv_vfcvt_f_x_v_f32m4(v155, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v157 = __riscv_vfmul_vf_f32m4(v156, v148, 16);
    float* v158 = v12 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v158, v157, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v159 = __riscv_vrgather_vv_i8m1(v5, v153, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v160 = __riscv_vsext_vf4_i32m4(v159, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v161 = __riscv_vfcvt_f_x_v_f32m4(v160, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v162 = __riscv_vfmul_vf_f32m4(v161, v148, 16);
    float* v163 = v12 + 144;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v163, v162, 16);
    const uint8_t* v164 = v9 + 6;
    const uint8_t* v165 = (const uint8_t*) v164;
    const uint8_t v166 = v165[0];
    int v167 = (int) v166;
    int v168 = v167 >> 4;
    int v169 = v168 & 15;
    int v170 = v23 >> 10;
    int v171 = v170 & 3;
    int v172 = v171 << 4;
    int v173 = v169 | v172;
    int v174 = v173 - 32;
    float v175 = (float) v174;
    float v176 = v13 * v175;
    const uint8_t* v177 = v9 + 88;
    const uint8_t* v178 = (const uint8_t*) v177;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v179 = __riscv_vle8_v_u8m1(v178, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v180 = __riscv_vand_vx_u8m1(v179, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v181 = __riscv_vsrl_vx_u8m1(v179, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v182 = __riscv_vrgather_vv_i8m1(v5, v180, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v183 = __riscv_vsext_vf4_i32m4(v182, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v184 = __riscv_vfcvt_f_x_v_f32m4(v183, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v185 = __riscv_vfmul_vf_f32m4(v184, v176, 16);
    float* v186 = v12 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v186, v185, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v187 = __riscv_vrgather_vv_i8m1(v5, v181, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v188 = __riscv_vsext_vf4_i32m4(v187, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v189 = __riscv_vfcvt_f_x_v_f32m4(v188, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v190 = __riscv_vfmul_vf_f32m4(v189, v176, 16);
    float* v191 = v12 + 176;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v191, v190, 16);
    const uint8_t* v192 = v9 + 7;
    const uint8_t* v193 = (const uint8_t*) v192;
    const uint8_t v194 = v193[0];
    int v195 = (int) v194;
    int v196 = v195 >> 0;
    int v197 = v196 & 15;
    int v198 = v23 >> 12;
    int v199 = v198 & 3;
    int v200 = v199 << 4;
    int v201 = v197 | v200;
    int v202 = v201 - 32;
    float v203 = (float) v202;
    float v204 = v13 * v203;
    const uint8_t* v205 = v9 + 104;
    const uint8_t* v206 = (const uint8_t*) v205;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v207 = __riscv_vle8_v_u8m1(v206, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v208 = __riscv_vand_vx_u8m1(v207, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v209 = __riscv_vsrl_vx_u8m1(v207, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v210 = __riscv_vrgather_vv_i8m1(v5, v208, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v211 = __riscv_vsext_vf4_i32m4(v210, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v212 = __riscv_vfcvt_f_x_v_f32m4(v211, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v213 = __riscv_vfmul_vf_f32m4(v212, v204, 16);
    float* v214 = v12 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v214, v213, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v215 = __riscv_vrgather_vv_i8m1(v5, v209, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v216 = __riscv_vsext_vf4_i32m4(v215, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v217 = __riscv_vfcvt_f_x_v_f32m4(v216, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v218 = __riscv_vfmul_vf_f32m4(v217, v204, 16);
    float* v219 = v12 + 208;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v219, v218, 16);
    const uint8_t* v220 = v9 + 7;
    const uint8_t* v221 = (const uint8_t*) v220;
    const uint8_t v222 = v221[0];
    int v223 = (int) v222;
    int v224 = v223 >> 4;
    int v225 = v224 & 15;
    int v226 = v23 >> 14;
    int v227 = v226 & 3;
    int v228 = v227 << 4;
    int v229 = v225 | v228;
    int v230 = v229 - 32;
    float v231 = (float) v230;
    float v232 = v13 * v231;
    const uint8_t* v233 = v9 + 120;
    const uint8_t* v234 = (const uint8_t*) v233;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v235 = __riscv_vle8_v_u8m1(v234, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v236 = __riscv_vand_vx_u8m1(v235, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v237 = __riscv_vsrl_vx_u8m1(v235, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v238 = __riscv_vrgather_vv_i8m1(v5, v236, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v239 = __riscv_vsext_vf4_i32m4(v238, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v240 = __riscv_vfcvt_f_x_v_f32m4(v239, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v241 = __riscv_vfmul_vf_f32m4(v240, v232, 16);
    float* v242 = v12 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v242, v241, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v243 = __riscv_vrgather_vv_i8m1(v5, v237, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v244 = __riscv_vsext_vf4_i32m4(v243, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v245 = __riscv_vfcvt_f_x_v_f32m4(v244, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v246 = __riscv_vfmul_vf_f32m4(v245, v232, 16);
    float* v247 = v12 + 240;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v247, v246, 16);
  }
  return;
}


