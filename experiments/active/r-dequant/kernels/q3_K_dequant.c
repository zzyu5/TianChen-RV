#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_q3_K_kernel_dequant_q3_K(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v7 = v6 * 110;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 256;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=q3_K_decode
    const uint8_t* v12 = v8 + 108;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v12);
    const uint8_t* v14 = v8 + 96;
    const uint8_t* v15 = (const uint8_t*) v14;
    const uint8_t v16 = v15[0];
    int v17 = (int) v16;
    int v18 = v17 & 15;
    const uint8_t* v19 = v8 + 104;
    const uint8_t* v20 = (const uint8_t*) v19;
    const uint8_t v21 = v20[0];
    int v22 = (int) v21;
    int v23 = v22 & 3;
    int v24 = v23 << 4;
    int v25 = v18 | v24;
    int v26 = v25 - 32;
    float v27 = (float) v26;
    float v28 = v13 * v27;
    const uint8_t* v29 = v8 + 32;
    const uint8_t* v30 = (const uint8_t*) v29;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v31 = __riscv_vle8_v_u8m1(v30, 16);
    const uint8_t* v32 = v8 + 0;
    const uint8_t* v33 = (const uint8_t*) v32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v34 = __riscv_vle8_v_u8m1(v33, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v35 = __riscv_vsrl_vx_u8m1(v31, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v36 = __riscv_vand_vx_u8m1(v35, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v37 = __riscv_vsrl_vx_u8m1(v34, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v38 = __riscv_vand_vx_u8m1(v37, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v39 = __riscv_vzext_vf4_u32m4(v36, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v40 = __riscv_vreinterpret_v_u32m4_i32m4(v39);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v41 = __riscv_vzext_vf4_u32m4(v38, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v42 = __riscv_vreinterpret_v_u32m4_i32m4(v41);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v43 = __riscv_vrsub_vx_i32m4(v42, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v44 = __riscv_vsll_vx_i32m4(v43, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v45 = __riscv_vsub_vv_i32m4(v40, v44, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v46 = __riscv_vfcvt_f_x_v_f32m4(v45, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v47 = __riscv_vfmul_vf_f32m4(v46, v28, 16);
    float* v48 = v11 + 0;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v48, v47, 16);
    const uint8_t* v49 = v8 + 97;
    const uint8_t* v50 = (const uint8_t*) v49;
    const uint8_t v51 = v50[0];
    int v52 = (int) v51;
    int v53 = v52 & 15;
    const uint8_t* v54 = v8 + 105;
    const uint8_t* v55 = (const uint8_t*) v54;
    const uint8_t v56 = v55[0];
    int v57 = (int) v56;
    int v58 = v57 & 3;
    int v59 = v58 << 4;
    int v60 = v53 | v59;
    int v61 = v60 - 32;
    float v62 = (float) v61;
    float v63 = v13 * v62;
    const uint8_t* v64 = v8 + 48;
    const uint8_t* v65 = (const uint8_t*) v64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v66 = __riscv_vle8_v_u8m1(v65, 16);
    const uint8_t* v67 = v8 + 16;
    const uint8_t* v68 = (const uint8_t*) v67;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v69 = __riscv_vle8_v_u8m1(v68, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v70 = __riscv_vsrl_vx_u8m1(v66, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v71 = __riscv_vand_vx_u8m1(v70, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v72 = __riscv_vsrl_vx_u8m1(v69, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v73 = __riscv_vand_vx_u8m1(v72, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v74 = __riscv_vzext_vf4_u32m4(v71, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v75 = __riscv_vreinterpret_v_u32m4_i32m4(v74);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v76 = __riscv_vzext_vf4_u32m4(v73, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v77 = __riscv_vreinterpret_v_u32m4_i32m4(v76);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v78 = __riscv_vrsub_vx_i32m4(v77, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v79 = __riscv_vsll_vx_i32m4(v78, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v80 = __riscv_vsub_vv_i32m4(v75, v79, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v81 = __riscv_vfcvt_f_x_v_f32m4(v80, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v82 = __riscv_vfmul_vf_f32m4(v81, v63, 16);
    float* v83 = v11 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v83, v82, 16);
    const uint8_t* v84 = v8 + 98;
    const uint8_t* v85 = (const uint8_t*) v84;
    const uint8_t v86 = v85[0];
    int v87 = (int) v86;
    int v88 = v87 & 15;
    const uint8_t* v89 = v8 + 106;
    const uint8_t* v90 = (const uint8_t*) v89;
    const uint8_t v91 = v90[0];
    int v92 = (int) v91;
    int v93 = v92 & 3;
    int v94 = v93 << 4;
    int v95 = v88 | v94;
    int v96 = v95 - 32;
    float v97 = (float) v96;
    float v98 = v13 * v97;
    const uint8_t* v99 = v8 + 32;
    const uint8_t* v100 = (const uint8_t*) v99;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v101 = __riscv_vle8_v_u8m1(v100, 16);
    const uint8_t* v102 = v8 + 0;
    const uint8_t* v103 = (const uint8_t*) v102;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v104 = __riscv_vle8_v_u8m1(v103, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v105 = __riscv_vsrl_vx_u8m1(v101, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v106 = __riscv_vand_vx_u8m1(v105, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v107 = __riscv_vsrl_vx_u8m1(v104, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v108 = __riscv_vand_vx_u8m1(v107, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v109 = __riscv_vzext_vf4_u32m4(v106, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v110 = __riscv_vreinterpret_v_u32m4_i32m4(v109);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v111 = __riscv_vzext_vf4_u32m4(v108, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v112 = __riscv_vreinterpret_v_u32m4_i32m4(v111);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v113 = __riscv_vrsub_vx_i32m4(v112, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v114 = __riscv_vsll_vx_i32m4(v113, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v115 = __riscv_vsub_vv_i32m4(v110, v114, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v116 = __riscv_vfcvt_f_x_v_f32m4(v115, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v117 = __riscv_vfmul_vf_f32m4(v116, v98, 16);
    float* v118 = v11 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v118, v117, 16);
    const uint8_t* v119 = v8 + 99;
    const uint8_t* v120 = (const uint8_t*) v119;
    const uint8_t v121 = v120[0];
    int v122 = (int) v121;
    int v123 = v122 & 15;
    const uint8_t* v124 = v8 + 107;
    const uint8_t* v125 = (const uint8_t*) v124;
    const uint8_t v126 = v125[0];
    int v127 = (int) v126;
    int v128 = v127 & 3;
    int v129 = v128 << 4;
    int v130 = v123 | v129;
    int v131 = v130 - 32;
    float v132 = (float) v131;
    float v133 = v13 * v132;
    const uint8_t* v134 = v8 + 48;
    const uint8_t* v135 = (const uint8_t*) v134;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v136 = __riscv_vle8_v_u8m1(v135, 16);
    const uint8_t* v137 = v8 + 16;
    const uint8_t* v138 = (const uint8_t*) v137;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v139 = __riscv_vle8_v_u8m1(v138, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v140 = __riscv_vsrl_vx_u8m1(v136, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v141 = __riscv_vand_vx_u8m1(v140, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v142 = __riscv_vsrl_vx_u8m1(v139, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v143 = __riscv_vand_vx_u8m1(v142, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v144 = __riscv_vzext_vf4_u32m4(v141, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v145 = __riscv_vreinterpret_v_u32m4_i32m4(v144);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v146 = __riscv_vzext_vf4_u32m4(v143, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v147 = __riscv_vreinterpret_v_u32m4_i32m4(v146);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v148 = __riscv_vrsub_vx_i32m4(v147, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v149 = __riscv_vsll_vx_i32m4(v148, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v150 = __riscv_vsub_vv_i32m4(v145, v149, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v151 = __riscv_vfcvt_f_x_v_f32m4(v150, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v152 = __riscv_vfmul_vf_f32m4(v151, v133, 16);
    float* v153 = v11 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v153, v152, 16);
    const uint8_t* v154 = v8 + 100;
    const uint8_t* v155 = (const uint8_t*) v154;
    const uint8_t v156 = v155[0];
    int v157 = (int) v156;
    int v158 = v157 & 15;
    const uint8_t* v159 = v8 + 104;
    const uint8_t* v160 = (const uint8_t*) v159;
    const uint8_t v161 = v160[0];
    int v162 = (int) v161;
    int v163 = v162 >> 2;
    int v164 = v163 & 3;
    int v165 = v164 << 4;
    int v166 = v158 | v165;
    int v167 = v166 - 32;
    float v168 = (float) v167;
    float v169 = v13 * v168;
    const uint8_t* v170 = v8 + 32;
    const uint8_t* v171 = (const uint8_t*) v170;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v172 = __riscv_vle8_v_u8m1(v171, 16);
    const uint8_t* v173 = v8 + 0;
    const uint8_t* v174 = (const uint8_t*) v173;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v175 = __riscv_vle8_v_u8m1(v174, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v176 = __riscv_vsrl_vx_u8m1(v172, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v177 = __riscv_vand_vx_u8m1(v176, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v178 = __riscv_vsrl_vx_u8m1(v175, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v179 = __riscv_vand_vx_u8m1(v178, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v180 = __riscv_vzext_vf4_u32m4(v177, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v181 = __riscv_vreinterpret_v_u32m4_i32m4(v180);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v182 = __riscv_vzext_vf4_u32m4(v179, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v183 = __riscv_vreinterpret_v_u32m4_i32m4(v182);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v184 = __riscv_vrsub_vx_i32m4(v183, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v185 = __riscv_vsll_vx_i32m4(v184, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v186 = __riscv_vsub_vv_i32m4(v181, v185, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v187 = __riscv_vfcvt_f_x_v_f32m4(v186, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v188 = __riscv_vfmul_vf_f32m4(v187, v169, 16);
    float* v189 = v11 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v189, v188, 16);
    const uint8_t* v190 = v8 + 101;
    const uint8_t* v191 = (const uint8_t*) v190;
    const uint8_t v192 = v191[0];
    int v193 = (int) v192;
    int v194 = v193 & 15;
    const uint8_t* v195 = v8 + 105;
    const uint8_t* v196 = (const uint8_t*) v195;
    const uint8_t v197 = v196[0];
    int v198 = (int) v197;
    int v199 = v198 >> 2;
    int v200 = v199 & 3;
    int v201 = v200 << 4;
    int v202 = v194 | v201;
    int v203 = v202 - 32;
    float v204 = (float) v203;
    float v205 = v13 * v204;
    const uint8_t* v206 = v8 + 48;
    const uint8_t* v207 = (const uint8_t*) v206;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v208 = __riscv_vle8_v_u8m1(v207, 16);
    const uint8_t* v209 = v8 + 16;
    const uint8_t* v210 = (const uint8_t*) v209;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v211 = __riscv_vle8_v_u8m1(v210, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v212 = __riscv_vsrl_vx_u8m1(v208, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v213 = __riscv_vand_vx_u8m1(v212, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v214 = __riscv_vsrl_vx_u8m1(v211, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v215 = __riscv_vand_vx_u8m1(v214, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v216 = __riscv_vzext_vf4_u32m4(v213, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v217 = __riscv_vreinterpret_v_u32m4_i32m4(v216);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v218 = __riscv_vzext_vf4_u32m4(v215, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v219 = __riscv_vreinterpret_v_u32m4_i32m4(v218);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v220 = __riscv_vrsub_vx_i32m4(v219, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v221 = __riscv_vsll_vx_i32m4(v220, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v222 = __riscv_vsub_vv_i32m4(v217, v221, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v223 = __riscv_vfcvt_f_x_v_f32m4(v222, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v224 = __riscv_vfmul_vf_f32m4(v223, v205, 16);
    float* v225 = v11 + 80;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v225, v224, 16);
    const uint8_t* v226 = v8 + 102;
    const uint8_t* v227 = (const uint8_t*) v226;
    const uint8_t v228 = v227[0];
    int v229 = (int) v228;
    int v230 = v229 & 15;
    const uint8_t* v231 = v8 + 106;
    const uint8_t* v232 = (const uint8_t*) v231;
    const uint8_t v233 = v232[0];
    int v234 = (int) v233;
    int v235 = v234 >> 2;
    int v236 = v235 & 3;
    int v237 = v236 << 4;
    int v238 = v230 | v237;
    int v239 = v238 - 32;
    float v240 = (float) v239;
    float v241 = v13 * v240;
    const uint8_t* v242 = v8 + 32;
    const uint8_t* v243 = (const uint8_t*) v242;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v244 = __riscv_vle8_v_u8m1(v243, 16);
    const uint8_t* v245 = v8 + 0;
    const uint8_t* v246 = (const uint8_t*) v245;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v247 = __riscv_vle8_v_u8m1(v246, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v248 = __riscv_vsrl_vx_u8m1(v244, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v249 = __riscv_vand_vx_u8m1(v248, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v250 = __riscv_vsrl_vx_u8m1(v247, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v251 = __riscv_vand_vx_u8m1(v250, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v252 = __riscv_vzext_vf4_u32m4(v249, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v253 = __riscv_vreinterpret_v_u32m4_i32m4(v252);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v254 = __riscv_vzext_vf4_u32m4(v251, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v255 = __riscv_vreinterpret_v_u32m4_i32m4(v254);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v256 = __riscv_vrsub_vx_i32m4(v255, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v257 = __riscv_vsll_vx_i32m4(v256, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v258 = __riscv_vsub_vv_i32m4(v253, v257, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v259 = __riscv_vfcvt_f_x_v_f32m4(v258, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v260 = __riscv_vfmul_vf_f32m4(v259, v241, 16);
    float* v261 = v11 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v261, v260, 16);
    const uint8_t* v262 = v8 + 103;
    const uint8_t* v263 = (const uint8_t*) v262;
    const uint8_t v264 = v263[0];
    int v265 = (int) v264;
    int v266 = v265 & 15;
    const uint8_t* v267 = v8 + 107;
    const uint8_t* v268 = (const uint8_t*) v267;
    const uint8_t v269 = v268[0];
    int v270 = (int) v269;
    int v271 = v270 >> 2;
    int v272 = v271 & 3;
    int v273 = v272 << 4;
    int v274 = v266 | v273;
    int v275 = v274 - 32;
    float v276 = (float) v275;
    float v277 = v13 * v276;
    const uint8_t* v278 = v8 + 48;
    const uint8_t* v279 = (const uint8_t*) v278;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v280 = __riscv_vle8_v_u8m1(v279, 16);
    const uint8_t* v281 = v8 + 16;
    const uint8_t* v282 = (const uint8_t*) v281;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v283 = __riscv_vle8_v_u8m1(v282, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v284 = __riscv_vsrl_vx_u8m1(v280, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v285 = __riscv_vand_vx_u8m1(v284, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v286 = __riscv_vsrl_vx_u8m1(v283, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v287 = __riscv_vand_vx_u8m1(v286, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v288 = __riscv_vzext_vf4_u32m4(v285, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v289 = __riscv_vreinterpret_v_u32m4_i32m4(v288);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v290 = __riscv_vzext_vf4_u32m4(v287, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v291 = __riscv_vreinterpret_v_u32m4_i32m4(v290);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v292 = __riscv_vrsub_vx_i32m4(v291, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v293 = __riscv_vsll_vx_i32m4(v292, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v294 = __riscv_vsub_vv_i32m4(v289, v293, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v295 = __riscv_vfcvt_f_x_v_f32m4(v294, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v296 = __riscv_vfmul_vf_f32m4(v295, v277, 16);
    float* v297 = v11 + 112;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v297, v296, 16);
    const uint8_t* v298 = v8 + 96;
    const uint8_t* v299 = (const uint8_t*) v298;
    const uint8_t v300 = v299[0];
    int v301 = (int) v300;
    int v302 = v301 >> 4;
    int v303 = v302 & 15;
    const uint8_t* v304 = v8 + 104;
    const uint8_t* v305 = (const uint8_t*) v304;
    const uint8_t v306 = v305[0];
    int v307 = (int) v306;
    int v308 = v307 >> 4;
    int v309 = v308 & 3;
    int v310 = v309 << 4;
    int v311 = v303 | v310;
    int v312 = v311 - 32;
    float v313 = (float) v312;
    float v314 = v13 * v313;
    const uint8_t* v315 = v8 + 64;
    const uint8_t* v316 = (const uint8_t*) v315;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v317 = __riscv_vle8_v_u8m1(v316, 16);
    const uint8_t* v318 = v8 + 0;
    const uint8_t* v319 = (const uint8_t*) v318;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v320 = __riscv_vle8_v_u8m1(v319, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v321 = __riscv_vsrl_vx_u8m1(v317, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v322 = __riscv_vand_vx_u8m1(v321, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v323 = __riscv_vsrl_vx_u8m1(v320, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v324 = __riscv_vand_vx_u8m1(v323, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v325 = __riscv_vzext_vf4_u32m4(v322, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v326 = __riscv_vreinterpret_v_u32m4_i32m4(v325);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v327 = __riscv_vzext_vf4_u32m4(v324, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v328 = __riscv_vreinterpret_v_u32m4_i32m4(v327);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v329 = __riscv_vrsub_vx_i32m4(v328, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v330 = __riscv_vsll_vx_i32m4(v329, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v331 = __riscv_vsub_vv_i32m4(v326, v330, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v332 = __riscv_vfcvt_f_x_v_f32m4(v331, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v333 = __riscv_vfmul_vf_f32m4(v332, v314, 16);
    float* v334 = v11 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v334, v333, 16);
    const uint8_t* v335 = v8 + 97;
    const uint8_t* v336 = (const uint8_t*) v335;
    const uint8_t v337 = v336[0];
    int v338 = (int) v337;
    int v339 = v338 >> 4;
    int v340 = v339 & 15;
    const uint8_t* v341 = v8 + 105;
    const uint8_t* v342 = (const uint8_t*) v341;
    const uint8_t v343 = v342[0];
    int v344 = (int) v343;
    int v345 = v344 >> 4;
    int v346 = v345 & 3;
    int v347 = v346 << 4;
    int v348 = v340 | v347;
    int v349 = v348 - 32;
    float v350 = (float) v349;
    float v351 = v13 * v350;
    const uint8_t* v352 = v8 + 80;
    const uint8_t* v353 = (const uint8_t*) v352;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v354 = __riscv_vle8_v_u8m1(v353, 16);
    const uint8_t* v355 = v8 + 16;
    const uint8_t* v356 = (const uint8_t*) v355;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v357 = __riscv_vle8_v_u8m1(v356, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v358 = __riscv_vsrl_vx_u8m1(v354, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v359 = __riscv_vand_vx_u8m1(v358, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v360 = __riscv_vsrl_vx_u8m1(v357, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v361 = __riscv_vand_vx_u8m1(v360, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v362 = __riscv_vzext_vf4_u32m4(v359, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v363 = __riscv_vreinterpret_v_u32m4_i32m4(v362);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v364 = __riscv_vzext_vf4_u32m4(v361, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v365 = __riscv_vreinterpret_v_u32m4_i32m4(v364);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v366 = __riscv_vrsub_vx_i32m4(v365, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v367 = __riscv_vsll_vx_i32m4(v366, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v368 = __riscv_vsub_vv_i32m4(v363, v367, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v369 = __riscv_vfcvt_f_x_v_f32m4(v368, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v370 = __riscv_vfmul_vf_f32m4(v369, v351, 16);
    float* v371 = v11 + 144;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v371, v370, 16);
    const uint8_t* v372 = v8 + 98;
    const uint8_t* v373 = (const uint8_t*) v372;
    const uint8_t v374 = v373[0];
    int v375 = (int) v374;
    int v376 = v375 >> 4;
    int v377 = v376 & 15;
    const uint8_t* v378 = v8 + 106;
    const uint8_t* v379 = (const uint8_t*) v378;
    const uint8_t v380 = v379[0];
    int v381 = (int) v380;
    int v382 = v381 >> 4;
    int v383 = v382 & 3;
    int v384 = v383 << 4;
    int v385 = v377 | v384;
    int v386 = v385 - 32;
    float v387 = (float) v386;
    float v388 = v13 * v387;
    const uint8_t* v389 = v8 + 64;
    const uint8_t* v390 = (const uint8_t*) v389;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v391 = __riscv_vle8_v_u8m1(v390, 16);
    const uint8_t* v392 = v8 + 0;
    const uint8_t* v393 = (const uint8_t*) v392;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v394 = __riscv_vle8_v_u8m1(v393, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v395 = __riscv_vsrl_vx_u8m1(v391, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v396 = __riscv_vand_vx_u8m1(v395, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v397 = __riscv_vsrl_vx_u8m1(v394, 5, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v398 = __riscv_vand_vx_u8m1(v397, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v399 = __riscv_vzext_vf4_u32m4(v396, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v400 = __riscv_vreinterpret_v_u32m4_i32m4(v399);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v401 = __riscv_vzext_vf4_u32m4(v398, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v402 = __riscv_vreinterpret_v_u32m4_i32m4(v401);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v403 = __riscv_vrsub_vx_i32m4(v402, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v404 = __riscv_vsll_vx_i32m4(v403, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v405 = __riscv_vsub_vv_i32m4(v400, v404, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v406 = __riscv_vfcvt_f_x_v_f32m4(v405, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v407 = __riscv_vfmul_vf_f32m4(v406, v388, 16);
    float* v408 = v11 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v408, v407, 16);
    const uint8_t* v409 = v8 + 99;
    const uint8_t* v410 = (const uint8_t*) v409;
    const uint8_t v411 = v410[0];
    int v412 = (int) v411;
    int v413 = v412 >> 4;
    int v414 = v413 & 15;
    const uint8_t* v415 = v8 + 107;
    const uint8_t* v416 = (const uint8_t*) v415;
    const uint8_t v417 = v416[0];
    int v418 = (int) v417;
    int v419 = v418 >> 4;
    int v420 = v419 & 3;
    int v421 = v420 << 4;
    int v422 = v414 | v421;
    int v423 = v422 - 32;
    float v424 = (float) v423;
    float v425 = v13 * v424;
    const uint8_t* v426 = v8 + 80;
    const uint8_t* v427 = (const uint8_t*) v426;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v428 = __riscv_vle8_v_u8m1(v427, 16);
    const uint8_t* v429 = v8 + 16;
    const uint8_t* v430 = (const uint8_t*) v429;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v431 = __riscv_vle8_v_u8m1(v430, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v432 = __riscv_vsrl_vx_u8m1(v428, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v433 = __riscv_vand_vx_u8m1(v432, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v434 = __riscv_vsrl_vx_u8m1(v431, 5, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v435 = __riscv_vand_vx_u8m1(v434, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v436 = __riscv_vzext_vf4_u32m4(v433, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v437 = __riscv_vreinterpret_v_u32m4_i32m4(v436);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v438 = __riscv_vzext_vf4_u32m4(v435, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v439 = __riscv_vreinterpret_v_u32m4_i32m4(v438);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v440 = __riscv_vrsub_vx_i32m4(v439, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v441 = __riscv_vsll_vx_i32m4(v440, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v442 = __riscv_vsub_vv_i32m4(v437, v441, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v443 = __riscv_vfcvt_f_x_v_f32m4(v442, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v444 = __riscv_vfmul_vf_f32m4(v443, v425, 16);
    float* v445 = v11 + 176;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v445, v444, 16);
    const uint8_t* v446 = v8 + 100;
    const uint8_t* v447 = (const uint8_t*) v446;
    const uint8_t v448 = v447[0];
    int v449 = (int) v448;
    int v450 = v449 >> 4;
    int v451 = v450 & 15;
    const uint8_t* v452 = v8 + 104;
    const uint8_t* v453 = (const uint8_t*) v452;
    const uint8_t v454 = v453[0];
    int v455 = (int) v454;
    int v456 = v455 >> 6;
    int v457 = v456 & 3;
    int v458 = v457 << 4;
    int v459 = v451 | v458;
    int v460 = v459 - 32;
    float v461 = (float) v460;
    float v462 = v13 * v461;
    const uint8_t* v463 = v8 + 64;
    const uint8_t* v464 = (const uint8_t*) v463;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v465 = __riscv_vle8_v_u8m1(v464, 16);
    const uint8_t* v466 = v8 + 0;
    const uint8_t* v467 = (const uint8_t*) v466;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v468 = __riscv_vle8_v_u8m1(v467, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v469 = __riscv_vsrl_vx_u8m1(v465, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v470 = __riscv_vand_vx_u8m1(v469, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v471 = __riscv_vsrl_vx_u8m1(v468, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v472 = __riscv_vand_vx_u8m1(v471, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v473 = __riscv_vzext_vf4_u32m4(v470, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v474 = __riscv_vreinterpret_v_u32m4_i32m4(v473);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v475 = __riscv_vzext_vf4_u32m4(v472, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v476 = __riscv_vreinterpret_v_u32m4_i32m4(v475);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v477 = __riscv_vrsub_vx_i32m4(v476, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v478 = __riscv_vsll_vx_i32m4(v477, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v479 = __riscv_vsub_vv_i32m4(v474, v478, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v480 = __riscv_vfcvt_f_x_v_f32m4(v479, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v481 = __riscv_vfmul_vf_f32m4(v480, v462, 16);
    float* v482 = v11 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v482, v481, 16);
    const uint8_t* v483 = v8 + 101;
    const uint8_t* v484 = (const uint8_t*) v483;
    const uint8_t v485 = v484[0];
    int v486 = (int) v485;
    int v487 = v486 >> 4;
    int v488 = v487 & 15;
    const uint8_t* v489 = v8 + 105;
    const uint8_t* v490 = (const uint8_t*) v489;
    const uint8_t v491 = v490[0];
    int v492 = (int) v491;
    int v493 = v492 >> 6;
    int v494 = v493 & 3;
    int v495 = v494 << 4;
    int v496 = v488 | v495;
    int v497 = v496 - 32;
    float v498 = (float) v497;
    float v499 = v13 * v498;
    const uint8_t* v500 = v8 + 80;
    const uint8_t* v501 = (const uint8_t*) v500;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v502 = __riscv_vle8_v_u8m1(v501, 16);
    const uint8_t* v503 = v8 + 16;
    const uint8_t* v504 = (const uint8_t*) v503;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v505 = __riscv_vle8_v_u8m1(v504, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v506 = __riscv_vsrl_vx_u8m1(v502, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v507 = __riscv_vand_vx_u8m1(v506, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v508 = __riscv_vsrl_vx_u8m1(v505, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v509 = __riscv_vand_vx_u8m1(v508, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v510 = __riscv_vzext_vf4_u32m4(v507, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v511 = __riscv_vreinterpret_v_u32m4_i32m4(v510);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v512 = __riscv_vzext_vf4_u32m4(v509, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v513 = __riscv_vreinterpret_v_u32m4_i32m4(v512);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v514 = __riscv_vrsub_vx_i32m4(v513, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v515 = __riscv_vsll_vx_i32m4(v514, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v516 = __riscv_vsub_vv_i32m4(v511, v515, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v517 = __riscv_vfcvt_f_x_v_f32m4(v516, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v518 = __riscv_vfmul_vf_f32m4(v517, v499, 16);
    float* v519 = v11 + 208;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v519, v518, 16);
    const uint8_t* v520 = v8 + 102;
    const uint8_t* v521 = (const uint8_t*) v520;
    const uint8_t v522 = v521[0];
    int v523 = (int) v522;
    int v524 = v523 >> 4;
    int v525 = v524 & 15;
    const uint8_t* v526 = v8 + 106;
    const uint8_t* v527 = (const uint8_t*) v526;
    const uint8_t v528 = v527[0];
    int v529 = (int) v528;
    int v530 = v529 >> 6;
    int v531 = v530 & 3;
    int v532 = v531 << 4;
    int v533 = v525 | v532;
    int v534 = v533 - 32;
    float v535 = (float) v534;
    float v536 = v13 * v535;
    const uint8_t* v537 = v8 + 64;
    const uint8_t* v538 = (const uint8_t*) v537;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v539 = __riscv_vle8_v_u8m1(v538, 16);
    const uint8_t* v540 = v8 + 0;
    const uint8_t* v541 = (const uint8_t*) v540;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v542 = __riscv_vle8_v_u8m1(v541, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v543 = __riscv_vsrl_vx_u8m1(v539, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v544 = __riscv_vand_vx_u8m1(v543, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v545 = __riscv_vsrl_vx_u8m1(v542, 7, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v546 = __riscv_vand_vx_u8m1(v545, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v547 = __riscv_vzext_vf4_u32m4(v544, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v548 = __riscv_vreinterpret_v_u32m4_i32m4(v547);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v549 = __riscv_vzext_vf4_u32m4(v546, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v550 = __riscv_vreinterpret_v_u32m4_i32m4(v549);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v551 = __riscv_vrsub_vx_i32m4(v550, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v552 = __riscv_vsll_vx_i32m4(v551, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v553 = __riscv_vsub_vv_i32m4(v548, v552, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v554 = __riscv_vfcvt_f_x_v_f32m4(v553, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v555 = __riscv_vfmul_vf_f32m4(v554, v536, 16);
    float* v556 = v11 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v556, v555, 16);
    const uint8_t* v557 = v8 + 103;
    const uint8_t* v558 = (const uint8_t*) v557;
    const uint8_t v559 = v558[0];
    int v560 = (int) v559;
    int v561 = v560 >> 4;
    int v562 = v561 & 15;
    const uint8_t* v563 = v8 + 107;
    const uint8_t* v564 = (const uint8_t*) v563;
    const uint8_t v565 = v564[0];
    int v566 = (int) v565;
    int v567 = v566 >> 6;
    int v568 = v567 & 3;
    int v569 = v568 << 4;
    int v570 = v562 | v569;
    int v571 = v570 - 32;
    float v572 = (float) v571;
    float v573 = v13 * v572;
    const uint8_t* v574 = v8 + 80;
    const uint8_t* v575 = (const uint8_t*) v574;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v576 = __riscv_vle8_v_u8m1(v575, 16);
    const uint8_t* v577 = v8 + 16;
    const uint8_t* v578 = (const uint8_t*) v577;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v579 = __riscv_vle8_v_u8m1(v578, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v580 = __riscv_vsrl_vx_u8m1(v576, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v581 = __riscv_vand_vx_u8m1(v580, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v582 = __riscv_vsrl_vx_u8m1(v579, 7, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v583 = __riscv_vand_vx_u8m1(v582, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v584 = __riscv_vzext_vf4_u32m4(v581, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v585 = __riscv_vreinterpret_v_u32m4_i32m4(v584);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v586 = __riscv_vzext_vf4_u32m4(v583, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v587 = __riscv_vreinterpret_v_u32m4_i32m4(v586);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrsub_vx_i32m4
    vint32m4_t v588 = __riscv_vrsub_vx_i32m4(v587, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i32m4
    vint32m4_t v589 = __riscv_vsll_vx_i32m4(v588, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vv_i32m4
    vint32m4_t v590 = __riscv_vsub_vv_i32m4(v585, v589, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v591 = __riscv_vfcvt_f_x_v_f32m4(v590, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v592 = __riscv_vfmul_vf_f32m4(v591, v573, 16);
    float* v593 = v11 + 240;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v593, v592, 16);
  }
  return;
}


