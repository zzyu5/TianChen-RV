#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_q6_K_kernel_dequant_q6_K(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v7 = v6 * 210;
    const uint8_t* v8 = v2 + v7;
    const int8_t* v9 = (const int8_t*) v8;
    size_t v10 = v6 * 256;
    float* v11 = v3 + v10;
    float* v12 = (float*) v11;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=q6_K_decode
    const uint8_t* v13 = v8 + 208;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v13);
    const int8_t* v15 = v9 + 192;
    const int8_t v16 = v15[0];
    int v17 = (int) v16;
    float v18 = (float) v17;
    float v19 = v14 * v18;
    const uint8_t* v20 = v8 + 0;
    const uint8_t* v21 = (const uint8_t*) v20;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v22 = __riscv_vle8_v_u8m1(v21, 16);
    const uint8_t* v23 = v8 + 128;
    const uint8_t* v24 = (const uint8_t*) v23;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v25 = __riscv_vle8_v_u8m1(v24, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v26 = __riscv_vand_vx_u8m1(v22, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v27 = __riscv_vsrl_vx_u8m1(v25, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v28 = __riscv_vand_vx_u8m1(v27, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v29 = __riscv_vsll_vx_u8m1(v28, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v30 = __riscv_vor_vv_u8m1(v26, v29, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v31 = __riscv_vzext_vf4_u32m4(v30, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v32 = __riscv_vreinterpret_v_u32m4_i32m4(v31);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v33 = __riscv_vsub_vx_i32m4(v32, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v34 = __riscv_vfcvt_f_x_v_f32m4(v33, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v35 = __riscv_vfmul_vf_f32m4(v34, v19, 16);
    float* v36 = v12 + 0;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v36, v35, 16);
    const int8_t* v37 = v9 + 194;
    const int8_t v38 = v37[0];
    int v39 = (int) v38;
    float v40 = (float) v39;
    float v41 = v14 * v40;
    const uint8_t* v42 = v8 + 32;
    const uint8_t* v43 = (const uint8_t*) v42;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v44 = __riscv_vle8_v_u8m1(v43, 16);
    const uint8_t* v45 = v8 + 128;
    const uint8_t* v46 = (const uint8_t*) v45;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v47 = __riscv_vle8_v_u8m1(v46, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v48 = __riscv_vand_vx_u8m1(v44, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v49 = __riscv_vsrl_vx_u8m1(v47, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v50 = __riscv_vand_vx_u8m1(v49, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v51 = __riscv_vsll_vx_u8m1(v50, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v52 = __riscv_vor_vv_u8m1(v48, v51, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v53 = __riscv_vzext_vf4_u32m4(v52, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v54 = __riscv_vreinterpret_v_u32m4_i32m4(v53);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v55 = __riscv_vsub_vx_i32m4(v54, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v56 = __riscv_vfcvt_f_x_v_f32m4(v55, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v57 = __riscv_vfmul_vf_f32m4(v56, v41, 16);
    float* v58 = v12 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v58, v57, 16);
    const int8_t* v59 = v9 + 196;
    const int8_t v60 = v59[0];
    int v61 = (int) v60;
    float v62 = (float) v61;
    float v63 = v14 * v62;
    const uint8_t* v64 = v8 + 0;
    const uint8_t* v65 = (const uint8_t*) v64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v66 = __riscv_vle8_v_u8m1(v65, 16);
    const uint8_t* v67 = v8 + 128;
    const uint8_t* v68 = (const uint8_t*) v67;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v69 = __riscv_vle8_v_u8m1(v68, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v70 = __riscv_vsrl_vx_u8m1(v66, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v71 = __riscv_vsrl_vx_u8m1(v69, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v72 = __riscv_vand_vx_u8m1(v71, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v73 = __riscv_vsll_vx_u8m1(v72, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v74 = __riscv_vor_vv_u8m1(v70, v73, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v75 = __riscv_vzext_vf4_u32m4(v74, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v76 = __riscv_vreinterpret_v_u32m4_i32m4(v75);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v77 = __riscv_vsub_vx_i32m4(v76, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v78 = __riscv_vfcvt_f_x_v_f32m4(v77, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v79 = __riscv_vfmul_vf_f32m4(v78, v63, 16);
    float* v80 = v12 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v80, v79, 16);
    const int8_t* v81 = v9 + 198;
    const int8_t v82 = v81[0];
    int v83 = (int) v82;
    float v84 = (float) v83;
    float v85 = v14 * v84;
    const uint8_t* v86 = v8 + 32;
    const uint8_t* v87 = (const uint8_t*) v86;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v88 = __riscv_vle8_v_u8m1(v87, 16);
    const uint8_t* v89 = v8 + 128;
    const uint8_t* v90 = (const uint8_t*) v89;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v91 = __riscv_vle8_v_u8m1(v90, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v92 = __riscv_vsrl_vx_u8m1(v88, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v93 = __riscv_vsrl_vx_u8m1(v91, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v94 = __riscv_vand_vx_u8m1(v93, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v95 = __riscv_vsll_vx_u8m1(v94, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v96 = __riscv_vor_vv_u8m1(v92, v95, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v97 = __riscv_vzext_vf4_u32m4(v96, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v98 = __riscv_vreinterpret_v_u32m4_i32m4(v97);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v99 = __riscv_vsub_vx_i32m4(v98, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v100 = __riscv_vfcvt_f_x_v_f32m4(v99, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v101 = __riscv_vfmul_vf_f32m4(v100, v85, 16);
    float* v102 = v12 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v102, v101, 16);
    const int8_t* v103 = v9 + 193;
    const int8_t v104 = v103[0];
    int v105 = (int) v104;
    float v106 = (float) v105;
    float v107 = v14 * v106;
    const uint8_t* v108 = v8 + 16;
    const uint8_t* v109 = (const uint8_t*) v108;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v110 = __riscv_vle8_v_u8m1(v109, 16);
    const uint8_t* v111 = v8 + 144;
    const uint8_t* v112 = (const uint8_t*) v111;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v113 = __riscv_vle8_v_u8m1(v112, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v114 = __riscv_vand_vx_u8m1(v110, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v115 = __riscv_vsrl_vx_u8m1(v113, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v116 = __riscv_vand_vx_u8m1(v115, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v117 = __riscv_vsll_vx_u8m1(v116, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v118 = __riscv_vor_vv_u8m1(v114, v117, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v119 = __riscv_vzext_vf4_u32m4(v118, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v120 = __riscv_vreinterpret_v_u32m4_i32m4(v119);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v121 = __riscv_vsub_vx_i32m4(v120, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v122 = __riscv_vfcvt_f_x_v_f32m4(v121, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v123 = __riscv_vfmul_vf_f32m4(v122, v107, 16);
    float* v124 = v12 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v124, v123, 16);
    const int8_t* v125 = v9 + 195;
    const int8_t v126 = v125[0];
    int v127 = (int) v126;
    float v128 = (float) v127;
    float v129 = v14 * v128;
    const uint8_t* v130 = v8 + 48;
    const uint8_t* v131 = (const uint8_t*) v130;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v132 = __riscv_vle8_v_u8m1(v131, 16);
    const uint8_t* v133 = v8 + 144;
    const uint8_t* v134 = (const uint8_t*) v133;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v135 = __riscv_vle8_v_u8m1(v134, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v136 = __riscv_vand_vx_u8m1(v132, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v137 = __riscv_vsrl_vx_u8m1(v135, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v138 = __riscv_vand_vx_u8m1(v137, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v139 = __riscv_vsll_vx_u8m1(v138, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v140 = __riscv_vor_vv_u8m1(v136, v139, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v141 = __riscv_vzext_vf4_u32m4(v140, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v142 = __riscv_vreinterpret_v_u32m4_i32m4(v141);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v143 = __riscv_vsub_vx_i32m4(v142, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v144 = __riscv_vfcvt_f_x_v_f32m4(v143, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v145 = __riscv_vfmul_vf_f32m4(v144, v129, 16);
    float* v146 = v12 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v146, v145, 16);
    const int8_t* v147 = v9 + 197;
    const int8_t v148 = v147[0];
    int v149 = (int) v148;
    float v150 = (float) v149;
    float v151 = v14 * v150;
    const uint8_t* v152 = v8 + 16;
    const uint8_t* v153 = (const uint8_t*) v152;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v154 = __riscv_vle8_v_u8m1(v153, 16);
    const uint8_t* v155 = v8 + 144;
    const uint8_t* v156 = (const uint8_t*) v155;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v157 = __riscv_vle8_v_u8m1(v156, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v158 = __riscv_vsrl_vx_u8m1(v154, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v159 = __riscv_vsrl_vx_u8m1(v157, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v160 = __riscv_vand_vx_u8m1(v159, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v161 = __riscv_vsll_vx_u8m1(v160, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v162 = __riscv_vor_vv_u8m1(v158, v161, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v163 = __riscv_vzext_vf4_u32m4(v162, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v164 = __riscv_vreinterpret_v_u32m4_i32m4(v163);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v165 = __riscv_vsub_vx_i32m4(v164, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v166 = __riscv_vfcvt_f_x_v_f32m4(v165, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v167 = __riscv_vfmul_vf_f32m4(v166, v151, 16);
    float* v168 = v12 + 80;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v168, v167, 16);
    const int8_t* v169 = v9 + 199;
    const int8_t v170 = v169[0];
    int v171 = (int) v170;
    float v172 = (float) v171;
    float v173 = v14 * v172;
    const uint8_t* v174 = v8 + 48;
    const uint8_t* v175 = (const uint8_t*) v174;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v176 = __riscv_vle8_v_u8m1(v175, 16);
    const uint8_t* v177 = v8 + 144;
    const uint8_t* v178 = (const uint8_t*) v177;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v179 = __riscv_vle8_v_u8m1(v178, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v180 = __riscv_vsrl_vx_u8m1(v176, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v181 = __riscv_vsrl_vx_u8m1(v179, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v182 = __riscv_vand_vx_u8m1(v181, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v183 = __riscv_vsll_vx_u8m1(v182, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v184 = __riscv_vor_vv_u8m1(v180, v183, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v185 = __riscv_vzext_vf4_u32m4(v184, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v186 = __riscv_vreinterpret_v_u32m4_i32m4(v185);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v187 = __riscv_vsub_vx_i32m4(v186, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v188 = __riscv_vfcvt_f_x_v_f32m4(v187, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v189 = __riscv_vfmul_vf_f32m4(v188, v173, 16);
    float* v190 = v12 + 112;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v190, v189, 16);
    const int8_t* v191 = v9 + 200;
    const int8_t v192 = v191[0];
    int v193 = (int) v192;
    float v194 = (float) v193;
    float v195 = v14 * v194;
    const uint8_t* v196 = v8 + 64;
    const uint8_t* v197 = (const uint8_t*) v196;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v198 = __riscv_vle8_v_u8m1(v197, 16);
    const uint8_t* v199 = v8 + 160;
    const uint8_t* v200 = (const uint8_t*) v199;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v201 = __riscv_vle8_v_u8m1(v200, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v202 = __riscv_vand_vx_u8m1(v198, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v203 = __riscv_vsrl_vx_u8m1(v201, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v204 = __riscv_vand_vx_u8m1(v203, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v205 = __riscv_vsll_vx_u8m1(v204, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v206 = __riscv_vor_vv_u8m1(v202, v205, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v207 = __riscv_vzext_vf4_u32m4(v206, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v208 = __riscv_vreinterpret_v_u32m4_i32m4(v207);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v209 = __riscv_vsub_vx_i32m4(v208, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v210 = __riscv_vfcvt_f_x_v_f32m4(v209, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v211 = __riscv_vfmul_vf_f32m4(v210, v195, 16);
    float* v212 = v12 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v212, v211, 16);
    const int8_t* v213 = v9 + 202;
    const int8_t v214 = v213[0];
    int v215 = (int) v214;
    float v216 = (float) v215;
    float v217 = v14 * v216;
    const uint8_t* v218 = v8 + 96;
    const uint8_t* v219 = (const uint8_t*) v218;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v220 = __riscv_vle8_v_u8m1(v219, 16);
    const uint8_t* v221 = v8 + 160;
    const uint8_t* v222 = (const uint8_t*) v221;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v223 = __riscv_vle8_v_u8m1(v222, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v224 = __riscv_vand_vx_u8m1(v220, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v225 = __riscv_vsrl_vx_u8m1(v223, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v226 = __riscv_vand_vx_u8m1(v225, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v227 = __riscv_vsll_vx_u8m1(v226, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v228 = __riscv_vor_vv_u8m1(v224, v227, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v229 = __riscv_vzext_vf4_u32m4(v228, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v230 = __riscv_vreinterpret_v_u32m4_i32m4(v229);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v231 = __riscv_vsub_vx_i32m4(v230, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v232 = __riscv_vfcvt_f_x_v_f32m4(v231, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v233 = __riscv_vfmul_vf_f32m4(v232, v217, 16);
    float* v234 = v12 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v234, v233, 16);
    const int8_t* v235 = v9 + 204;
    const int8_t v236 = v235[0];
    int v237 = (int) v236;
    float v238 = (float) v237;
    float v239 = v14 * v238;
    const uint8_t* v240 = v8 + 64;
    const uint8_t* v241 = (const uint8_t*) v240;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v242 = __riscv_vle8_v_u8m1(v241, 16);
    const uint8_t* v243 = v8 + 160;
    const uint8_t* v244 = (const uint8_t*) v243;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v245 = __riscv_vle8_v_u8m1(v244, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v246 = __riscv_vsrl_vx_u8m1(v242, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v247 = __riscv_vsrl_vx_u8m1(v245, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v248 = __riscv_vand_vx_u8m1(v247, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v249 = __riscv_vsll_vx_u8m1(v248, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v250 = __riscv_vor_vv_u8m1(v246, v249, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v251 = __riscv_vzext_vf4_u32m4(v250, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v252 = __riscv_vreinterpret_v_u32m4_i32m4(v251);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v253 = __riscv_vsub_vx_i32m4(v252, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v254 = __riscv_vfcvt_f_x_v_f32m4(v253, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v255 = __riscv_vfmul_vf_f32m4(v254, v239, 16);
    float* v256 = v12 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v256, v255, 16);
    const int8_t* v257 = v9 + 206;
    const int8_t v258 = v257[0];
    int v259 = (int) v258;
    float v260 = (float) v259;
    float v261 = v14 * v260;
    const uint8_t* v262 = v8 + 96;
    const uint8_t* v263 = (const uint8_t*) v262;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v264 = __riscv_vle8_v_u8m1(v263, 16);
    const uint8_t* v265 = v8 + 160;
    const uint8_t* v266 = (const uint8_t*) v265;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v267 = __riscv_vle8_v_u8m1(v266, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v268 = __riscv_vsrl_vx_u8m1(v264, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v269 = __riscv_vsrl_vx_u8m1(v267, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v270 = __riscv_vand_vx_u8m1(v269, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v271 = __riscv_vsll_vx_u8m1(v270, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v272 = __riscv_vor_vv_u8m1(v268, v271, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v273 = __riscv_vzext_vf4_u32m4(v272, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v274 = __riscv_vreinterpret_v_u32m4_i32m4(v273);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v275 = __riscv_vsub_vx_i32m4(v274, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v276 = __riscv_vfcvt_f_x_v_f32m4(v275, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v277 = __riscv_vfmul_vf_f32m4(v276, v261, 16);
    float* v278 = v12 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v278, v277, 16);
    const int8_t* v279 = v9 + 201;
    const int8_t v280 = v279[0];
    int v281 = (int) v280;
    float v282 = (float) v281;
    float v283 = v14 * v282;
    const uint8_t* v284 = v8 + 80;
    const uint8_t* v285 = (const uint8_t*) v284;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v286 = __riscv_vle8_v_u8m1(v285, 16);
    const uint8_t* v287 = v8 + 176;
    const uint8_t* v288 = (const uint8_t*) v287;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v289 = __riscv_vle8_v_u8m1(v288, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v290 = __riscv_vand_vx_u8m1(v286, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v291 = __riscv_vsrl_vx_u8m1(v289, 0, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v292 = __riscv_vand_vx_u8m1(v291, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v293 = __riscv_vsll_vx_u8m1(v292, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v294 = __riscv_vor_vv_u8m1(v290, v293, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v295 = __riscv_vzext_vf4_u32m4(v294, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v296 = __riscv_vreinterpret_v_u32m4_i32m4(v295);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v297 = __riscv_vsub_vx_i32m4(v296, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v298 = __riscv_vfcvt_f_x_v_f32m4(v297, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v299 = __riscv_vfmul_vf_f32m4(v298, v283, 16);
    float* v300 = v12 + 144;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v300, v299, 16);
    const int8_t* v301 = v9 + 203;
    const int8_t v302 = v301[0];
    int v303 = (int) v302;
    float v304 = (float) v303;
    float v305 = v14 * v304;
    const uint8_t* v306 = v8 + 112;
    const uint8_t* v307 = (const uint8_t*) v306;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v308 = __riscv_vle8_v_u8m1(v307, 16);
    const uint8_t* v309 = v8 + 176;
    const uint8_t* v310 = (const uint8_t*) v309;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v311 = __riscv_vle8_v_u8m1(v310, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v312 = __riscv_vand_vx_u8m1(v308, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v313 = __riscv_vsrl_vx_u8m1(v311, 2, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v314 = __riscv_vand_vx_u8m1(v313, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v315 = __riscv_vsll_vx_u8m1(v314, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v316 = __riscv_vor_vv_u8m1(v312, v315, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v317 = __riscv_vzext_vf4_u32m4(v316, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v318 = __riscv_vreinterpret_v_u32m4_i32m4(v317);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v319 = __riscv_vsub_vx_i32m4(v318, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v320 = __riscv_vfcvt_f_x_v_f32m4(v319, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v321 = __riscv_vfmul_vf_f32m4(v320, v305, 16);
    float* v322 = v12 + 176;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v322, v321, 16);
    const int8_t* v323 = v9 + 205;
    const int8_t v324 = v323[0];
    int v325 = (int) v324;
    float v326 = (float) v325;
    float v327 = v14 * v326;
    const uint8_t* v328 = v8 + 80;
    const uint8_t* v329 = (const uint8_t*) v328;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v330 = __riscv_vle8_v_u8m1(v329, 16);
    const uint8_t* v331 = v8 + 176;
    const uint8_t* v332 = (const uint8_t*) v331;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v333 = __riscv_vle8_v_u8m1(v332, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v334 = __riscv_vsrl_vx_u8m1(v330, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v335 = __riscv_vsrl_vx_u8m1(v333, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v336 = __riscv_vand_vx_u8m1(v335, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v337 = __riscv_vsll_vx_u8m1(v336, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v338 = __riscv_vor_vv_u8m1(v334, v337, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v339 = __riscv_vzext_vf4_u32m4(v338, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v340 = __riscv_vreinterpret_v_u32m4_i32m4(v339);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v341 = __riscv_vsub_vx_i32m4(v340, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v342 = __riscv_vfcvt_f_x_v_f32m4(v341, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v343 = __riscv_vfmul_vf_f32m4(v342, v327, 16);
    float* v344 = v12 + 208;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v344, v343, 16);
    const int8_t* v345 = v9 + 207;
    const int8_t v346 = v345[0];
    int v347 = (int) v346;
    float v348 = (float) v347;
    float v349 = v14 * v348;
    const uint8_t* v350 = v8 + 112;
    const uint8_t* v351 = (const uint8_t*) v350;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v352 = __riscv_vle8_v_u8m1(v351, 16);
    const uint8_t* v353 = v8 + 176;
    const uint8_t* v354 = (const uint8_t*) v353;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v355 = __riscv_vle8_v_u8m1(v354, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v356 = __riscv_vsrl_vx_u8m1(v352, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v357 = __riscv_vsrl_vx_u8m1(v355, 6, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v358 = __riscv_vand_vx_u8m1(v357, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m1
    vuint8m1_t v359 = __riscv_vsll_vx_u8m1(v358, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v360 = __riscv_vor_vv_u8m1(v356, v359, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf4_u32m4
    vuint32m4_t v361 = __riscv_vzext_vf4_u32m4(v360, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m4_i32m4
    vint32m4_t v362 = __riscv_vreinterpret_v_u32m4_i32m4(v361);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v363 = __riscv_vsub_vx_i32m4(v362, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v364 = __riscv_vfcvt_f_x_v_f32m4(v363, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v365 = __riscv_vfmul_vf_f32m4(v364, v349, 16);
    float* v366 = v12 + 240;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v366, v365, 16);
  }
  return;
}


