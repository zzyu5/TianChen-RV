#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_ggml_vec_dot_iq4_xs_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  static const int8_t tcrv_iq4_xs_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 256;
  // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=codebook_table_load
  vint8m1_t v8 = __riscv_vle8_v_i8m1(tcrv_iq4_xs_kvalues, 16);
  // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v9 = 0; v9 < v7; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v10 = v9 * 136;
    const uint8_t* v11 = v3 + v10;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v12 = v9 * 292;
    const uint8_t* v13 = v4 + v12;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v11);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v15 = (const float*) v13;
    const float v16 = v15[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d4d8
    float v17 = v14 * v16;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scales_h_load
    const uint8_t* v18 = v11 + 2;
    const uint16_t* v19 = (const uint16_t*) v18;
    const uint16_t v20 = v19[0];
    int v21 = (int) v20;
    const uint8_t* v22 = v11 + 4;
    const uint8_t* v23 = (const uint8_t*) v22;
    const uint8_t* v24 = v11 + 8;
    const uint8_t* v25 = (const uint8_t*) v24;
    const uint8_t* v26 = v13 + 4;
    const int8_t* v27 = (const int8_t*) v26;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_signed_scale
    const uint8_t v28 = v23[0];
    int v29 = (int) v28;
    int v30 = v29 >> 0;
    int v31 = v30 & 15;
    int v32 = v21 >> 0;
    int v33 = v32 & 3;
    int v34 = v33 << 4;
    int v35 = v31 | v34;
    int v36 = v35 - 32;
    float v37 = (float) v36;
    float v38 = v17 * v37;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_codebook_dot
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v39 = __riscv_vsetvl_e8m1(16);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v40 = __riscv_vle8_v_u8m1(v25, v39);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v41 = __riscv_vle8_v_i8m1(v27, v39);
    const int8_t* v42 = v27 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v43 = __riscv_vle8_v_i8m1(v42, v39);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v44 = __riscv_vand_vx_u8m1(v40, 0x0F, v39);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v45 = __riscv_vsrl_vx_u8m1(v40, 0x04, v39);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v46 = __riscv_vrgather_vv_i8m1(v8, v44, v39);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v47 = __riscv_vrgather_vv_i8m1(v8, v45, v39);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v48 = __riscv_vwmul_vv_i16m2(v46, v41, v39);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v49 = __riscv_vwmacc_vv_i16m2(v48, v47, v43, v39);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v50 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v51 = __riscv_vwredsum_vs_i16m2_i32m1(v49, v50, v39);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v52 = __riscv_vmv_x_s_i32m1_i32(v51);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v53 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v53 + v38 * (float) v52;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_signed_scale
    const uint8_t v54 = v23[0];
    int v55 = (int) v54;
    int v56 = v55 >> 4;
    int v57 = v56 & 15;
    int v58 = v21 >> 2;
    int v59 = v58 & 3;
    int v60 = v59 << 4;
    int v61 = v57 | v60;
    int v62 = v61 - 32;
    float v63 = (float) v62;
    float v64 = v17 * v63;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_codebook_dot
    const uint8_t* v65 = v25 + 16;
    const int8_t* v66 = v27 + 32;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v67 = __riscv_vsetvl_e8m1(16);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v68 = __riscv_vle8_v_u8m1(v65, v67);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v69 = __riscv_vle8_v_i8m1(v66, v67);
    const int8_t* v70 = v66 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v71 = __riscv_vle8_v_i8m1(v70, v67);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v72 = __riscv_vand_vx_u8m1(v68, 0x0F, v67);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v73 = __riscv_vsrl_vx_u8m1(v68, 0x04, v67);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v74 = __riscv_vrgather_vv_i8m1(v8, v72, v67);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v75 = __riscv_vrgather_vv_i8m1(v8, v73, v67);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v76 = __riscv_vwmul_vv_i16m2(v74, v69, v67);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v77 = __riscv_vwmacc_vv_i16m2(v76, v75, v71, v67);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v78 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v79 = __riscv_vwredsum_vs_i16m2_i32m1(v77, v78, v67);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v80 = __riscv_vmv_x_s_i32m1_i32(v79);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v81 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v81 + v64 * (float) v80;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_signed_scale
    const uint8_t v82 = v23[1];
    int v83 = (int) v82;
    int v84 = v83 >> 0;
    int v85 = v84 & 15;
    int v86 = v21 >> 4;
    int v87 = v86 & 3;
    int v88 = v87 << 4;
    int v89 = v85 | v88;
    int v90 = v89 - 32;
    float v91 = (float) v90;
    float v92 = v17 * v91;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_codebook_dot
    const uint8_t* v93 = v25 + 32;
    const int8_t* v94 = v27 + 64;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v95 = __riscv_vsetvl_e8m1(16);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v96 = __riscv_vle8_v_u8m1(v93, v95);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v97 = __riscv_vle8_v_i8m1(v94, v95);
    const int8_t* v98 = v94 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v99 = __riscv_vle8_v_i8m1(v98, v95);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v100 = __riscv_vand_vx_u8m1(v96, 0x0F, v95);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v101 = __riscv_vsrl_vx_u8m1(v96, 0x04, v95);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v102 = __riscv_vrgather_vv_i8m1(v8, v100, v95);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v103 = __riscv_vrgather_vv_i8m1(v8, v101, v95);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v104 = __riscv_vwmul_vv_i16m2(v102, v97, v95);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v105 = __riscv_vwmacc_vv_i16m2(v104, v103, v99, v95);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v106 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v107 = __riscv_vwredsum_vs_i16m2_i32m1(v105, v106, v95);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v108 = __riscv_vmv_x_s_i32m1_i32(v107);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v109 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v109 + v92 * (float) v108;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_signed_scale
    const uint8_t v110 = v23[1];
    int v111 = (int) v110;
    int v112 = v111 >> 4;
    int v113 = v112 & 15;
    int v114 = v21 >> 6;
    int v115 = v114 & 3;
    int v116 = v115 << 4;
    int v117 = v113 | v116;
    int v118 = v117 - 32;
    float v119 = (float) v118;
    float v120 = v17 * v119;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_codebook_dot
    const uint8_t* v121 = v25 + 48;
    const int8_t* v122 = v27 + 96;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v123 = __riscv_vsetvl_e8m1(16);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v124 = __riscv_vle8_v_u8m1(v121, v123);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v125 = __riscv_vle8_v_i8m1(v122, v123);
    const int8_t* v126 = v122 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v127 = __riscv_vle8_v_i8m1(v126, v123);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v128 = __riscv_vand_vx_u8m1(v124, 0x0F, v123);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v129 = __riscv_vsrl_vx_u8m1(v124, 0x04, v123);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v130 = __riscv_vrgather_vv_i8m1(v8, v128, v123);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v131 = __riscv_vrgather_vv_i8m1(v8, v129, v123);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v132 = __riscv_vwmul_vv_i16m2(v130, v125, v123);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v133 = __riscv_vwmacc_vv_i16m2(v132, v131, v127, v123);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v134 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v135 = __riscv_vwredsum_vs_i16m2_i32m1(v133, v134, v123);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v136 = __riscv_vmv_x_s_i32m1_i32(v135);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v137 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v137 + v120 * (float) v136;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_signed_scale
    const uint8_t v138 = v23[2];
    int v139 = (int) v138;
    int v140 = v139 >> 0;
    int v141 = v140 & 15;
    int v142 = v21 >> 8;
    int v143 = v142 & 3;
    int v144 = v143 << 4;
    int v145 = v141 | v144;
    int v146 = v145 - 32;
    float v147 = (float) v146;
    float v148 = v17 * v147;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_codebook_dot
    const uint8_t* v149 = v25 + 64;
    const int8_t* v150 = v27 + 128;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v151 = __riscv_vsetvl_e8m1(16);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v152 = __riscv_vle8_v_u8m1(v149, v151);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v153 = __riscv_vle8_v_i8m1(v150, v151);
    const int8_t* v154 = v150 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v155 = __riscv_vle8_v_i8m1(v154, v151);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v156 = __riscv_vand_vx_u8m1(v152, 0x0F, v151);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v157 = __riscv_vsrl_vx_u8m1(v152, 0x04, v151);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v158 = __riscv_vrgather_vv_i8m1(v8, v156, v151);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v159 = __riscv_vrgather_vv_i8m1(v8, v157, v151);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v160 = __riscv_vwmul_vv_i16m2(v158, v153, v151);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v161 = __riscv_vwmacc_vv_i16m2(v160, v159, v155, v151);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v162 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v163 = __riscv_vwredsum_vs_i16m2_i32m1(v161, v162, v151);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v164 = __riscv_vmv_x_s_i32m1_i32(v163);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v165 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v165 + v148 * (float) v164;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_signed_scale
    const uint8_t v166 = v23[2];
    int v167 = (int) v166;
    int v168 = v167 >> 4;
    int v169 = v168 & 15;
    int v170 = v21 >> 10;
    int v171 = v170 & 3;
    int v172 = v171 << 4;
    int v173 = v169 | v172;
    int v174 = v173 - 32;
    float v175 = (float) v174;
    float v176 = v17 * v175;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_codebook_dot
    const uint8_t* v177 = v25 + 80;
    const int8_t* v178 = v27 + 160;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v179 = __riscv_vsetvl_e8m1(16);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v180 = __riscv_vle8_v_u8m1(v177, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v181 = __riscv_vle8_v_i8m1(v178, v179);
    const int8_t* v182 = v178 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v183 = __riscv_vle8_v_i8m1(v182, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v184 = __riscv_vand_vx_u8m1(v180, 0x0F, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v185 = __riscv_vsrl_vx_u8m1(v180, 0x04, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v186 = __riscv_vrgather_vv_i8m1(v8, v184, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v187 = __riscv_vrgather_vv_i8m1(v8, v185, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v188 = __riscv_vwmul_vv_i16m2(v186, v181, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v189 = __riscv_vwmacc_vv_i16m2(v188, v187, v183, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v190 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v191 = __riscv_vwredsum_vs_i16m2_i32m1(v189, v190, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v192 = __riscv_vmv_x_s_i32m1_i32(v191);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v193 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v193 + v176 * (float) v192;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_signed_scale
    const uint8_t v194 = v23[3];
    int v195 = (int) v194;
    int v196 = v195 >> 0;
    int v197 = v196 & 15;
    int v198 = v21 >> 12;
    int v199 = v198 & 3;
    int v200 = v199 << 4;
    int v201 = v197 | v200;
    int v202 = v201 - 32;
    float v203 = (float) v202;
    float v204 = v17 * v203;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_codebook_dot
    const uint8_t* v205 = v25 + 96;
    const int8_t* v206 = v27 + 192;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v207 = __riscv_vsetvl_e8m1(16);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v208 = __riscv_vle8_v_u8m1(v205, v207);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v209 = __riscv_vle8_v_i8m1(v206, v207);
    const int8_t* v210 = v206 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v211 = __riscv_vle8_v_i8m1(v210, v207);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v212 = __riscv_vand_vx_u8m1(v208, 0x0F, v207);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v213 = __riscv_vsrl_vx_u8m1(v208, 0x04, v207);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v214 = __riscv_vrgather_vv_i8m1(v8, v212, v207);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v215 = __riscv_vrgather_vv_i8m1(v8, v213, v207);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v216 = __riscv_vwmul_vv_i16m2(v214, v209, v207);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v217 = __riscv_vwmacc_vv_i16m2(v216, v215, v211, v207);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v218 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v219 = __riscv_vwredsum_vs_i16m2_i32m1(v217, v218, v207);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v220 = __riscv_vmv_x_s_i32m1_i32(v219);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v221 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v221 + v204 * (float) v220;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_signed_scale
    const uint8_t v222 = v23[3];
    int v223 = (int) v222;
    int v224 = v223 >> 4;
    int v225 = v224 & 15;
    int v226 = v21 >> 14;
    int v227 = v226 & 3;
    int v228 = v227 << 4;
    int v229 = v225 | v228;
    int v230 = v229 - 32;
    float v231 = (float) v230;
    float v232 = v17 * v231;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_codebook_dot
    const uint8_t* v233 = v25 + 112;
    const int8_t* v234 = v27 + 224;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v235 = __riscv_vsetvl_e8m1(16);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v236 = __riscv_vle8_v_u8m1(v233, v235);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v237 = __riscv_vle8_v_i8m1(v234, v235);
    const int8_t* v238 = v234 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v239 = __riscv_vle8_v_i8m1(v238, v235);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v240 = __riscv_vand_vx_u8m1(v236, 0x0F, v235);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v241 = __riscv_vsrl_vx_u8m1(v236, 0x04, v235);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v242 = __riscv_vrgather_vv_i8m1(v8, v240, v235);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v243 = __riscv_vrgather_vv_i8m1(v8, v241, v235);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v244 = __riscv_vwmul_vv_i16m2(v242, v237, v235);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v245 = __riscv_vwmacc_vv_i16m2(v244, v243, v239, v235);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v246 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v247 = __riscv_vwredsum_vs_i16m2_i32m1(v245, v246, v235);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v248 = __riscv_vmv_x_s_i32m1_i32(v247);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v249 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v249 + v232 * (float) v248;
  }
  // tcrv_emitc.source_op=tcrv_rvv.iq4_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v250 = v6;
  v2[0] = v250;
  return;
}


