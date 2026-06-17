#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_ggml_vec_dot_nvfp4_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  static const int8_t tcrv_nvfp4_kvalues[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 64;
  // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=codebook_table_load
  vint8m1_t v8 = __riscv_vle8_v_i8m1(tcrv_nvfp4_kvalues, 16);
  for (size_t v9 = 0; v9 < v7; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v10 = v9 * 36;
    const uint8_t* v11 = v3 + v10;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_block_base_index
    size_t v12 = v9 * 2;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_scale_load
    const uint8_t* v13 = (const uint8_t*) v11;
    const uint8_t v14 = v13[0];
    uint32_t v15 = (uint32_t) v14;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_exp_man_split
    uint32_t v16 = v15 >> 3;
    uint32_t v17 = v16 & 0xF;
    uint32_t v18 = v15 & 0x7;
    int v19 = (int) v17;
    int v20 = (int) v18;
    float v21 = (float) v20;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_ldexpf_branches
    float v22 = ldexpf(v21, -9);
    float v23 = v21 / 8.0f;
    float v24 = 1.0f + v23;
    int v25 = v19 - 7;
    float v26 = ldexpf(v24, v25);
    bool v27 = v17 == 0;
    float v28 = v27 ? v22 : v26;
    float v29 = v28 * 0.5f;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_specials
    bool v30 = v15 == 0;
    bool v31 = v15 == 0x7F;
    bool v32 = v30 || v31;
    float v33 = v32 ? 0.0f : v29;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_weight_quants
    const uint8_t* v34 = v11 + 4;
    const uint8_t* v35 = (const uint8_t*) v34;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_base
    size_t v36 = v12 * 34;
    const uint8_t* v37 = v4 + v36;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v38 = (float)*(const _Float16 *)(v37);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_quants
    const uint8_t* v39 = v37 + 2;
    const int8_t* v40 = (const int8_t*) v39;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v41 = __riscv_vsetvl_e8m1(8);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v42 = __riscv_vle8_v_u8m1(v35, v41);
    const int8_t* v43 = v40 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v44 = __riscv_vle8_v_i8m1(v40, v41);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v45 = __riscv_vle8_v_i8m1(v43, v41);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v46 = __riscv_vand_vx_u8m1(v42, 0x0F, v41);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v47 = __riscv_vsrl_vx_u8m1(v42, 0x04, v41);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v48 = __riscv_vrgather_vv_i8m1(v8, v46, v41);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v49 = __riscv_vrgather_vv_i8m1(v8, v47, v41);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v50 = __riscv_vwmul_vv_i16m2(v48, v44, v41);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v51 = __riscv_vwmacc_vv_i16m2(v50, v49, v45, v41);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v52 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v53 = __riscv_vwredsum_vs_i16m2_i32m1(v51, v52, v41);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v54 = __riscv_vmv_x_s_i32m1_i32(v53);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v55 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v55 + (v38 * v33) * (float) v54;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block
    const uint8_t* v56 = v11 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_scale_load
    const uint8_t* v57 = (const uint8_t*) v56;
    const uint8_t v58 = v57[0];
    uint32_t v59 = (uint32_t) v58;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_exp_man_split
    uint32_t v60 = v59 >> 3;
    uint32_t v61 = v60 & 0xF;
    uint32_t v62 = v59 & 0x7;
    int v63 = (int) v61;
    int v64 = (int) v62;
    float v65 = (float) v64;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_ldexpf_branches
    float v66 = ldexpf(v65, -9);
    float v67 = v65 / 8.0f;
    float v68 = 1.0f + v67;
    int v69 = v63 - 7;
    float v70 = ldexpf(v68, v69);
    bool v71 = v61 == 0;
    float v72 = v71 ? v66 : v70;
    float v73 = v72 * 0.5f;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_specials
    bool v74 = v59 == 0;
    bool v75 = v59 == 0x7F;
    bool v76 = v74 || v75;
    float v77 = v76 ? 0.0f : v73;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_weight_quants
    const uint8_t* v78 = v11 + 12;
    const uint8_t* v79 = (const uint8_t*) v78;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_base
    size_t v80 = v12 * 34;
    const uint8_t* v81 = v4 + v80;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v82 = (float)*(const _Float16 *)(v81);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_quants
    const uint8_t* v83 = v81 + 18;
    const int8_t* v84 = (const int8_t*) v83;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v85 = __riscv_vsetvl_e8m1(8);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v86 = __riscv_vle8_v_u8m1(v79, v85);
    const int8_t* v87 = v84 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v88 = __riscv_vle8_v_i8m1(v84, v85);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v89 = __riscv_vle8_v_i8m1(v87, v85);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v90 = __riscv_vand_vx_u8m1(v86, 0x0F, v85);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v91 = __riscv_vsrl_vx_u8m1(v86, 0x04, v85);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v92 = __riscv_vrgather_vv_i8m1(v8, v90, v85);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v93 = __riscv_vrgather_vv_i8m1(v8, v91, v85);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v94 = __riscv_vwmul_vv_i16m2(v92, v88, v85);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v95 = __riscv_vwmacc_vv_i16m2(v94, v93, v89, v85);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v96 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v97 = __riscv_vwredsum_vs_i16m2_i32m1(v95, v96, v85);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v98 = __riscv_vmv_x_s_i32m1_i32(v97);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v99 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v99 + (v82 * v77) * (float) v98;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block
    const uint8_t* v100 = v11 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_scale_load
    const uint8_t* v101 = (const uint8_t*) v100;
    const uint8_t v102 = v101[0];
    uint32_t v103 = (uint32_t) v102;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_exp_man_split
    uint32_t v104 = v103 >> 3;
    uint32_t v105 = v104 & 0xF;
    uint32_t v106 = v103 & 0x7;
    int v107 = (int) v105;
    int v108 = (int) v106;
    float v109 = (float) v108;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_ldexpf_branches
    float v110 = ldexpf(v109, -9);
    float v111 = v109 / 8.0f;
    float v112 = 1.0f + v111;
    int v113 = v107 - 7;
    float v114 = ldexpf(v112, v113);
    bool v115 = v105 == 0;
    float v116 = v115 ? v110 : v114;
    float v117 = v116 * 0.5f;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_specials
    bool v118 = v103 == 0;
    bool v119 = v103 == 0x7F;
    bool v120 = v118 || v119;
    float v121 = v120 ? 0.0f : v117;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_weight_quants
    const uint8_t* v122 = v11 + 20;
    const uint8_t* v123 = (const uint8_t*) v122;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_base
    size_t v124 = v12 + 1;
    size_t v125 = v124 * 34;
    const uint8_t* v126 = v4 + v125;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v127 = (float)*(const _Float16 *)(v126);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_quants
    const uint8_t* v128 = v126 + 2;
    const int8_t* v129 = (const int8_t*) v128;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v130 = __riscv_vsetvl_e8m1(8);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v131 = __riscv_vle8_v_u8m1(v123, v130);
    const int8_t* v132 = v129 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v133 = __riscv_vle8_v_i8m1(v129, v130);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v134 = __riscv_vle8_v_i8m1(v132, v130);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v135 = __riscv_vand_vx_u8m1(v131, 0x0F, v130);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v136 = __riscv_vsrl_vx_u8m1(v131, 0x04, v130);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v137 = __riscv_vrgather_vv_i8m1(v8, v135, v130);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v138 = __riscv_vrgather_vv_i8m1(v8, v136, v130);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v139 = __riscv_vwmul_vv_i16m2(v137, v133, v130);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v140 = __riscv_vwmacc_vv_i16m2(v139, v138, v134, v130);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v141 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v142 = __riscv_vwredsum_vs_i16m2_i32m1(v140, v141, v130);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v143 = __riscv_vmv_x_s_i32m1_i32(v142);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v144 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v144 + (v127 * v121) * (float) v143;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block
    const uint8_t* v145 = v11 + 3;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_scale_load
    const uint8_t* v146 = (const uint8_t*) v145;
    const uint8_t v147 = v146[0];
    uint32_t v148 = (uint32_t) v147;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_exp_man_split
    uint32_t v149 = v148 >> 3;
    uint32_t v150 = v149 & 0xF;
    uint32_t v151 = v148 & 0x7;
    int v152 = (int) v150;
    int v153 = (int) v151;
    float v154 = (float) v153;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_ldexpf_branches
    float v155 = ldexpf(v154, -9);
    float v156 = v154 / 8.0f;
    float v157 = 1.0f + v156;
    int v158 = v152 - 7;
    float v159 = ldexpf(v157, v158);
    bool v160 = v150 == 0;
    float v161 = v160 ? v155 : v159;
    float v162 = v161 * 0.5f;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=ue4m3_specials
    bool v163 = v148 == 0;
    bool v164 = v148 == 0x7F;
    bool v165 = v163 || v164;
    float v166 = v165 ? 0.0f : v162;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_weight_quants
    const uint8_t* v167 = v11 + 28;
    const uint8_t* v168 = (const uint8_t*) v167;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_base
    size_t v169 = v12 + 1;
    size_t v170 = v169 * 34;
    const uint8_t* v171 = v4 + v170;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v172 = (float)*(const _Float16 *)(v171);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_q8_quants
    const uint8_t* v173 = v171 + 18;
    const int8_t* v174 = (const int8_t*) v173;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v175 = __riscv_vsetvl_e8m1(8);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v176 = __riscv_vle8_v_u8m1(v168, v175);
    const int8_t* v177 = v174 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v178 = __riscv_vle8_v_i8m1(v174, v175);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v179 = __riscv_vle8_v_i8m1(v177, v175);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v180 = __riscv_vand_vx_u8m1(v176, 0x0F, v175);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v181 = __riscv_vsrl_vx_u8m1(v176, 0x04, v175);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v182 = __riscv_vrgather_vv_i8m1(v8, v180, v175);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v183 = __riscv_vrgather_vv_i8m1(v8, v181, v175);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v184 = __riscv_vwmul_vv_i16m2(v182, v178, v175);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v185 = __riscv_vwmacc_vv_i16m2(v184, v183, v179, v175);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v186 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v187 = __riscv_vwredsum_vs_i16m2_i32m1(v185, v186, v175);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v188 = __riscv_vmv_x_s_i32m1_i32(v187);
    // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v189 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v189 + (v172 * v166) * (float) v188;
  }
  // tcrv_emitc.source_op=tcrv_rvv.nvfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v190 = v6;
  v2[0] = v190;
  return;
}


