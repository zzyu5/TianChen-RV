#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
#include <math.h>
extern "C" void weft_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_rvv_nvfp4_q8_0_block_dot_ATTACK(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_nvfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  // weft_emitc.local_variable=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 64;
  // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_table_load
  vint8m1_t v8 = __riscv_vle8_v_i8m1(weft_nvfp4_kvalues, 16);
  for (size_t v9 = 0; v9 < v7; v9 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_base_x
    size_t v10 = v9 * 36;
    const uint8_t* v11 = v3 + v10;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=q8_block_base_index
    size_t v12 = v9 * 2;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_scale_load
    const uint8_t* v13 = (const uint8_t*) v11;
    const uint8_t v14 = v13[0];
    uint32_t v15 = (uint32_t) v14;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_exp_man_split
    uint32_t v16 = v15 >> 3;
    uint32_t v17 = v16 & 0xF;
    uint32_t v18 = v15 & 0x7;
    int v19 = (int) v17;
    int v20 = (int) v18;
    float v21 = (float) v20;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_ldexpf_branches
    float v22 = ldexpf(v21, -9);
    float v23 = v21 / 8.0f;
    float v24 = 1.0f + v23;
    int v25 = v19 - 7;
    float v26 = ldexpf(v24, v25);
    bool v27 = v17 == 0;
    float v28 = v27 ? v22 : v26;
    float v29 = v28 * 0.5f;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_specials
    bool v30 = v15 == 0;
    bool v31 = v15 == 0x7F;
    bool v32 = v30 || v31;
    float v33 = v32 ? 0.0f : v29;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_weight_quants
    const uint8_t* v34 = v11 + 4;
    const uint8_t* v35 = (const uint8_t*) v34;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_q8_base
    size_t v36 = v12 * 34;
    const uint8_t* v37 = v4 + v36;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v38 = (float)*(const _Float16 *)(v37);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_q8_quants
    const uint8_t* v39 = v37 + 2;
    const int8_t* v40 = (const int8_t*) v39;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v41 = __riscv_vsetvl_e8m1(8);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v42 = __riscv_vle8_v_u8m1(v35, v41);
    const int8_t* v43 = v40 + 8;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v44 = __riscv_vle8_v_i8m1(v40, v41);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v45 = __riscv_vle8_v_i8m1(v43, v41);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v46 = __riscv_vand_vx_u8m1(v42, 0x0F, v41);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v47 = __riscv_vsrl_vx_u8m1(v42, 0x04, v41);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v48 = __riscv_vrgather_vv_i8m1(v8, v46, v41);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v49 = __riscv_vrgather_vv_i8m1(v8, v47, v41);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v50 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v48);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v51 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v49);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v52 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v44);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v53 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v45);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
    vint16m1_t v54 = __riscv_vwmul_vv_i16m1(v50, v52, v41);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m1
    vint16m1_t v55 = __riscv_vwmacc_vv_i16m1(v54, v51, v53, v41);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v56 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m1_i32m1
    vint32m1_t v57 = __riscv_vwredsum_vs_i16m1_i32m1(v55, v56, v41);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v58 = __riscv_vmv_x_s_i32m1_i32(v57);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp32_accumulate
    float v59 = v6;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v6 = v59 + (v38 * v33) * (float) v58;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block
    const uint8_t* v60 = v11 + 1;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_scale_load
    const uint8_t* v61 = (const uint8_t*) v60;
    const uint8_t v62 = v61[0];
    uint32_t v63 = (uint32_t) v62;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_exp_man_split
    uint32_t v64 = v63 >> 3;
    uint32_t v65 = v64 & 0xF;
    uint32_t v66 = v63 & 0x7;
    int v67 = (int) v65;
    int v68 = (int) v66;
    float v69 = (float) v68;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_ldexpf_branches
    float v70 = ldexpf(v69, -9);
    float v71 = v69 / 8.0f;
    float v72 = 1.0f + v71;
    int v73 = v67 - 7;
    float v74 = ldexpf(v72, v73);
    bool v75 = v65 == 0;
    float v76 = v75 ? v70 : v74;
    float v77 = v76 * 0.5f;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_specials
    bool v78 = v63 == 0;
    bool v79 = v63 == 0x7F;
    bool v80 = v78 || v79;
    float v81 = v80 ? 0.0f : v77;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_weight_quants
    const uint8_t* v82 = v11 + 12;
    const uint8_t* v83 = (const uint8_t*) v82;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_q8_base
    size_t v84 = v12 * 34;
    const uint8_t* v85 = v4 + v84;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v86 = (float)*(const _Float16 *)(v85);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_q8_quants
    const uint8_t* v87 = v85 + 18;
    const int8_t* v88 = (const int8_t*) v87;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v89 = __riscv_vsetvl_e8m1(8);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v90 = __riscv_vle8_v_u8m1(v83, v89);
    const int8_t* v91 = v88 + 8;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v92 = __riscv_vle8_v_i8m1(v88, v89);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v93 = __riscv_vle8_v_i8m1(v91, v89);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v94 = __riscv_vand_vx_u8m1(v90, 0x0F, v89);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v95 = __riscv_vsrl_vx_u8m1(v90, 0x04, v89);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v96 = __riscv_vrgather_vv_i8m1(v8, v94, v89);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v97 = __riscv_vrgather_vv_i8m1(v8, v95, v89);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v98 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v96);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v99 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v97);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v100 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v92);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v101 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v93);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
    vint16m1_t v102 = __riscv_vwmul_vv_i16m1(v98, v100, v89);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m1
    vint16m1_t v103 = __riscv_vwmacc_vv_i16m1(v102, v99, v101, v89);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v104 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m1_i32m1
    vint32m1_t v105 = __riscv_vwredsum_vs_i16m1_i32m1(v103, v104, v89);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v106 = __riscv_vmv_x_s_i32m1_i32(v105);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp32_accumulate
    float v107 = v6;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v6 = v107 + (v86 * v81) * (float) v106;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block
    const uint8_t* v108 = v11 + 2;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_scale_load
    const uint8_t* v109 = (const uint8_t*) v108;
    const uint8_t v110 = v109[0];
    uint32_t v111 = (uint32_t) v110;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_exp_man_split
    uint32_t v112 = v111 >> 3;
    uint32_t v113 = v112 & 0xF;
    uint32_t v114 = v111 & 0x7;
    int v115 = (int) v113;
    int v116 = (int) v114;
    float v117 = (float) v116;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_ldexpf_branches
    float v118 = ldexpf(v117, -9);
    float v119 = v117 / 8.0f;
    float v120 = 1.0f + v119;
    int v121 = v115 - 7;
    float v122 = ldexpf(v120, v121);
    bool v123 = v113 == 0;
    float v124 = v123 ? v118 : v122;
    float v125 = v124 * 0.5f;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_specials
    bool v126 = v111 == 0;
    bool v127 = v111 == 0x7F;
    bool v128 = v126 || v127;
    float v129 = v128 ? 0.0f : v125;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_weight_quants
    const uint8_t* v130 = v11 + 20;
    const uint8_t* v131 = (const uint8_t*) v130;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_q8_base
    size_t v132 = v12 + 1;
    size_t v133 = v132 * 34;
    const uint8_t* v134 = v4 + v133;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v135 = (float)*(const _Float16 *)(v134);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_q8_quants
    const uint8_t* v136 = v134 + 2;
    const int8_t* v137 = (const int8_t*) v136;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v138 = __riscv_vsetvl_e8m1(8);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v139 = __riscv_vle8_v_u8m1(v131, v138);
    const int8_t* v140 = v137 + 8;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v141 = __riscv_vle8_v_i8m1(v137, v138);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v142 = __riscv_vle8_v_i8m1(v140, v138);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v143 = __riscv_vand_vx_u8m1(v139, 0x0F, v138);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v144 = __riscv_vsrl_vx_u8m1(v139, 0x04, v138);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v145 = __riscv_vrgather_vv_i8m1(v8, v143, v138);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v146 = __riscv_vrgather_vv_i8m1(v8, v144, v138);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v147 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v145);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v148 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v146);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v149 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v141);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v150 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v142);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
    vint16m1_t v151 = __riscv_vwmul_vv_i16m1(v147, v149, v138);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m1
    vint16m1_t v152 = __riscv_vwmacc_vv_i16m1(v151, v148, v150, v138);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v153 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m1_i32m1
    vint32m1_t v154 = __riscv_vwredsum_vs_i16m1_i32m1(v152, v153, v138);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v155 = __riscv_vmv_x_s_i32m1_i32(v154);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp32_accumulate
    float v156 = v6;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v6 = v156 + (v135 * v129) * (float) v155;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block
    const uint8_t* v157 = v11 + 3;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_scale_load
    const uint8_t* v158 = (const uint8_t*) v157;
    const uint8_t v159 = v158[0];
    uint32_t v160 = (uint32_t) v159;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_exp_man_split
    uint32_t v161 = v160 >> 3;
    uint32_t v162 = v161 & 0xF;
    uint32_t v163 = v160 & 0x7;
    int v164 = (int) v162;
    int v165 = (int) v163;
    float v166 = (float) v165;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_ldexpf_branches
    float v167 = ldexpf(v166, -9);
    float v168 = v166 / 8.0f;
    float v169 = 1.0f + v168;
    int v170 = v164 - 7;
    float v171 = ldexpf(v169, v170);
    bool v172 = v162 == 0;
    float v173 = v172 ? v167 : v171;
    float v174 = v173 * 0.5f;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=ue4m3_specials
    bool v175 = v160 == 0;
    bool v176 = v160 == 0x7F;
    bool v177 = v175 || v176;
    float v178 = v177 ? 0.0f : v174;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_weight_quants
    const uint8_t* v179 = v11 + 28;
    const uint8_t* v180 = (const uint8_t*) v179;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_q8_base
    size_t v181 = v12 + 1;
    size_t v182 = v181 * 34;
    const uint8_t* v183 = v4 + v182;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v184 = (float)*(const _Float16 *)(v183);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_q8_quants
    const uint8_t* v185 = v183 + 18;
    const int8_t* v186 = (const int8_t*) v185;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v187 = __riscv_vsetvl_e8m1(8);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v188 = __riscv_vle8_v_u8m1(v180, v187);
    const int8_t* v189 = v186 + 8;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v190 = __riscv_vle8_v_i8m1(v186, v187);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v191 = __riscv_vle8_v_i8m1(v189, v187);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v192 = __riscv_vand_vx_u8m1(v188, 0x0F, v187);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v193 = __riscv_vsrl_vx_u8m1(v188, 0x04, v187);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v194 = __riscv_vrgather_vv_i8m1(v8, v192, v187);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v195 = __riscv_vrgather_vv_i8m1(v8, v193, v187);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v196 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v194);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v197 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v195);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v198 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v190);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=narrow_trunc_mf2
    vint8mf2_t v199 = __riscv_vlmul_trunc_v_i8m1_i8mf2(v191);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
    vint16m1_t v200 = __riscv_vwmul_vv_i16m1(v196, v198, v187);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m1
    vint16m1_t v201 = __riscv_vwmacc_vv_i16m1(v200, v197, v199, v187);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v202 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m1_i32m1
    vint32m1_t v203 = __riscv_vwredsum_vs_i16m1_i32m1(v201, v202, v187);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v204 = __riscv_vmv_x_s_i32m1_i32(v203);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp32_accumulate
    float v205 = v6;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v6 = v205 + (v184 * v178) * (float) v204;
  }
  // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=store_s
  float v206 = v6;
  v2[0] = v206;
  return;
}


