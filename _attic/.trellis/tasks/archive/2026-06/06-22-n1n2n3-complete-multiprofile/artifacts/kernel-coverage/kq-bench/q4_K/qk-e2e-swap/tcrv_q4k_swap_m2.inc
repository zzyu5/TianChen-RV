// AUTO-GENERATED q4_K vec_dot swap (.inc, C) — LMUL=m2
// emitted kernel (oracle-verified bit-exact both VLENs); std includes + stdio provided by quants.c
static void tcrv_emitc_q4k_kernel(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 256;
  // tcrv_emitc.local_variable=aux8 source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  int8_t v7[256];
  const int8_t* v8 = &v7[0];
  // tcrv_emitc.local_variable=utmp source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  uint32_t v9[4];
  // tcrv_emitc.local_variable=sums8 source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v10[8];
  // tcrv_emitc.local_variable=sums source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  vfloat32m2_t v11;
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
  vfloat32m2_t v12 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
  v11 = v12;
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v13;
  v13 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v14 = 0; v14 < v6; v14 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v15 = v14 * 144;
    const uint8_t* v16 = v3 + v15;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v17 = v14 * 292;
    const uint8_t* v18 = v4 + v17;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=unpack_4bit
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v19 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v20 = v16 + 16;
    const uint8_t* v21 = (const uint8_t*) v20;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v22 = __riscv_vle8_v_u8m2(v21, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v23 = __riscv_vand_vx_u8m2(v22, 0x0F, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v24 = __riscv_vreinterpret_v_u8m2_i8m2(v23);
    int8_t* v25 = &v7[0];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v25, v24, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v26 = __riscv_vsrl_vx_u8m2(v22, 0x04, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v27 = __riscv_vreinterpret_v_u8m2_i8m2(v26);
    int8_t* v28 = &v7[32];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v28, v27, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v29 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v30 = v16 + 48;
    const uint8_t* v31 = (const uint8_t*) v30;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v32 = __riscv_vle8_v_u8m2(v31, v29);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v33 = __riscv_vand_vx_u8m2(v32, 0x0F, v29);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v34 = __riscv_vreinterpret_v_u8m2_i8m2(v33);
    int8_t* v35 = &v7[64];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v35, v34, v29);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v36 = __riscv_vsrl_vx_u8m2(v32, 0x04, v29);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v37 = __riscv_vreinterpret_v_u8m2_i8m2(v36);
    int8_t* v38 = &v7[96];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v38, v37, v29);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v39 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v40 = v16 + 80;
    const uint8_t* v41 = (const uint8_t*) v40;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v42 = __riscv_vle8_v_u8m2(v41, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v43 = __riscv_vand_vx_u8m2(v42, 0x0F, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v44 = __riscv_vreinterpret_v_u8m2_i8m2(v43);
    int8_t* v45 = &v7[128];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v45, v44, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v46 = __riscv_vsrl_vx_u8m2(v42, 0x04, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v47 = __riscv_vreinterpret_v_u8m2_i8m2(v46);
    int8_t* v48 = &v7[160];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v48, v47, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v49 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v50 = v16 + 112;
    const uint8_t* v51 = (const uint8_t*) v50;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v52 = __riscv_vle8_v_u8m2(v51, v49);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v53 = __riscv_vand_vx_u8m2(v52, 0x0F, v49);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v54 = __riscv_vreinterpret_v_u8m2_i8m2(v53);
    int8_t* v55 = &v7[192];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v55, v54, v49);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v56 = __riscv_vsrl_vx_u8m2(v52, 0x04, v49);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v57 = __riscv_vreinterpret_v_u8m2_i8m2(v56);
    int8_t* v58 = &v7[224];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v58, v57, v49);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_min_bit_dance
    const uint8_t* v59 = v16 + 4;
    const uint32_t* v60 = (const uint32_t*) v59;
    const uint32_t v61 = v60[0];
    uint32_t v62 = (uint32_t) v61;
    const uint32_t v63 = v60[1];
    uint32_t v64 = (uint32_t) v63;
    const uint32_t v65 = v60[2];
    uint32_t v66 = (uint32_t) v65;
    uint32_t v67 = v64 >> 6;
    uint32_t v68 = v67 & 0x03030303;
    uint32_t v69 = v68 << 4;
    uint32_t v70 = v66 >> 4;
    uint32_t v71 = v70 & 0x0f0f0f0f;
    uint32_t v72 = v71 | v69;
    uint32_t v73 = v64 & 0x3f3f3f3f;
    uint32_t v74 = v62 >> 6;
    uint32_t v75 = v74 & 0x03030303;
    uint32_t v76 = v75 << 4;
    uint32_t v77 = v66 & 0x0f0f0f0f;
    uint32_t v78 = v77 | v76;
    uint32_t v79 = v62 & 0x3f3f3f3f;
    v9[0] = v79;
    v9[1] = v78;
    v9[2] = v73;
    v9[3] = v72;
    uint32_t* v80 = &v9[0];
    const uint8_t* v81 = (const uint8_t*) v80;
    // tcrv_emitc.local_variable=aux32 source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    vint32m8_t v82;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m8
    vint32m8_t v83 = __riscv_vmv_v_x_i32m8(0, 32);
    v82 = v83;
    const uint8_t* v84 = v18 + 4;
    const int8_t* v85 = (const int8_t*) v84;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_loop
    for (size_t v86 = 0; v86 < 8; v86 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_load
      const uint8_t v87 = v81[v86];
      int v88 = (int) v87;
      size_t v89 = v86 * 32;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v90 = __riscv_vsetvl_e8m2(32);
      const int8_t* v91 = v85 + v89;
      const int8_t* v92 = v8 + v89;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v93 = __riscv_vle8_v_i8m2(v91, v90);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v94 = __riscv_vle8_v_i8m2(v92, v90);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v95 = __riscv_vwmul_vv_i16m4(v93, v94, v90);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m8
      vint32m8_t v96 = v82;
      vint32m8_t v97 = __riscv_vwmacc_vx_i32m8(v96, v88, v95, v90);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v82 = v97;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=aux32_fold_back_to_8
    vint32m8_t v98 = v82;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i32m8
    vint32m8_t v99 = __riscv_vslidedown_vx_i32m8(v98, 8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m8
    vint32m8_t v100 = __riscv_vadd_vv_i32m8(v98, v99, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i32m8
    vint32m8_t v101 = __riscv_vslidedown_vx_i32m8(v98, 16, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m8
    vint32m8_t v102 = __riscv_vadd_vv_i32m8(v100, v101, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i32m8
    vint32m8_t v103 = __riscv_vslidedown_vx_i32m8(v98, 24, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m8
    vint32m8_t v104 = __riscv_vadd_vv_i32m8(v102, v103, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vget_v_i32m8_i32m2
    vint32m2_t v105 = __riscv_vget_v_i32m8_i32m2(v104, 0);
    // tcrv_emitc.local_variable=aux32c source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    vint32m2_t v106;
    v106 = v105;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v107 = (const float*) v18;
    const float v108 = v107[0];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_term_bsums
    const uint8_t* v109 = v18 + 260;
    const int16_t* v110 = (const int16_t*) v109;
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int v111;
    v111 = 0;
    const int16_t v112 = v110[0];
    int v113 = (int) v112;
    const uint8_t v114 = v81[8];
    int v115 = (int) v114;
    int v116 = v113 * v115;
    int v117 = v111;
    int v118 = v117 + v116;
    v111 = v118;
    const int16_t v119 = v110[1];
    int v120 = (int) v119;
    const uint8_t v121 = v81[8];
    int v122 = (int) v121;
    int v123 = v120 * v122;
    int v124 = v111;
    int v125 = v124 + v123;
    v111 = v125;
    const int16_t v126 = v110[2];
    int v127 = (int) v126;
    const uint8_t v128 = v81[9];
    int v129 = (int) v128;
    int v130 = v127 * v129;
    int v131 = v111;
    int v132 = v131 + v130;
    v111 = v132;
    const int16_t v133 = v110[3];
    int v134 = (int) v133;
    const uint8_t v135 = v81[9];
    int v136 = (int) v135;
    int v137 = v134 * v136;
    int v138 = v111;
    int v139 = v138 + v137;
    v111 = v139;
    const int16_t v140 = v110[4];
    int v141 = (int) v140;
    const uint8_t v142 = v81[10];
    int v143 = (int) v142;
    int v144 = v141 * v143;
    int v145 = v111;
    int v146 = v145 + v144;
    v111 = v146;
    const int16_t v147 = v110[5];
    int v148 = (int) v147;
    const uint8_t v149 = v81[10];
    int v150 = (int) v149;
    int v151 = v148 * v150;
    int v152 = v111;
    int v153 = v152 + v151;
    v111 = v153;
    const int16_t v154 = v110[6];
    int v155 = (int) v154;
    const uint8_t v156 = v81[11];
    int v157 = (int) v156;
    int v158 = v155 * v157;
    int v159 = v111;
    int v160 = v159 + v158;
    v111 = v160;
    const int16_t v161 = v110[7];
    int v162 = (int) v161;
    const uint8_t v163 = v81[11];
    int v164 = (int) v163;
    int v165 = v162 * v164;
    int v166 = v111;
    int v167 = v166 + v165;
    v111 = v167;
    const int16_t v168 = v110[8];
    int v169 = (int) v168;
    const uint8_t v170 = v81[12];
    int v171 = (int) v170;
    int v172 = v169 * v171;
    int v173 = v111;
    int v174 = v173 + v172;
    v111 = v174;
    const int16_t v175 = v110[9];
    int v176 = (int) v175;
    const uint8_t v177 = v81[12];
    int v178 = (int) v177;
    int v179 = v176 * v178;
    int v180 = v111;
    int v181 = v180 + v179;
    v111 = v181;
    const int16_t v182 = v110[10];
    int v183 = (int) v182;
    const uint8_t v184 = v81[13];
    int v185 = (int) v184;
    int v186 = v183 * v185;
    int v187 = v111;
    int v188 = v187 + v186;
    v111 = v188;
    const int16_t v189 = v110[11];
    int v190 = (int) v189;
    const uint8_t v191 = v81[13];
    int v192 = (int) v191;
    int v193 = v190 * v192;
    int v194 = v111;
    int v195 = v194 + v193;
    v111 = v195;
    const int16_t v196 = v110[12];
    int v197 = (int) v196;
    const uint8_t v198 = v81[14];
    int v199 = (int) v198;
    int v200 = v197 * v199;
    int v201 = v111;
    int v202 = v201 + v200;
    v111 = v202;
    const int16_t v203 = v110[13];
    int v204 = (int) v203;
    const uint8_t v205 = v81[14];
    int v206 = (int) v205;
    int v207 = v204 * v206;
    int v208 = v111;
    int v209 = v208 + v207;
    v111 = v209;
    const int16_t v210 = v110[14];
    int v211 = (int) v210;
    const uint8_t v212 = v81[15];
    int v213 = (int) v212;
    int v214 = v211 * v213;
    int v215 = v111;
    int v216 = v215 + v214;
    v111 = v216;
    const int16_t v217 = v110[15];
    int v218 = (int) v217;
    const uint8_t v219 = v81[15];
    int v220 = (int) v219;
    int v221 = v218 * v220;
    int v222 = v111;
    int v223 = v222 + v221;
    v111 = v223;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v224 = (float)*(const _Float16 *)(v16);
    float v225 = v224 * v108;
    vint32m2_t v226 = v106;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v227 = __riscv_vfcvt_f_x_v_f32m2(v226, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v228 = __riscv_vfmul_vf_f32m2(v227, v225, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
    vfloat32m2_t v229 = v11;
    vfloat32m2_t v230 = __riscv_vfadd_vv_f32m2(v229, v228, 8);
    // tcrv_emitc.assign target=sums source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v230;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_dmin
    const uint8_t* v231 = v16 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v232 = (float)*(const _Float16 *)(v231);
    float v233 = v232 * v108;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_subtract
    int v234 = v111;
    float v235 = v13;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v13 = v235 - v233 * (float) v234;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_sums_lanes
  float* v236 = &v10[0];
  vfloat32m2_t v237 = v11;
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
  __riscv_vse32_v_f32m2(v236, v237, 8);
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=horizontal_sum
  float v238 = v13;
  float v239 = v10[0];
  float v240 = v238 + v239;
  float v241 = v10[1];
  float v242 = v240 + v241;
  float v243 = v10[2];
  float v244 = v242 + v243;
  float v245 = v10[3];
  float v246 = v244 + v245;
  float v247 = v10[4];
  float v248 = v246 + v247;
  float v249 = v10[5];
  float v250 = v248 + v249;
  float v251 = v10[6];
  float v252 = v250 + v251;
  float v253 = v10[7];
  float v254 = v252 + v253;
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  v2[0] = v254;
  return;
}



// ABI shim: ggml 8-arg vec_dot -> emitted 4-arg kernel (n, s, x, y); bs/bx/by dropped, nrc==1.
static void tcrv_q4k_vl256_swap(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx,
                                size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    static int tcrv_q4k_once_m2 = 0;
    if (!tcrv_q4k_once_m2) { tcrv_q4k_once_m2 = 1; fprintf(stderr, "TCRV Q4_K BLOCK-DOT ENGAGED (LMUL=m2 compiler-emitted)\n"); }
    tcrv_emitc_q4k_kernel((size_t)n, s, (const uint8_t*)vx, (const uint8_t*)vy);
}
