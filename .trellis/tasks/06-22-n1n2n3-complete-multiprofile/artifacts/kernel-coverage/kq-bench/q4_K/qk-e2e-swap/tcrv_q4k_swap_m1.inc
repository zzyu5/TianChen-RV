// AUTO-GENERATED q4_K vec_dot swap (.inc, C) — LMUL=m1
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
    vint32m4_t v82;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m4
    vint32m4_t v83 = __riscv_vmv_v_x_i32m4(0, 16);
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
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v90 = __riscv_vsetvl_e8m1(16);
      const int8_t* v91 = v85 + v89;
      const int8_t* v92 = v8 + v89;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v93 = __riscv_vle8_v_i8m1(v91, v90);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v94 = __riscv_vle8_v_i8m1(v92, v90);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v95 = __riscv_vwmul_vv_i16m2(v93, v94, v90);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m4
      vint32m4_t v96 = v82;
      vint32m4_t v97 = __riscv_vwmacc_vx_i32m4(v96, v88, v95, v90);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v82 = v97;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v98 = __riscv_vsetvl_e8m1(16);
      size_t v99 = v89 + 16;
      const int8_t* v100 = v85 + v99;
      const int8_t* v101 = v8 + v99;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v102 = __riscv_vle8_v_i8m1(v100, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v103 = __riscv_vle8_v_i8m1(v101, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v104 = __riscv_vwmul_vv_i16m2(v102, v103, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m4
      vint32m4_t v105 = v82;
      vint32m4_t v106 = __riscv_vwmacc_vx_i32m4(v105, v88, v104, v98);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v82 = v106;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=aux32_fold_back_to_8
    vint32m4_t v107 = v82;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i32m4
    vint32m4_t v108 = __riscv_vslidedown_vx_i32m4(v107, 8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m4
    vint32m4_t v109 = __riscv_vadd_vv_i32m4(v107, v108, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vget_v_i32m4_i32m2
    vint32m2_t v110 = __riscv_vget_v_i32m4_i32m2(v109, 0);
    // tcrv_emitc.local_variable=aux32c source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    vint32m2_t v111;
    v111 = v110;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v112 = (const float*) v18;
    const float v113 = v112[0];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_term_bsums
    const uint8_t* v114 = v18 + 260;
    const int16_t* v115 = (const int16_t*) v114;
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int v116;
    v116 = 0;
    const int16_t v117 = v115[0];
    int v118 = (int) v117;
    const uint8_t v119 = v81[8];
    int v120 = (int) v119;
    int v121 = v118 * v120;
    int v122 = v116;
    int v123 = v122 + v121;
    v116 = v123;
    const int16_t v124 = v115[1];
    int v125 = (int) v124;
    const uint8_t v126 = v81[8];
    int v127 = (int) v126;
    int v128 = v125 * v127;
    int v129 = v116;
    int v130 = v129 + v128;
    v116 = v130;
    const int16_t v131 = v115[2];
    int v132 = (int) v131;
    const uint8_t v133 = v81[9];
    int v134 = (int) v133;
    int v135 = v132 * v134;
    int v136 = v116;
    int v137 = v136 + v135;
    v116 = v137;
    const int16_t v138 = v115[3];
    int v139 = (int) v138;
    const uint8_t v140 = v81[9];
    int v141 = (int) v140;
    int v142 = v139 * v141;
    int v143 = v116;
    int v144 = v143 + v142;
    v116 = v144;
    const int16_t v145 = v115[4];
    int v146 = (int) v145;
    const uint8_t v147 = v81[10];
    int v148 = (int) v147;
    int v149 = v146 * v148;
    int v150 = v116;
    int v151 = v150 + v149;
    v116 = v151;
    const int16_t v152 = v115[5];
    int v153 = (int) v152;
    const uint8_t v154 = v81[10];
    int v155 = (int) v154;
    int v156 = v153 * v155;
    int v157 = v116;
    int v158 = v157 + v156;
    v116 = v158;
    const int16_t v159 = v115[6];
    int v160 = (int) v159;
    const uint8_t v161 = v81[11];
    int v162 = (int) v161;
    int v163 = v160 * v162;
    int v164 = v116;
    int v165 = v164 + v163;
    v116 = v165;
    const int16_t v166 = v115[7];
    int v167 = (int) v166;
    const uint8_t v168 = v81[11];
    int v169 = (int) v168;
    int v170 = v167 * v169;
    int v171 = v116;
    int v172 = v171 + v170;
    v116 = v172;
    const int16_t v173 = v115[8];
    int v174 = (int) v173;
    const uint8_t v175 = v81[12];
    int v176 = (int) v175;
    int v177 = v174 * v176;
    int v178 = v116;
    int v179 = v178 + v177;
    v116 = v179;
    const int16_t v180 = v115[9];
    int v181 = (int) v180;
    const uint8_t v182 = v81[12];
    int v183 = (int) v182;
    int v184 = v181 * v183;
    int v185 = v116;
    int v186 = v185 + v184;
    v116 = v186;
    const int16_t v187 = v115[10];
    int v188 = (int) v187;
    const uint8_t v189 = v81[13];
    int v190 = (int) v189;
    int v191 = v188 * v190;
    int v192 = v116;
    int v193 = v192 + v191;
    v116 = v193;
    const int16_t v194 = v115[11];
    int v195 = (int) v194;
    const uint8_t v196 = v81[13];
    int v197 = (int) v196;
    int v198 = v195 * v197;
    int v199 = v116;
    int v200 = v199 + v198;
    v116 = v200;
    const int16_t v201 = v115[12];
    int v202 = (int) v201;
    const uint8_t v203 = v81[14];
    int v204 = (int) v203;
    int v205 = v202 * v204;
    int v206 = v116;
    int v207 = v206 + v205;
    v116 = v207;
    const int16_t v208 = v115[13];
    int v209 = (int) v208;
    const uint8_t v210 = v81[14];
    int v211 = (int) v210;
    int v212 = v209 * v211;
    int v213 = v116;
    int v214 = v213 + v212;
    v116 = v214;
    const int16_t v215 = v115[14];
    int v216 = (int) v215;
    const uint8_t v217 = v81[15];
    int v218 = (int) v217;
    int v219 = v216 * v218;
    int v220 = v116;
    int v221 = v220 + v219;
    v116 = v221;
    const int16_t v222 = v115[15];
    int v223 = (int) v222;
    const uint8_t v224 = v81[15];
    int v225 = (int) v224;
    int v226 = v223 * v225;
    int v227 = v116;
    int v228 = v227 + v226;
    v116 = v228;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v229 = (float)*(const _Float16 *)(v16);
    float v230 = v229 * v113;
    vint32m2_t v231 = v111;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v232 = __riscv_vfcvt_f_x_v_f32m2(v231, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v233 = __riscv_vfmul_vf_f32m2(v232, v230, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
    vfloat32m2_t v234 = v11;
    vfloat32m2_t v235 = __riscv_vfadd_vv_f32m2(v234, v233, 8);
    // tcrv_emitc.assign target=sums source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v235;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_dmin
    const uint8_t* v236 = v16 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v237 = (float)*(const _Float16 *)(v236);
    float v238 = v237 * v113;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_subtract
    int v239 = v116;
    float v240 = v13;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v13 = v240 - v238 * (float) v239;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_sums_lanes
  float* v241 = &v10[0];
  vfloat32m2_t v242 = v11;
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
  __riscv_vse32_v_f32m2(v241, v242, 8);
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=horizontal_sum
  float v243 = v13;
  float v244 = v10[0];
  float v245 = v243 + v244;
  float v246 = v10[1];
  float v247 = v245 + v246;
  float v248 = v10[2];
  float v249 = v247 + v248;
  float v250 = v10[3];
  float v251 = v249 + v250;
  float v252 = v10[4];
  float v253 = v251 + v252;
  float v254 = v10[5];
  float v255 = v253 + v254;
  float v256 = v10[6];
  float v257 = v255 + v256;
  float v258 = v10[7];
  float v259 = v257 + v258;
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  v2[0] = v259;
  return;
}



// ABI shim: ggml 8-arg vec_dot -> emitted 4-arg kernel (n, s, x, y); bs/bx/by dropped, nrc==1.
static void tcrv_q4k_vl256_swap(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx,
                                size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    static int tcrv_q4k_once_m1 = 0;
    if (!tcrv_q4k_once_m1) { tcrv_q4k_once_m1 = 1; fprintf(stderr, "TCRV Q4_K BLOCK-DOT ENGAGED (LMUL=m1 compiler-emitted)\n"); }
    tcrv_emitc_q4k_kernel((size_t)n, s, (const uint8_t*)vx, (const uint8_t*)vy);
}
