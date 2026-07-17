// AUTO-GENERATED q4_K vec_dot swap (.inc, C) — LMUL=mf2
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
    vint32m2_t v82;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
    vint32m2_t v83 = __riscv_vmv_v_x_i32m2(0, 8);
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
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v90 = __riscv_vsetvl_e8mf2(8);
      const int8_t* v91 = v85 + v89;
      const int8_t* v92 = v8 + v89;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v93 = __riscv_vle8_v_i8mf2(v91, v90);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v94 = __riscv_vle8_v_i8mf2(v92, v90);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v95 = __riscv_vwmul_vv_i16m1(v93, v94, v90);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v96 = v82;
      vint32m2_t v97 = __riscv_vwmacc_vx_i32m2(v96, v88, v95, v90);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v82 = v97;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v98 = __riscv_vsetvl_e8mf2(8);
      size_t v99 = v89 + 8;
      const int8_t* v100 = v85 + v99;
      const int8_t* v101 = v8 + v99;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v102 = __riscv_vle8_v_i8mf2(v100, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v103 = __riscv_vle8_v_i8mf2(v101, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v104 = __riscv_vwmul_vv_i16m1(v102, v103, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v105 = v82;
      vint32m2_t v106 = __riscv_vwmacc_vx_i32m2(v105, v88, v104, v98);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v82 = v106;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v107 = __riscv_vsetvl_e8mf2(8);
      size_t v108 = v89 + 16;
      const int8_t* v109 = v85 + v108;
      const int8_t* v110 = v8 + v108;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v111 = __riscv_vle8_v_i8mf2(v109, v107);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v112 = __riscv_vle8_v_i8mf2(v110, v107);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v113 = __riscv_vwmul_vv_i16m1(v111, v112, v107);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v114 = v82;
      vint32m2_t v115 = __riscv_vwmacc_vx_i32m2(v114, v88, v113, v107);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v82 = v115;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v116 = __riscv_vsetvl_e8mf2(8);
      size_t v117 = v89 + 24;
      const int8_t* v118 = v85 + v117;
      const int8_t* v119 = v8 + v117;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v120 = __riscv_vle8_v_i8mf2(v118, v116);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v121 = __riscv_vle8_v_i8mf2(v119, v116);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v122 = __riscv_vwmul_vv_i16m1(v120, v121, v116);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v123 = v82;
      vint32m2_t v124 = __riscv_vwmacc_vx_i32m2(v123, v88, v122, v116);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v82 = v124;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v125 = (const float*) v18;
    const float v126 = v125[0];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_term_bsums
    const uint8_t* v127 = v18 + 260;
    const int16_t* v128 = (const int16_t*) v127;
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int v129;
    v129 = 0;
    const int16_t v130 = v128[0];
    int v131 = (int) v130;
    const uint8_t v132 = v81[8];
    int v133 = (int) v132;
    int v134 = v131 * v133;
    int v135 = v129;
    int v136 = v135 + v134;
    v129 = v136;
    const int16_t v137 = v128[1];
    int v138 = (int) v137;
    const uint8_t v139 = v81[8];
    int v140 = (int) v139;
    int v141 = v138 * v140;
    int v142 = v129;
    int v143 = v142 + v141;
    v129 = v143;
    const int16_t v144 = v128[2];
    int v145 = (int) v144;
    const uint8_t v146 = v81[9];
    int v147 = (int) v146;
    int v148 = v145 * v147;
    int v149 = v129;
    int v150 = v149 + v148;
    v129 = v150;
    const int16_t v151 = v128[3];
    int v152 = (int) v151;
    const uint8_t v153 = v81[9];
    int v154 = (int) v153;
    int v155 = v152 * v154;
    int v156 = v129;
    int v157 = v156 + v155;
    v129 = v157;
    const int16_t v158 = v128[4];
    int v159 = (int) v158;
    const uint8_t v160 = v81[10];
    int v161 = (int) v160;
    int v162 = v159 * v161;
    int v163 = v129;
    int v164 = v163 + v162;
    v129 = v164;
    const int16_t v165 = v128[5];
    int v166 = (int) v165;
    const uint8_t v167 = v81[10];
    int v168 = (int) v167;
    int v169 = v166 * v168;
    int v170 = v129;
    int v171 = v170 + v169;
    v129 = v171;
    const int16_t v172 = v128[6];
    int v173 = (int) v172;
    const uint8_t v174 = v81[11];
    int v175 = (int) v174;
    int v176 = v173 * v175;
    int v177 = v129;
    int v178 = v177 + v176;
    v129 = v178;
    const int16_t v179 = v128[7];
    int v180 = (int) v179;
    const uint8_t v181 = v81[11];
    int v182 = (int) v181;
    int v183 = v180 * v182;
    int v184 = v129;
    int v185 = v184 + v183;
    v129 = v185;
    const int16_t v186 = v128[8];
    int v187 = (int) v186;
    const uint8_t v188 = v81[12];
    int v189 = (int) v188;
    int v190 = v187 * v189;
    int v191 = v129;
    int v192 = v191 + v190;
    v129 = v192;
    const int16_t v193 = v128[9];
    int v194 = (int) v193;
    const uint8_t v195 = v81[12];
    int v196 = (int) v195;
    int v197 = v194 * v196;
    int v198 = v129;
    int v199 = v198 + v197;
    v129 = v199;
    const int16_t v200 = v128[10];
    int v201 = (int) v200;
    const uint8_t v202 = v81[13];
    int v203 = (int) v202;
    int v204 = v201 * v203;
    int v205 = v129;
    int v206 = v205 + v204;
    v129 = v206;
    const int16_t v207 = v128[11];
    int v208 = (int) v207;
    const uint8_t v209 = v81[13];
    int v210 = (int) v209;
    int v211 = v208 * v210;
    int v212 = v129;
    int v213 = v212 + v211;
    v129 = v213;
    const int16_t v214 = v128[12];
    int v215 = (int) v214;
    const uint8_t v216 = v81[14];
    int v217 = (int) v216;
    int v218 = v215 * v217;
    int v219 = v129;
    int v220 = v219 + v218;
    v129 = v220;
    const int16_t v221 = v128[13];
    int v222 = (int) v221;
    const uint8_t v223 = v81[14];
    int v224 = (int) v223;
    int v225 = v222 * v224;
    int v226 = v129;
    int v227 = v226 + v225;
    v129 = v227;
    const int16_t v228 = v128[14];
    int v229 = (int) v228;
    const uint8_t v230 = v81[15];
    int v231 = (int) v230;
    int v232 = v229 * v231;
    int v233 = v129;
    int v234 = v233 + v232;
    v129 = v234;
    const int16_t v235 = v128[15];
    int v236 = (int) v235;
    const uint8_t v237 = v81[15];
    int v238 = (int) v237;
    int v239 = v236 * v238;
    int v240 = v129;
    int v241 = v240 + v239;
    v129 = v241;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v242 = (float)*(const _Float16 *)(v16);
    float v243 = v242 * v126;
    vint32m2_t v244 = v82;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v245 = __riscv_vfcvt_f_x_v_f32m2(v244, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v246 = __riscv_vfmul_vf_f32m2(v245, v243, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
    vfloat32m2_t v247 = v11;
    vfloat32m2_t v248 = __riscv_vfadd_vv_f32m2(v247, v246, 8);
    // tcrv_emitc.assign target=sums source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v248;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_dmin
    const uint8_t* v249 = v16 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v250 = (float)*(const _Float16 *)(v249);
    float v251 = v250 * v126;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_subtract
    int v252 = v129;
    float v253 = v13;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v13 = v253 - v251 * (float) v252;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_sums_lanes
  float* v254 = &v10[0];
  vfloat32m2_t v255 = v11;
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
  __riscv_vse32_v_f32m2(v254, v255, 8);
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=horizontal_sum
  float v256 = v13;
  float v257 = v10[0];
  float v258 = v256 + v257;
  float v259 = v10[1];
  float v260 = v258 + v259;
  float v261 = v10[2];
  float v262 = v260 + v261;
  float v263 = v10[3];
  float v264 = v262 + v263;
  float v265 = v10[4];
  float v266 = v264 + v265;
  float v267 = v10[5];
  float v268 = v266 + v267;
  float v269 = v10[6];
  float v270 = v268 + v269;
  float v271 = v10[7];
  float v272 = v270 + v271;
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  v2[0] = v272;
  return;
}



// ABI shim: ggml 8-arg vec_dot -> emitted 4-arg kernel (n, s, x, y); bs/bx/by dropped, nrc==1.
static void tcrv_q4k_vl256_swap(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx,
                                size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    static int tcrv_q4k_once_mf2 = 0;
    if (!tcrv_q4k_once_mf2) { tcrv_q4k_once_mf2 = 1; fprintf(stderr, "TCRV Q4_K BLOCK-DOT ENGAGED (LMUL=mf2 compiler-emitted)\n"); }
    tcrv_emitc_q4k_kernel((size_t)n, s, (const uint8_t*)vx, (const uint8_t*)vy);
}
