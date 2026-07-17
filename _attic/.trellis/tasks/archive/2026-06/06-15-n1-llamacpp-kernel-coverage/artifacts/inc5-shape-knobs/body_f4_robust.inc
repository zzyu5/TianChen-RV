void tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v9 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v10;
  v10 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v11 = v1 / 32;
  size_t v12 = v11 % 4;
  size_t v13 = v11 - v12;
  for (size_t v14 = 0; v14 < v13; v14 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v15 = v14 * 18;
    const uint8_t* v16 = v4 + v15;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v17 = v14 * 34;
    const uint8_t* v18 = v6 + v17;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v19 = (float)*(const _Float16 *)(v16);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v20 = (float)*(const _Float16 *)(v18);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v21;
    v21 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v22 = __riscv_vsetvl_e8m1(16);
    for (size_t v23 = 0; v23 < 16; v23 += v22) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v24 = 16 - v23;
      size_t v25 = __riscv_vsetvl_e8m1(v24);
      const uint8_t* v26 = v16 + 2;
      const uint8_t* v27 = v26 + v23;
      const int8_t* v28 = (const int8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v29 = __riscv_vle8_v_i8m1(v28, v25);
      const uint8_t* v30 = v18 + 2;
      const uint8_t* v31 = v30 + v23;
      const int8_t* v32 = (const int8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v33 = __riscv_vle8_v_i8m1(v32, v25);
      const uint8_t* v34 = v18 + 18;
      const uint8_t* v35 = v34 + v23;
      const int8_t* v36 = (const int8_t*) v35;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v37 = __riscv_vle8_v_i8m1(v36, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
      vint8m1_t v38 = __riscv_vxor_vx_i8m1(v29, 0x88, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
      vint8m1_t v39 = __riscv_vsll_vx_i8m1(v38, 4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v40 = __riscv_vsra_vx_i8m1(v39, 4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v41 = __riscv_vsra_vx_i8m1(v38, 4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v42 = __riscv_vwmul_vv_i16m2(v40, v33, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v43 = __riscv_vwmacc_vv_i16m2(v42, v41, v37, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v44 = v21;
      vint32m1_t v45 = __riscv_vmv_v_x_i32m1(v44, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v46 = __riscv_vwredsum_vs_i16m2_i32m1(v43, v45, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v47 = __riscv_vmv_x_s_i32m1_i32(v46);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v21 = v47;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v48 = v14 + 1;
    size_t v49 = v48 * 18;
    const uint8_t* v50 = v4 + v49;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v51 = v14 + 1;
    size_t v52 = v51 * 34;
    const uint8_t* v53 = v6 + v52;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v54 = (float)*(const _Float16 *)(v50);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v55 = (float)*(const _Float16 *)(v53);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v56;
    v56 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v57 = __riscv_vsetvl_e8m1(16);
    for (size_t v58 = 0; v58 < 16; v58 += v57) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v59 = 16 - v58;
      size_t v60 = __riscv_vsetvl_e8m1(v59);
      const uint8_t* v61 = v50 + 2;
      const uint8_t* v62 = v61 + v58;
      const int8_t* v63 = (const int8_t*) v62;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v64 = __riscv_vle8_v_i8m1(v63, v60);
      const uint8_t* v65 = v53 + 2;
      const uint8_t* v66 = v65 + v58;
      const int8_t* v67 = (const int8_t*) v66;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v68 = __riscv_vle8_v_i8m1(v67, v60);
      const uint8_t* v69 = v53 + 18;
      const uint8_t* v70 = v69 + v58;
      const int8_t* v71 = (const int8_t*) v70;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v72 = __riscv_vle8_v_i8m1(v71, v60);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
      vint8m1_t v73 = __riscv_vxor_vx_i8m1(v64, 0x88, v60);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
      vint8m1_t v74 = __riscv_vsll_vx_i8m1(v73, 4, v60);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v75 = __riscv_vsra_vx_i8m1(v74, 4, v60);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v76 = __riscv_vsra_vx_i8m1(v73, 4, v60);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v77 = __riscv_vwmul_vv_i16m2(v75, v68, v60);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v78 = __riscv_vwmacc_vv_i16m2(v77, v76, v72, v60);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v79 = v56;
      vint32m1_t v80 = __riscv_vmv_v_x_i32m1(v79, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v81 = __riscv_vwredsum_vs_i16m2_i32m1(v78, v80, v60);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v82 = __riscv_vmv_x_s_i32m1_i32(v81);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v56 = v82;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v83 = v14 + 2;
    size_t v84 = v83 * 18;
    const uint8_t* v85 = v4 + v84;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v86 = v14 + 2;
    size_t v87 = v86 * 34;
    const uint8_t* v88 = v6 + v87;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v89 = (float)*(const _Float16 *)(v85);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v90 = (float)*(const _Float16 *)(v88);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v91;
    v91 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v92 = __riscv_vsetvl_e8m1(16);
    for (size_t v93 = 0; v93 < 16; v93 += v92) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v94 = 16 - v93;
      size_t v95 = __riscv_vsetvl_e8m1(v94);
      const uint8_t* v96 = v85 + 2;
      const uint8_t* v97 = v96 + v93;
      const int8_t* v98 = (const int8_t*) v97;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v99 = __riscv_vle8_v_i8m1(v98, v95);
      const uint8_t* v100 = v88 + 2;
      const uint8_t* v101 = v100 + v93;
      const int8_t* v102 = (const int8_t*) v101;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v103 = __riscv_vle8_v_i8m1(v102, v95);
      const uint8_t* v104 = v88 + 18;
      const uint8_t* v105 = v104 + v93;
      const int8_t* v106 = (const int8_t*) v105;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v107 = __riscv_vle8_v_i8m1(v106, v95);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
      vint8m1_t v108 = __riscv_vxor_vx_i8m1(v99, 0x88, v95);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
      vint8m1_t v109 = __riscv_vsll_vx_i8m1(v108, 4, v95);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v110 = __riscv_vsra_vx_i8m1(v109, 4, v95);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v111 = __riscv_vsra_vx_i8m1(v108, 4, v95);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v112 = __riscv_vwmul_vv_i16m2(v110, v103, v95);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v113 = __riscv_vwmacc_vv_i16m2(v112, v111, v107, v95);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v114 = v91;
      vint32m1_t v115 = __riscv_vmv_v_x_i32m1(v114, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v116 = __riscv_vwredsum_vs_i16m2_i32m1(v113, v115, v95);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v117 = __riscv_vmv_x_s_i32m1_i32(v116);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v91 = v117;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v118 = v14 + 3;
    size_t v119 = v118 * 18;
    const uint8_t* v120 = v4 + v119;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v121 = v14 + 3;
    size_t v122 = v121 * 34;
    const uint8_t* v123 = v6 + v122;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v124 = (float)*(const _Float16 *)(v120);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v125 = (float)*(const _Float16 *)(v123);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v126;
    v126 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v127 = __riscv_vsetvl_e8m1(16);
    for (size_t v128 = 0; v128 < 16; v128 += v127) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v129 = 16 - v128;
      size_t v130 = __riscv_vsetvl_e8m1(v129);
      const uint8_t* v131 = v120 + 2;
      const uint8_t* v132 = v131 + v128;
      const int8_t* v133 = (const int8_t*) v132;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v134 = __riscv_vle8_v_i8m1(v133, v130);
      const uint8_t* v135 = v123 + 2;
      const uint8_t* v136 = v135 + v128;
      const int8_t* v137 = (const int8_t*) v136;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v138 = __riscv_vle8_v_i8m1(v137, v130);
      const uint8_t* v139 = v123 + 18;
      const uint8_t* v140 = v139 + v128;
      const int8_t* v141 = (const int8_t*) v140;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v142 = __riscv_vle8_v_i8m1(v141, v130);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
      vint8m1_t v143 = __riscv_vxor_vx_i8m1(v134, 0x88, v130);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
      vint8m1_t v144 = __riscv_vsll_vx_i8m1(v143, 4, v130);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v145 = __riscv_vsra_vx_i8m1(v144, 4, v130);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v146 = __riscv_vsra_vx_i8m1(v143, 4, v130);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v147 = __riscv_vwmul_vv_i16m2(v145, v138, v130);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v148 = __riscv_vwmacc_vv_i16m2(v147, v146, v142, v130);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v149 = v126;
      vint32m1_t v150 = __riscv_vmv_v_x_i32m1(v149, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v151 = __riscv_vwredsum_vs_i16m2_i32m1(v148, v150, v130);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v152 = __riscv_vmv_x_s_i32m1_i32(v151);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v126 = v152;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v153 = v21;
    float v154 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v154 + ((float) v153 * v19) * v20;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v155 = v56;
    float v156 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v156 + ((float) v155 * v54) * v55;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v157 = v91;
    float v158 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v158 + ((float) v157 * v89) * v90;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v159 = v126;
    float v160 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v160 + ((float) v159 * v124) * v125;
  }
  for (size_t v161 = v13; v161 < v11; v161 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v162 = v161 * 18;
    const uint8_t* v163 = v4 + v162;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v164 = v161 * 34;
    const uint8_t* v165 = v6 + v164;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v166 = (float)*(const _Float16 *)(v163);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v167 = (float)*(const _Float16 *)(v165);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v168;
    v168 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v169 = __riscv_vsetvl_e8m1(16);
    for (size_t v170 = 0; v170 < 16; v170 += v169) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v171 = 16 - v170;
      size_t v172 = __riscv_vsetvl_e8m1(v171);
      const uint8_t* v173 = v163 + 2;
      const uint8_t* v174 = v173 + v170;
      const int8_t* v175 = (const int8_t*) v174;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v176 = __riscv_vle8_v_i8m1(v175, v172);
      const uint8_t* v177 = v165 + 2;
      const uint8_t* v178 = v177 + v170;
      const int8_t* v179 = (const int8_t*) v178;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v180 = __riscv_vle8_v_i8m1(v179, v172);
      const uint8_t* v181 = v165 + 18;
      const uint8_t* v182 = v181 + v170;
      const int8_t* v183 = (const int8_t*) v182;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v184 = __riscv_vle8_v_i8m1(v183, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
      vint8m1_t v185 = __riscv_vxor_vx_i8m1(v176, 0x88, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
      vint8m1_t v186 = __riscv_vsll_vx_i8m1(v185, 4, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v187 = __riscv_vsra_vx_i8m1(v186, 4, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v188 = __riscv_vsra_vx_i8m1(v185, 4, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v189 = __riscv_vwmul_vv_i16m2(v187, v180, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v190 = __riscv_vwmacc_vv_i16m2(v189, v188, v184, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v191 = v168;
      vint32m1_t v192 = __riscv_vmv_v_x_i32m1(v191, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v193 = __riscv_vwredsum_vs_i16m2_i32m1(v190, v192, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v194 = __riscv_vmv_x_s_i32m1_i32(v193);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v168 = v194;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v195 = v168;
    float v196 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v196 + ((float) v195 * v166) * v167;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v197 = v10;
  v2[0] = v197;
  return;
}


