#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v9 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v10;
  v10 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v11 = v1 / 32;
  size_t v12 = v11 % 2;
  size_t v13 = v11 - v12;
  for (size_t v14 = 0; v14 < v13; v14 += 2) {
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
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v83 = v21;
    float v84 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v84 + ((float) v83 * v19) * v20;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v85 = v56;
    float v86 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v86 + ((float) v85 * v54) * v55;
  }
  for (size_t v87 = v13; v87 < v11; v87 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v88 = v87 * 18;
    const uint8_t* v89 = v4 + v88;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v90 = v87 * 34;
    const uint8_t* v91 = v6 + v90;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v92 = (float)*(const _Float16 *)(v89);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v93 = (float)*(const _Float16 *)(v91);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v94;
    v94 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v95 = __riscv_vsetvl_e8m1(16);
    for (size_t v96 = 0; v96 < 16; v96 += v95) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v97 = 16 - v96;
      size_t v98 = __riscv_vsetvl_e8m1(v97);
      const uint8_t* v99 = v89 + 2;
      const uint8_t* v100 = v99 + v96;
      const int8_t* v101 = (const int8_t*) v100;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v102 = __riscv_vle8_v_i8m1(v101, v98);
      const uint8_t* v103 = v91 + 2;
      const uint8_t* v104 = v103 + v96;
      const int8_t* v105 = (const int8_t*) v104;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v106 = __riscv_vle8_v_i8m1(v105, v98);
      const uint8_t* v107 = v91 + 18;
      const uint8_t* v108 = v107 + v96;
      const int8_t* v109 = (const int8_t*) v108;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v110 = __riscv_vle8_v_i8m1(v109, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
      vint8m1_t v111 = __riscv_vxor_vx_i8m1(v102, 0x88, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
      vint8m1_t v112 = __riscv_vsll_vx_i8m1(v111, 4, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v113 = __riscv_vsra_vx_i8m1(v112, 4, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v114 = __riscv_vsra_vx_i8m1(v111, 4, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v115 = __riscv_vwmul_vv_i16m2(v113, v106, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v116 = __riscv_vwmacc_vv_i16m2(v115, v114, v110, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v117 = v94;
      vint32m1_t v118 = __riscv_vmv_v_x_i32m1(v117, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v119 = __riscv_vwredsum_vs_i16m2_i32m1(v116, v118, v98);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v120 = __riscv_vmv_x_s_i32m1_i32(v119);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v94 = v120;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v121 = v94;
    float v122 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v122 + ((float) v121 * v92) * v93;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v123 = v10;
  v2[0] = v123;
  return;
}


