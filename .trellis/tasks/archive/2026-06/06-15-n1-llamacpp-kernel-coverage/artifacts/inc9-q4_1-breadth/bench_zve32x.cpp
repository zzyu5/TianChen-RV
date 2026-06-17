#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_zve32x(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 2;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 2) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 20;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 36;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    const uint8_t* v17 = v12 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v18 = (float)*(const _Float16 *)(v17);
    const uint8_t* v19 = v14 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v20 = (float)*(const _Float16 *)(v19);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v21;
    v21 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v22 = __riscv_vsetvl_e8m1(16);
    for (size_t v23 = 0; v23 < 16; v23 += v22) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v24 = 16 - v23;
      size_t v25 = __riscv_vsetvl_e8m1(v24);
      const uint8_t* v26 = v12 + 4;
      const uint8_t* v27 = v26 + v23;
      const uint8_t* v28 = (const uint8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v29 = __riscv_vle8_v_u8m1(v28, v25);
      const uint8_t* v30 = v14 + 4;
      const uint8_t* v31 = v30 + v23;
      const int8_t* v32 = (const int8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v33 = __riscv_vle8_v_i8m1(v32, v25);
      const uint8_t* v34 = v14 + 20;
      const uint8_t* v35 = v34 + v23;
      const int8_t* v36 = (const int8_t*) v35;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v37 = __riscv_vle8_v_i8m1(v36, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v38 = __riscv_vand_vx_u8m1(v29, 0x0F, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v39 = __riscv_vsrl_vx_u8m1(v29, 0x04, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v40 = __riscv_vreinterpret_v_u8m1_i8m1(v38);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v41 = __riscv_vreinterpret_v_u8m1_i8m1(v39);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v42 = __riscv_vwmul_vv_i16m2(v40, v33, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v43 = __riscv_vwmacc_vv_i16m2(v42, v41, v37, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v44 = v21;
      vint32m1_t v45 = __riscv_vmv_v_x_i32m1(v44, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v46 = __riscv_vwredsum_vs_i16m2_i32m1(v43, v45, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v47 = __riscv_vmv_x_s_i32m1_i32(v46);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v21 = v47;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v48 = v10 + 1;
    size_t v49 = v48 * 20;
    const uint8_t* v50 = v3 + v49;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v51 = v10 + 1;
    size_t v52 = v51 * 36;
    const uint8_t* v53 = v4 + v52;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v54 = (float)*(const _Float16 *)(v50);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v55 = (float)*(const _Float16 *)(v53);
    const uint8_t* v56 = v50 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v57 = (float)*(const _Float16 *)(v56);
    const uint8_t* v58 = v53 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v59 = (float)*(const _Float16 *)(v58);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v60;
    v60 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v61 = __riscv_vsetvl_e8m1(16);
    for (size_t v62 = 0; v62 < 16; v62 += v61) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v63 = 16 - v62;
      size_t v64 = __riscv_vsetvl_e8m1(v63);
      const uint8_t* v65 = v50 + 4;
      const uint8_t* v66 = v65 + v62;
      const uint8_t* v67 = (const uint8_t*) v66;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v68 = __riscv_vle8_v_u8m1(v67, v64);
      const uint8_t* v69 = v53 + 4;
      const uint8_t* v70 = v69 + v62;
      const int8_t* v71 = (const int8_t*) v70;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v72 = __riscv_vle8_v_i8m1(v71, v64);
      const uint8_t* v73 = v53 + 20;
      const uint8_t* v74 = v73 + v62;
      const int8_t* v75 = (const int8_t*) v74;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v76 = __riscv_vle8_v_i8m1(v75, v64);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v77 = __riscv_vand_vx_u8m1(v68, 0x0F, v64);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v78 = __riscv_vsrl_vx_u8m1(v68, 0x04, v64);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v79 = __riscv_vreinterpret_v_u8m1_i8m1(v77);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v80 = __riscv_vreinterpret_v_u8m1_i8m1(v78);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v81 = __riscv_vwmul_vv_i16m2(v79, v72, v64);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v82 = __riscv_vwmacc_vv_i16m2(v81, v80, v76, v64);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v83 = v60;
      vint32m1_t v84 = __riscv_vmv_v_x_i32m1(v83, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v85 = __riscv_vwredsum_vs_i16m2_i32m1(v82, v84, v64);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v86 = __riscv_vmv_x_s_i32m1_i32(v85);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v60 = v86;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v87 = v21;
    float v88 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v88 + ((v15 * v16) * (float) v87 + v18 * v20);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v89 = v60;
    float v90 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v90 + ((v54 * v55) * (float) v89 + v57 * v59);
  }
  for (size_t v91 = v9; v91 < v7; v91 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v92 = v91 * 20;
    const uint8_t* v93 = v3 + v92;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v94 = v91 * 36;
    const uint8_t* v95 = v4 + v94;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v96 = (float)*(const _Float16 *)(v93);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v97 = (float)*(const _Float16 *)(v95);
    const uint8_t* v98 = v93 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v99 = (float)*(const _Float16 *)(v98);
    const uint8_t* v100 = v95 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v101 = (float)*(const _Float16 *)(v100);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v102;
    v102 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v103 = __riscv_vsetvl_e8m1(16);
    for (size_t v104 = 0; v104 < 16; v104 += v103) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v105 = 16 - v104;
      size_t v106 = __riscv_vsetvl_e8m1(v105);
      const uint8_t* v107 = v93 + 4;
      const uint8_t* v108 = v107 + v104;
      const uint8_t* v109 = (const uint8_t*) v108;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v110 = __riscv_vle8_v_u8m1(v109, v106);
      const uint8_t* v111 = v95 + 4;
      const uint8_t* v112 = v111 + v104;
      const int8_t* v113 = (const int8_t*) v112;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v114 = __riscv_vle8_v_i8m1(v113, v106);
      const uint8_t* v115 = v95 + 20;
      const uint8_t* v116 = v115 + v104;
      const int8_t* v117 = (const int8_t*) v116;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v118 = __riscv_vle8_v_i8m1(v117, v106);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v119 = __riscv_vand_vx_u8m1(v110, 0x0F, v106);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v120 = __riscv_vsrl_vx_u8m1(v110, 0x04, v106);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v121 = __riscv_vreinterpret_v_u8m1_i8m1(v119);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v122 = __riscv_vreinterpret_v_u8m1_i8m1(v120);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v123 = __riscv_vwmul_vv_i16m2(v121, v114, v106);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v124 = __riscv_vwmacc_vv_i16m2(v123, v122, v118, v106);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v125 = v102;
      vint32m1_t v126 = __riscv_vmv_v_x_i32m1(v125, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v127 = __riscv_vwredsum_vs_i16m2_i32m1(v124, v126, v106);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v128 = __riscv_vmv_x_s_i32m1_i32(v127);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v102 = v128;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v129 = v102;
    float v130 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v130 + ((v96 * v97) * (float) v129 + v99 * v101);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v131 = v6;
  v2[0] = v131;
  return;
}


