#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void kern_m2_mb4_elided(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 4;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 34;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v17;
    v17 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v18 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v19 = v12 + 2;
    const uint8_t* v20 = v19 + 0;
    const int8_t* v21 = (const int8_t*) v20;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v22 = __riscv_vle8_v_i8m2(v21, v18);
    const uint8_t* v23 = v14 + 2;
    const uint8_t* v24 = v23 + 0;
    const int8_t* v25 = (const int8_t*) v24;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v26 = __riscv_vle8_v_i8m2(v25, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v27 = __riscv_vwmul_vv_i16m4(v22, v26, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v28 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v29 = __riscv_vwredsum_vs_i16m4_i32m1(v27, v28, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v30 = __riscv_vmv_x_s_i32m1_i32(v29);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v17 = v30;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v31 = v10 + 1;
    size_t v32 = v31 * 34;
    const uint8_t* v33 = v3 + v32;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v34 = v10 + 1;
    size_t v35 = v34 * 34;
    const uint8_t* v36 = v4 + v35;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v37 = (float)*(const _Float16 *)(v33);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v38 = (float)*(const _Float16 *)(v36);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v39;
    v39 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v40 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v41 = v33 + 2;
    const uint8_t* v42 = v41 + 0;
    const int8_t* v43 = (const int8_t*) v42;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v44 = __riscv_vle8_v_i8m2(v43, v40);
    const uint8_t* v45 = v36 + 2;
    const uint8_t* v46 = v45 + 0;
    const int8_t* v47 = (const int8_t*) v46;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v48 = __riscv_vle8_v_i8m2(v47, v40);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v49 = __riscv_vwmul_vv_i16m4(v44, v48, v40);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v50 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v51 = __riscv_vwredsum_vs_i16m4_i32m1(v49, v50, v40);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v52 = __riscv_vmv_x_s_i32m1_i32(v51);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v39 = v52;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v53 = v10 + 2;
    size_t v54 = v53 * 34;
    const uint8_t* v55 = v3 + v54;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v56 = v10 + 2;
    size_t v57 = v56 * 34;
    const uint8_t* v58 = v4 + v57;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v59 = (float)*(const _Float16 *)(v55);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v60 = (float)*(const _Float16 *)(v58);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v61;
    v61 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v62 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v63 = v55 + 2;
    const uint8_t* v64 = v63 + 0;
    const int8_t* v65 = (const int8_t*) v64;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v66 = __riscv_vle8_v_i8m2(v65, v62);
    const uint8_t* v67 = v58 + 2;
    const uint8_t* v68 = v67 + 0;
    const int8_t* v69 = (const int8_t*) v68;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v70 = __riscv_vle8_v_i8m2(v69, v62);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v71 = __riscv_vwmul_vv_i16m4(v66, v70, v62);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v72 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v73 = __riscv_vwredsum_vs_i16m4_i32m1(v71, v72, v62);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v74 = __riscv_vmv_x_s_i32m1_i32(v73);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v61 = v74;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v75 = v10 + 3;
    size_t v76 = v75 * 34;
    const uint8_t* v77 = v3 + v76;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v78 = v10 + 3;
    size_t v79 = v78 * 34;
    const uint8_t* v80 = v4 + v79;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v81 = (float)*(const _Float16 *)(v77);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v82 = (float)*(const _Float16 *)(v80);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v83;
    v83 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v84 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v85 = v77 + 2;
    const uint8_t* v86 = v85 + 0;
    const int8_t* v87 = (const int8_t*) v86;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v88 = __riscv_vle8_v_i8m2(v87, v84);
    const uint8_t* v89 = v80 + 2;
    const uint8_t* v90 = v89 + 0;
    const int8_t* v91 = (const int8_t*) v90;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v92 = __riscv_vle8_v_i8m2(v91, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v93 = __riscv_vwmul_vv_i16m4(v88, v92, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v94 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v95 = __riscv_vwredsum_vs_i16m4_i32m1(v93, v94, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v96 = __riscv_vmv_x_s_i32m1_i32(v95);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v83 = v96;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v97 = v17;
    float v98 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v98 + (float) v97 * (v15 * v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v99 = v39;
    float v100 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v100 + (float) v99 * (v37 * v38);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v101 = v61;
    float v102 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v102 + (float) v101 * (v59 * v60);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v103 = v83;
    float v104 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v104 + (float) v103 * (v81 * v82);
  }
  for (size_t v105 = v9; v105 < v7; v105 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v106 = v105 * 34;
    const uint8_t* v107 = v3 + v106;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v108 = v105 * 34;
    const uint8_t* v109 = v4 + v108;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v110 = (float)*(const _Float16 *)(v107);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v111 = (float)*(const _Float16 *)(v109);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v112;
    v112 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v113 = __riscv_vsetvl_e8m2(32);
    for (size_t v114 = 0; v114 < 32; v114 += v113) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v115 = 32 - v114;
      size_t v116 = __riscv_vsetvl_e8m2(v115);
      const uint8_t* v117 = v107 + 2;
      const uint8_t* v118 = v117 + v114;
      const int8_t* v119 = (const int8_t*) v118;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v120 = __riscv_vle8_v_i8m2(v119, v116);
      const uint8_t* v121 = v109 + 2;
      const uint8_t* v122 = v121 + v114;
      const int8_t* v123 = (const int8_t*) v122;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v124 = __riscv_vle8_v_i8m2(v123, v116);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v125 = __riscv_vwmul_vv_i16m4(v120, v124, v116);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v126 = v112;
      vint32m1_t v127 = __riscv_vmv_v_x_i32m1(v126, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v128 = __riscv_vwredsum_vs_i16m4_i32m1(v125, v127, v116);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v129 = __riscv_vmv_x_s_i32m1_i32(v128);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v112 = v129;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v130 = v112;
    float v131 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v131 + (float) v130 * (v110 * v111);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v132 = v6;
  v2[0] = v132;
  return;
}


