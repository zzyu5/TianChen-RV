#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void kern_mb4_elided(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
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
    const uint8_t* v23 = v16 + 2;
    const uint8_t* v24 = v23 + 0;
    const int8_t* v25 = (const int8_t*) v24;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v26 = __riscv_vle8_v_i8m1(v25, v22);
    const uint8_t* v27 = v18 + 2;
    const uint8_t* v28 = v27 + 0;
    const int8_t* v29 = (const int8_t*) v28;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v30 = __riscv_vle8_v_i8m1(v29, v22);
    const uint8_t* v31 = v18 + 18;
    const uint8_t* v32 = v31 + 0;
    const int8_t* v33 = (const int8_t*) v32;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v34 = __riscv_vle8_v_i8m1(v33, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
    vint8m1_t v35 = __riscv_vxor_vx_i8m1(v26, 0x88, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
    vint8m1_t v36 = __riscv_vsll_vx_i8m1(v35, 4, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
    vint8m1_t v37 = __riscv_vsra_vx_i8m1(v36, 4, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
    vint8m1_t v38 = __riscv_vsra_vx_i8m1(v35, 4, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v39 = __riscv_vwmul_vv_i16m2(v37, v30, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v40 = __riscv_vwmacc_vv_i16m2(v39, v38, v34, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v41 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v42 = __riscv_vwredsum_vs_i16m2_i32m1(v40, v41, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v43 = __riscv_vmv_x_s_i32m1_i32(v42);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v21 = v43;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v44 = v14 + 1;
    size_t v45 = v44 * 18;
    const uint8_t* v46 = v4 + v45;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v47 = v14 + 1;
    size_t v48 = v47 * 34;
    const uint8_t* v49 = v6 + v48;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v50 = (float)*(const _Float16 *)(v46);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v51 = (float)*(const _Float16 *)(v49);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v52;
    v52 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v53 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v54 = v46 + 2;
    const uint8_t* v55 = v54 + 0;
    const int8_t* v56 = (const int8_t*) v55;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v57 = __riscv_vle8_v_i8m1(v56, v53);
    const uint8_t* v58 = v49 + 2;
    const uint8_t* v59 = v58 + 0;
    const int8_t* v60 = (const int8_t*) v59;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v61 = __riscv_vle8_v_i8m1(v60, v53);
    const uint8_t* v62 = v49 + 18;
    const uint8_t* v63 = v62 + 0;
    const int8_t* v64 = (const int8_t*) v63;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v65 = __riscv_vle8_v_i8m1(v64, v53);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
    vint8m1_t v66 = __riscv_vxor_vx_i8m1(v57, 0x88, v53);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
    vint8m1_t v67 = __riscv_vsll_vx_i8m1(v66, 4, v53);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
    vint8m1_t v68 = __riscv_vsra_vx_i8m1(v67, 4, v53);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
    vint8m1_t v69 = __riscv_vsra_vx_i8m1(v66, 4, v53);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v70 = __riscv_vwmul_vv_i16m2(v68, v61, v53);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v71 = __riscv_vwmacc_vv_i16m2(v70, v69, v65, v53);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v72 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v73 = __riscv_vwredsum_vs_i16m2_i32m1(v71, v72, v53);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v74 = __riscv_vmv_x_s_i32m1_i32(v73);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v52 = v74;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v75 = v14 + 2;
    size_t v76 = v75 * 18;
    const uint8_t* v77 = v4 + v76;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v78 = v14 + 2;
    size_t v79 = v78 * 34;
    const uint8_t* v80 = v6 + v79;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v81 = (float)*(const _Float16 *)(v77);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v82 = (float)*(const _Float16 *)(v80);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v83;
    v83 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v84 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v85 = v77 + 2;
    const uint8_t* v86 = v85 + 0;
    const int8_t* v87 = (const int8_t*) v86;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v88 = __riscv_vle8_v_i8m1(v87, v84);
    const uint8_t* v89 = v80 + 2;
    const uint8_t* v90 = v89 + 0;
    const int8_t* v91 = (const int8_t*) v90;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v92 = __riscv_vle8_v_i8m1(v91, v84);
    const uint8_t* v93 = v80 + 18;
    const uint8_t* v94 = v93 + 0;
    const int8_t* v95 = (const int8_t*) v94;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v96 = __riscv_vle8_v_i8m1(v95, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
    vint8m1_t v97 = __riscv_vxor_vx_i8m1(v88, 0x88, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
    vint8m1_t v98 = __riscv_vsll_vx_i8m1(v97, 4, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
    vint8m1_t v99 = __riscv_vsra_vx_i8m1(v98, 4, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
    vint8m1_t v100 = __riscv_vsra_vx_i8m1(v97, 4, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v101 = __riscv_vwmul_vv_i16m2(v99, v92, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v102 = __riscv_vwmacc_vv_i16m2(v101, v100, v96, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v103 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v104 = __riscv_vwredsum_vs_i16m2_i32m1(v102, v103, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v105 = __riscv_vmv_x_s_i32m1_i32(v104);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v83 = v105;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v106 = v14 + 3;
    size_t v107 = v106 * 18;
    const uint8_t* v108 = v4 + v107;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v109 = v14 + 3;
    size_t v110 = v109 * 34;
    const uint8_t* v111 = v6 + v110;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v112 = (float)*(const _Float16 *)(v108);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v113 = (float)*(const _Float16 *)(v111);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v114;
    v114 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v115 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v116 = v108 + 2;
    const uint8_t* v117 = v116 + 0;
    const int8_t* v118 = (const int8_t*) v117;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v119 = __riscv_vle8_v_i8m1(v118, v115);
    const uint8_t* v120 = v111 + 2;
    const uint8_t* v121 = v120 + 0;
    const int8_t* v122 = (const int8_t*) v121;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v123 = __riscv_vle8_v_i8m1(v122, v115);
    const uint8_t* v124 = v111 + 18;
    const uint8_t* v125 = v124 + 0;
    const int8_t* v126 = (const int8_t*) v125;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v127 = __riscv_vle8_v_i8m1(v126, v115);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
    vint8m1_t v128 = __riscv_vxor_vx_i8m1(v119, 0x88, v115);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
    vint8m1_t v129 = __riscv_vsll_vx_i8m1(v128, 4, v115);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
    vint8m1_t v130 = __riscv_vsra_vx_i8m1(v129, 4, v115);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
    vint8m1_t v131 = __riscv_vsra_vx_i8m1(v128, 4, v115);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v132 = __riscv_vwmul_vv_i16m2(v130, v123, v115);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v133 = __riscv_vwmacc_vv_i16m2(v132, v131, v127, v115);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v134 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v135 = __riscv_vwredsum_vs_i16m2_i32m1(v133, v134, v115);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v136 = __riscv_vmv_x_s_i32m1_i32(v135);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v114 = v136;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v137 = v21;
    float v138 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v138 + ((float) v137 * v19) * v20;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v139 = v52;
    float v140 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v140 + ((float) v139 * v50) * v51;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v141 = v83;
    float v142 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v142 + ((float) v141 * v81) * v82;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v143 = v114;
    float v144 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v144 + ((float) v143 * v112) * v113;
  }
  for (size_t v145 = v13; v145 < v11; v145 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v146 = v145 * 18;
    const uint8_t* v147 = v4 + v146;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v148 = v145 * 34;
    const uint8_t* v149 = v6 + v148;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v150 = (float)*(const _Float16 *)(v147);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v151 = (float)*(const _Float16 *)(v149);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v152;
    v152 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v153 = __riscv_vsetvl_e8m1(16);
    for (size_t v154 = 0; v154 < 16; v154 += v153) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v155 = 16 - v154;
      size_t v156 = __riscv_vsetvl_e8m1(v155);
      const uint8_t* v157 = v147 + 2;
      const uint8_t* v158 = v157 + v154;
      const int8_t* v159 = (const int8_t*) v158;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v160 = __riscv_vle8_v_i8m1(v159, v156);
      const uint8_t* v161 = v149 + 2;
      const uint8_t* v162 = v161 + v154;
      const int8_t* v163 = (const int8_t*) v162;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v164 = __riscv_vle8_v_i8m1(v163, v156);
      const uint8_t* v165 = v149 + 18;
      const uint8_t* v166 = v165 + v154;
      const int8_t* v167 = (const int8_t*) v166;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v168 = __riscv_vle8_v_i8m1(v167, v156);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
      vint8m1_t v169 = __riscv_vxor_vx_i8m1(v160, 0x88, v156);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
      vint8m1_t v170 = __riscv_vsll_vx_i8m1(v169, 4, v156);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v171 = __riscv_vsra_vx_i8m1(v170, 4, v156);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
      vint8m1_t v172 = __riscv_vsra_vx_i8m1(v169, 4, v156);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v173 = __riscv_vwmul_vv_i16m2(v171, v164, v156);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v174 = __riscv_vwmacc_vv_i16m2(v173, v172, v168, v156);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v175 = v152;
      vint32m1_t v176 = __riscv_vmv_v_x_i32m1(v175, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v177 = __riscv_vwredsum_vs_i16m2_i32m1(v174, v176, v156);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v178 = __riscv_vmv_x_s_i32m1_i32(v177);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v152 = v178;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v179 = v152;
    float v180 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v180 + ((float) v179 * v150) * v151;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v181 = v10;
  v2[0] = v181;
  return;
}


