#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb4e(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 4;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 4) {
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
    const uint8_t* v23 = v12 + 4;
    const uint8_t* v24 = v23 + 0;
    const uint8_t* v25 = (const uint8_t*) v24;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v26 = __riscv_vle8_v_u8m1(v25, v22);
    const uint8_t* v27 = v14 + 4;
    const uint8_t* v28 = v27 + 0;
    const int8_t* v29 = (const int8_t*) v28;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v30 = __riscv_vle8_v_i8m1(v29, v22);
    const uint8_t* v31 = v14 + 20;
    const uint8_t* v32 = v31 + 0;
    const int8_t* v33 = (const int8_t*) v32;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v34 = __riscv_vle8_v_i8m1(v33, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v35 = __riscv_vand_vx_u8m1(v26, 0x0F, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v36 = __riscv_vsrl_vx_u8m1(v26, 0x04, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v37 = __riscv_vreinterpret_v_u8m1_i8m1(v35);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v38 = __riscv_vreinterpret_v_u8m1_i8m1(v36);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v39 = __riscv_vwmul_vv_i16m2(v37, v30, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v40 = __riscv_vwmacc_vv_i16m2(v39, v38, v34, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v41 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v42 = __riscv_vwredsum_vs_i16m2_i32m1(v40, v41, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v43 = __riscv_vmv_x_s_i32m1_i32(v42);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v21 = v43;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v44 = v10 + 1;
    size_t v45 = v44 * 20;
    const uint8_t* v46 = v3 + v45;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v47 = v10 + 1;
    size_t v48 = v47 * 36;
    const uint8_t* v49 = v4 + v48;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v50 = (float)*(const _Float16 *)(v46);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v51 = (float)*(const _Float16 *)(v49);
    const uint8_t* v52 = v46 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v53 = (float)*(const _Float16 *)(v52);
    const uint8_t* v54 = v49 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v55 = (float)*(const _Float16 *)(v54);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v56;
    v56 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v57 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v58 = v46 + 4;
    const uint8_t* v59 = v58 + 0;
    const uint8_t* v60 = (const uint8_t*) v59;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v61 = __riscv_vle8_v_u8m1(v60, v57);
    const uint8_t* v62 = v49 + 4;
    const uint8_t* v63 = v62 + 0;
    const int8_t* v64 = (const int8_t*) v63;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v65 = __riscv_vle8_v_i8m1(v64, v57);
    const uint8_t* v66 = v49 + 20;
    const uint8_t* v67 = v66 + 0;
    const int8_t* v68 = (const int8_t*) v67;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v69 = __riscv_vle8_v_i8m1(v68, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v70 = __riscv_vand_vx_u8m1(v61, 0x0F, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v71 = __riscv_vsrl_vx_u8m1(v61, 0x04, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v72 = __riscv_vreinterpret_v_u8m1_i8m1(v70);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v73 = __riscv_vreinterpret_v_u8m1_i8m1(v71);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v74 = __riscv_vwmul_vv_i16m2(v72, v65, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v75 = __riscv_vwmacc_vv_i16m2(v74, v73, v69, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v76 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v77 = __riscv_vwredsum_vs_i16m2_i32m1(v75, v76, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v78 = __riscv_vmv_x_s_i32m1_i32(v77);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v56 = v78;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v79 = v10 + 2;
    size_t v80 = v79 * 20;
    const uint8_t* v81 = v3 + v80;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v82 = v10 + 2;
    size_t v83 = v82 * 36;
    const uint8_t* v84 = v4 + v83;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v85 = (float)*(const _Float16 *)(v81);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v86 = (float)*(const _Float16 *)(v84);
    const uint8_t* v87 = v81 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v88 = (float)*(const _Float16 *)(v87);
    const uint8_t* v89 = v84 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v90 = (float)*(const _Float16 *)(v89);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v91;
    v91 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v92 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v93 = v81 + 4;
    const uint8_t* v94 = v93 + 0;
    const uint8_t* v95 = (const uint8_t*) v94;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v96 = __riscv_vle8_v_u8m1(v95, v92);
    const uint8_t* v97 = v84 + 4;
    const uint8_t* v98 = v97 + 0;
    const int8_t* v99 = (const int8_t*) v98;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v100 = __riscv_vle8_v_i8m1(v99, v92);
    const uint8_t* v101 = v84 + 20;
    const uint8_t* v102 = v101 + 0;
    const int8_t* v103 = (const int8_t*) v102;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v104 = __riscv_vle8_v_i8m1(v103, v92);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v105 = __riscv_vand_vx_u8m1(v96, 0x0F, v92);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v106 = __riscv_vsrl_vx_u8m1(v96, 0x04, v92);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v107 = __riscv_vreinterpret_v_u8m1_i8m1(v105);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v108 = __riscv_vreinterpret_v_u8m1_i8m1(v106);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v109 = __riscv_vwmul_vv_i16m2(v107, v100, v92);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v110 = __riscv_vwmacc_vv_i16m2(v109, v108, v104, v92);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v111 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v112 = __riscv_vwredsum_vs_i16m2_i32m1(v110, v111, v92);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v113 = __riscv_vmv_x_s_i32m1_i32(v112);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v91 = v113;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v114 = v10 + 3;
    size_t v115 = v114 * 20;
    const uint8_t* v116 = v3 + v115;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v117 = v10 + 3;
    size_t v118 = v117 * 36;
    const uint8_t* v119 = v4 + v118;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v120 = (float)*(const _Float16 *)(v116);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v121 = (float)*(const _Float16 *)(v119);
    const uint8_t* v122 = v116 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v123 = (float)*(const _Float16 *)(v122);
    const uint8_t* v124 = v119 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v125 = (float)*(const _Float16 *)(v124);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v126;
    v126 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v127 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v128 = v116 + 4;
    const uint8_t* v129 = v128 + 0;
    const uint8_t* v130 = (const uint8_t*) v129;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v131 = __riscv_vle8_v_u8m1(v130, v127);
    const uint8_t* v132 = v119 + 4;
    const uint8_t* v133 = v132 + 0;
    const int8_t* v134 = (const int8_t*) v133;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v135 = __riscv_vle8_v_i8m1(v134, v127);
    const uint8_t* v136 = v119 + 20;
    const uint8_t* v137 = v136 + 0;
    const int8_t* v138 = (const int8_t*) v137;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v139 = __riscv_vle8_v_i8m1(v138, v127);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v140 = __riscv_vand_vx_u8m1(v131, 0x0F, v127);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v141 = __riscv_vsrl_vx_u8m1(v131, 0x04, v127);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v142 = __riscv_vreinterpret_v_u8m1_i8m1(v140);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v143 = __riscv_vreinterpret_v_u8m1_i8m1(v141);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v144 = __riscv_vwmul_vv_i16m2(v142, v135, v127);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v145 = __riscv_vwmacc_vv_i16m2(v144, v143, v139, v127);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v146 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v147 = __riscv_vwredsum_vs_i16m2_i32m1(v145, v146, v127);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v148 = __riscv_vmv_x_s_i32m1_i32(v147);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v126 = v148;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v149 = v21;
    float v150 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v150 + ((v15 * v16) * (float) v149 + v18 * v20);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v151 = v56;
    float v152 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v152 + ((v50 * v51) * (float) v151 + v53 * v55);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v153 = v91;
    float v154 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v154 + ((v85 * v86) * (float) v153 + v88 * v90);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v155 = v126;
    float v156 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v156 + ((v120 * v121) * (float) v155 + v123 * v125);
  }
  for (size_t v157 = v9; v157 < v7; v157 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v158 = v157 * 20;
    const uint8_t* v159 = v3 + v158;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v160 = v157 * 36;
    const uint8_t* v161 = v4 + v160;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v162 = (float)*(const _Float16 *)(v159);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v163 = (float)*(const _Float16 *)(v161);
    const uint8_t* v164 = v159 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v165 = (float)*(const _Float16 *)(v164);
    const uint8_t* v166 = v161 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v167 = (float)*(const _Float16 *)(v166);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v168;
    v168 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v169 = __riscv_vsetvl_e8m1(16);
    for (size_t v170 = 0; v170 < 16; v170 += v169) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v171 = 16 - v170;
      size_t v172 = __riscv_vsetvl_e8m1(v171);
      const uint8_t* v173 = v159 + 4;
      const uint8_t* v174 = v173 + v170;
      const uint8_t* v175 = (const uint8_t*) v174;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v176 = __riscv_vle8_v_u8m1(v175, v172);
      const uint8_t* v177 = v161 + 4;
      const uint8_t* v178 = v177 + v170;
      const int8_t* v179 = (const int8_t*) v178;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v180 = __riscv_vle8_v_i8m1(v179, v172);
      const uint8_t* v181 = v161 + 20;
      const uint8_t* v182 = v181 + v170;
      const int8_t* v183 = (const int8_t*) v182;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v184 = __riscv_vle8_v_i8m1(v183, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v185 = __riscv_vand_vx_u8m1(v176, 0x0F, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v186 = __riscv_vsrl_vx_u8m1(v176, 0x04, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v187 = __riscv_vreinterpret_v_u8m1_i8m1(v185);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v188 = __riscv_vreinterpret_v_u8m1_i8m1(v186);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v189 = __riscv_vwmul_vv_i16m2(v187, v180, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v190 = __riscv_vwmacc_vv_i16m2(v189, v188, v184, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v191 = v168;
      vint32m1_t v192 = __riscv_vmv_v_x_i32m1(v191, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v193 = __riscv_vwredsum_vs_i16m2_i32m1(v190, v192, v172);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v194 = __riscv_vmv_x_s_i32m1_i32(v193);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v168 = v194;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v195 = v168;
    float v196 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v196 + ((v162 * v163) * (float) v195 + v165 * v167);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v197 = v6;
  v2[0] = v197;
  return;
}


