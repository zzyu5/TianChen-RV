#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 4;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 24;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 36;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    const uint8_t* v17 = v12 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v18 = (float)*(const _Float16 *)(v17);
    const uint8_t* v19 = v14 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v20 = (float)*(const _Float16 *)(v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v21 = v12 + 4;
    uint32_t v22 = (uint16_t)*(const uint16_t *)(v21);
    const uint8_t* v23 = v12 + 6;
    uint32_t v24 = (uint16_t)*(const uint16_t *)(v23);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v25;
    v25 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v26 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v27 = v12 + 8;
    const uint8_t* v28 = v27 + 0;
    const uint8_t* v29 = (const uint8_t*) v28;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v30 = __riscv_vle8_v_u8m1(v29, v26);
    const uint8_t* v31 = v14 + 4;
    const uint8_t* v32 = v31 + 0;
    const int8_t* v33 = (const int8_t*) v32;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v34 = __riscv_vle8_v_i8m1(v33, v26);
    const uint8_t* v35 = v14 + 20;
    const uint8_t* v36 = v35 + 0;
    const int8_t* v37 = (const int8_t*) v36;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v38 = __riscv_vle8_v_i8m1(v37, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v39 = __riscv_vand_vx_u8m1(v30, 0x0F, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v40 = __riscv_vsrl_vx_u8m1(v30, 0x04, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v41 = __riscv_vid_v_u16m2(v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v42 = __riscv_vadd_vx_u16m2(v41, 0, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v43 = __riscv_vmv_v_x_u16m2(v22, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v44 = __riscv_vsrl_vv_u16m2(v43, v42, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v45 = __riscv_vand_vx_u16m2(v44, 0x1, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v46 = __riscv_vsll_vx_u16m2(v45, 0x4, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v47 = __riscv_vncvt_x_x_w_u8m1(v46, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v48 = __riscv_vid_v_u16m2(v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v49 = __riscv_vadd_vx_u16m2(v48, 0, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v50 = __riscv_vmv_v_x_u16m2(v24, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v51 = __riscv_vsrl_vv_u16m2(v50, v49, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v52 = __riscv_vand_vx_u16m2(v51, 0x1, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v53 = __riscv_vsll_vx_u16m2(v52, 0x4, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v54 = __riscv_vncvt_x_x_w_u8m1(v53, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v55 = __riscv_vor_vv_u8m1(v39, v47, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v56 = __riscv_vor_vv_u8m1(v40, v54, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v57 = __riscv_vreinterpret_v_u8m1_i8m1(v55);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v58 = __riscv_vreinterpret_v_u8m1_i8m1(v56);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v59 = __riscv_vwmul_vv_i16m2(v57, v34, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v60 = __riscv_vwmacc_vv_i16m2(v59, v58, v38, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v61 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v62 = __riscv_vwredsum_vs_i16m2_i32m1(v60, v61, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v63 = __riscv_vmv_x_s_i32m1_i32(v62);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v63;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v64 = v10 + 1;
    size_t v65 = v64 * 24;
    const uint8_t* v66 = v3 + v65;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v67 = v10 + 1;
    size_t v68 = v67 * 36;
    const uint8_t* v69 = v4 + v68;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v70 = (float)*(const _Float16 *)(v66);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v71 = (float)*(const _Float16 *)(v69);
    const uint8_t* v72 = v66 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v73 = (float)*(const _Float16 *)(v72);
    const uint8_t* v74 = v69 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v75 = (float)*(const _Float16 *)(v74);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v76 = v66 + 4;
    uint32_t v77 = (uint16_t)*(const uint16_t *)(v76);
    const uint8_t* v78 = v66 + 6;
    uint32_t v79 = (uint16_t)*(const uint16_t *)(v78);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v80;
    v80 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v81 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v82 = v66 + 8;
    const uint8_t* v83 = v82 + 0;
    const uint8_t* v84 = (const uint8_t*) v83;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v85 = __riscv_vle8_v_u8m1(v84, v81);
    const uint8_t* v86 = v69 + 4;
    const uint8_t* v87 = v86 + 0;
    const int8_t* v88 = (const int8_t*) v87;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v89 = __riscv_vle8_v_i8m1(v88, v81);
    const uint8_t* v90 = v69 + 20;
    const uint8_t* v91 = v90 + 0;
    const int8_t* v92 = (const int8_t*) v91;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v93 = __riscv_vle8_v_i8m1(v92, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v94 = __riscv_vand_vx_u8m1(v85, 0x0F, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v95 = __riscv_vsrl_vx_u8m1(v85, 0x04, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v96 = __riscv_vid_v_u16m2(v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v97 = __riscv_vadd_vx_u16m2(v96, 0, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v98 = __riscv_vmv_v_x_u16m2(v77, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v99 = __riscv_vsrl_vv_u16m2(v98, v97, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v100 = __riscv_vand_vx_u16m2(v99, 0x1, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v101 = __riscv_vsll_vx_u16m2(v100, 0x4, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v102 = __riscv_vncvt_x_x_w_u8m1(v101, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v103 = __riscv_vid_v_u16m2(v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v104 = __riscv_vadd_vx_u16m2(v103, 0, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v105 = __riscv_vmv_v_x_u16m2(v79, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v106 = __riscv_vsrl_vv_u16m2(v105, v104, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v107 = __riscv_vand_vx_u16m2(v106, 0x1, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v108 = __riscv_vsll_vx_u16m2(v107, 0x4, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v109 = __riscv_vncvt_x_x_w_u8m1(v108, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v110 = __riscv_vor_vv_u8m1(v94, v102, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v111 = __riscv_vor_vv_u8m1(v95, v109, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v112 = __riscv_vreinterpret_v_u8m1_i8m1(v110);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v113 = __riscv_vreinterpret_v_u8m1_i8m1(v111);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v114 = __riscv_vwmul_vv_i16m2(v112, v89, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v115 = __riscv_vwmacc_vv_i16m2(v114, v113, v93, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v116 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v117 = __riscv_vwredsum_vs_i16m2_i32m1(v115, v116, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v118 = __riscv_vmv_x_s_i32m1_i32(v117);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v80 = v118;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v119 = v10 + 2;
    size_t v120 = v119 * 24;
    const uint8_t* v121 = v3 + v120;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v122 = v10 + 2;
    size_t v123 = v122 * 36;
    const uint8_t* v124 = v4 + v123;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v125 = (float)*(const _Float16 *)(v121);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v126 = (float)*(const _Float16 *)(v124);
    const uint8_t* v127 = v121 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v128 = (float)*(const _Float16 *)(v127);
    const uint8_t* v129 = v124 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v130 = (float)*(const _Float16 *)(v129);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v131 = v121 + 4;
    uint32_t v132 = (uint16_t)*(const uint16_t *)(v131);
    const uint8_t* v133 = v121 + 6;
    uint32_t v134 = (uint16_t)*(const uint16_t *)(v133);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v135;
    v135 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v136 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v137 = v121 + 8;
    const uint8_t* v138 = v137 + 0;
    const uint8_t* v139 = (const uint8_t*) v138;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v140 = __riscv_vle8_v_u8m1(v139, v136);
    const uint8_t* v141 = v124 + 4;
    const uint8_t* v142 = v141 + 0;
    const int8_t* v143 = (const int8_t*) v142;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v144 = __riscv_vle8_v_i8m1(v143, v136);
    const uint8_t* v145 = v124 + 20;
    const uint8_t* v146 = v145 + 0;
    const int8_t* v147 = (const int8_t*) v146;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v148 = __riscv_vle8_v_i8m1(v147, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v149 = __riscv_vand_vx_u8m1(v140, 0x0F, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v150 = __riscv_vsrl_vx_u8m1(v140, 0x04, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v151 = __riscv_vid_v_u16m2(v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v152 = __riscv_vadd_vx_u16m2(v151, 0, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v153 = __riscv_vmv_v_x_u16m2(v132, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v154 = __riscv_vsrl_vv_u16m2(v153, v152, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v155 = __riscv_vand_vx_u16m2(v154, 0x1, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v156 = __riscv_vsll_vx_u16m2(v155, 0x4, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v157 = __riscv_vncvt_x_x_w_u8m1(v156, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v158 = __riscv_vid_v_u16m2(v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v159 = __riscv_vadd_vx_u16m2(v158, 0, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v160 = __riscv_vmv_v_x_u16m2(v134, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v161 = __riscv_vsrl_vv_u16m2(v160, v159, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v162 = __riscv_vand_vx_u16m2(v161, 0x1, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v163 = __riscv_vsll_vx_u16m2(v162, 0x4, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v164 = __riscv_vncvt_x_x_w_u8m1(v163, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v165 = __riscv_vor_vv_u8m1(v149, v157, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v166 = __riscv_vor_vv_u8m1(v150, v164, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v167 = __riscv_vreinterpret_v_u8m1_i8m1(v165);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v168 = __riscv_vreinterpret_v_u8m1_i8m1(v166);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v169 = __riscv_vwmul_vv_i16m2(v167, v144, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v170 = __riscv_vwmacc_vv_i16m2(v169, v168, v148, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v171 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v172 = __riscv_vwredsum_vs_i16m2_i32m1(v170, v171, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v173 = __riscv_vmv_x_s_i32m1_i32(v172);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v135 = v173;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v174 = v10 + 3;
    size_t v175 = v174 * 24;
    const uint8_t* v176 = v3 + v175;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v177 = v10 + 3;
    size_t v178 = v177 * 36;
    const uint8_t* v179 = v4 + v178;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v180 = (float)*(const _Float16 *)(v176);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v181 = (float)*(const _Float16 *)(v179);
    const uint8_t* v182 = v176 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v183 = (float)*(const _Float16 *)(v182);
    const uint8_t* v184 = v179 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v185 = (float)*(const _Float16 *)(v184);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v186 = v176 + 4;
    uint32_t v187 = (uint16_t)*(const uint16_t *)(v186);
    const uint8_t* v188 = v176 + 6;
    uint32_t v189 = (uint16_t)*(const uint16_t *)(v188);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v190;
    v190 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v191 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v192 = v176 + 8;
    const uint8_t* v193 = v192 + 0;
    const uint8_t* v194 = (const uint8_t*) v193;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v195 = __riscv_vle8_v_u8m1(v194, v191);
    const uint8_t* v196 = v179 + 4;
    const uint8_t* v197 = v196 + 0;
    const int8_t* v198 = (const int8_t*) v197;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v199 = __riscv_vle8_v_i8m1(v198, v191);
    const uint8_t* v200 = v179 + 20;
    const uint8_t* v201 = v200 + 0;
    const int8_t* v202 = (const int8_t*) v201;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v203 = __riscv_vle8_v_i8m1(v202, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v204 = __riscv_vand_vx_u8m1(v195, 0x0F, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v205 = __riscv_vsrl_vx_u8m1(v195, 0x04, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v206 = __riscv_vid_v_u16m2(v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v207 = __riscv_vadd_vx_u16m2(v206, 0, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v208 = __riscv_vmv_v_x_u16m2(v187, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v209 = __riscv_vsrl_vv_u16m2(v208, v207, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v210 = __riscv_vand_vx_u16m2(v209, 0x1, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v211 = __riscv_vsll_vx_u16m2(v210, 0x4, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v212 = __riscv_vncvt_x_x_w_u8m1(v211, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v213 = __riscv_vid_v_u16m2(v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v214 = __riscv_vadd_vx_u16m2(v213, 0, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v215 = __riscv_vmv_v_x_u16m2(v189, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v216 = __riscv_vsrl_vv_u16m2(v215, v214, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v217 = __riscv_vand_vx_u16m2(v216, 0x1, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v218 = __riscv_vsll_vx_u16m2(v217, 0x4, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v219 = __riscv_vncvt_x_x_w_u8m1(v218, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v220 = __riscv_vor_vv_u8m1(v204, v212, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v221 = __riscv_vor_vv_u8m1(v205, v219, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v222 = __riscv_vreinterpret_v_u8m1_i8m1(v220);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v223 = __riscv_vreinterpret_v_u8m1_i8m1(v221);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v224 = __riscv_vwmul_vv_i16m2(v222, v199, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v225 = __riscv_vwmacc_vv_i16m2(v224, v223, v203, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v226 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v227 = __riscv_vwredsum_vs_i16m2_i32m1(v225, v226, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v228 = __riscv_vmv_x_s_i32m1_i32(v227);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v190 = v228;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v229 = v25;
    float v230 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v230 + ((v15 * v16) * (float) v229 + v18 * v20);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v231 = v80;
    float v232 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v232 + ((v70 * v71) * (float) v231 + v73 * v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v233 = v135;
    float v234 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v234 + ((v125 * v126) * (float) v233 + v128 * v130);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v235 = v190;
    float v236 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v236 + ((v180 * v181) * (float) v235 + v183 * v185);
  }
  for (size_t v237 = v9; v237 < v7; v237 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v238 = v237 * 24;
    const uint8_t* v239 = v3 + v238;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v240 = v237 * 36;
    const uint8_t* v241 = v4 + v240;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v242 = (float)*(const _Float16 *)(v239);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v243 = (float)*(const _Float16 *)(v241);
    const uint8_t* v244 = v239 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v245 = (float)*(const _Float16 *)(v244);
    const uint8_t* v246 = v241 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v247 = (float)*(const _Float16 *)(v246);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v248 = v239 + 4;
    uint32_t v249 = (uint16_t)*(const uint16_t *)(v248);
    const uint8_t* v250 = v239 + 6;
    uint32_t v251 = (uint16_t)*(const uint16_t *)(v250);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v252;
    v252 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v253 = __riscv_vsetvl_e8m1(16);
    for (size_t v254 = 0; v254 < 16; v254 += v253) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v255 = 16 - v254;
      size_t v256 = __riscv_vsetvl_e8m1(v255);
      const uint8_t* v257 = v239 + 8;
      const uint8_t* v258 = v257 + v254;
      const uint8_t* v259 = (const uint8_t*) v258;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v260 = __riscv_vle8_v_u8m1(v259, v256);
      const uint8_t* v261 = v241 + 4;
      const uint8_t* v262 = v261 + v254;
      const int8_t* v263 = (const int8_t*) v262;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v264 = __riscv_vle8_v_i8m1(v263, v256);
      const uint8_t* v265 = v241 + 20;
      const uint8_t* v266 = v265 + v254;
      const int8_t* v267 = (const int8_t*) v266;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v268 = __riscv_vle8_v_i8m1(v267, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v269 = __riscv_vand_vx_u8m1(v260, 0x0F, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v270 = __riscv_vsrl_vx_u8m1(v260, 0x04, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v271 = __riscv_vid_v_u16m2(v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v272 = __riscv_vadd_vx_u16m2(v271, v254, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v273 = __riscv_vmv_v_x_u16m2(v249, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v274 = __riscv_vsrl_vv_u16m2(v273, v272, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v275 = __riscv_vand_vx_u16m2(v274, 0x1, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v276 = __riscv_vsll_vx_u16m2(v275, 0x4, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v277 = __riscv_vncvt_x_x_w_u8m1(v276, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v278 = __riscv_vid_v_u16m2(v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v279 = __riscv_vadd_vx_u16m2(v278, v254, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v280 = __riscv_vmv_v_x_u16m2(v251, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v281 = __riscv_vsrl_vv_u16m2(v280, v279, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v282 = __riscv_vand_vx_u16m2(v281, 0x1, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v283 = __riscv_vsll_vx_u16m2(v282, 0x4, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v284 = __riscv_vncvt_x_x_w_u8m1(v283, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v285 = __riscv_vor_vv_u8m1(v269, v277, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v286 = __riscv_vor_vv_u8m1(v270, v284, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v287 = __riscv_vreinterpret_v_u8m1_i8m1(v285);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v288 = __riscv_vreinterpret_v_u8m1_i8m1(v286);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v289 = __riscv_vwmul_vv_i16m2(v287, v264, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v290 = __riscv_vwmacc_vv_i16m2(v289, v288, v268, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v291 = v252;
      vint32m1_t v292 = __riscv_vmv_v_x_i32m1(v291, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v293 = __riscv_vwredsum_vs_i16m2_i32m1(v290, v292, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v294 = __riscv_vmv_x_s_i32m1_i32(v293);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v252 = v294;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v295 = v252;
    float v296 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v296 + ((v242 * v243) * (float) v295 + v245 * v247);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v297 = v6;
  v2[0] = v297;
  return;
}


