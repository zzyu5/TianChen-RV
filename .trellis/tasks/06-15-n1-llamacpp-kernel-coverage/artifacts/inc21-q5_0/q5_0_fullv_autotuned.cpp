#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 4;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 22;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v17 = v12 + 2;
    uint32_t v18 = (uint16_t)*(const uint16_t *)(v17);
    const uint8_t* v19 = v12 + 4;
    uint32_t v20 = (uint16_t)*(const uint16_t *)(v19);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v21;
    v21 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v22 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v23 = v12 + 6;
    const uint8_t* v24 = v23 + 0;
    const uint8_t* v25 = (const uint8_t*) v24;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v26 = __riscv_vle8_v_u8m1(v25, v22);
    const uint8_t* v27 = v14 + 2;
    const uint8_t* v28 = v27 + 0;
    const int8_t* v29 = (const int8_t*) v28;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v30 = __riscv_vle8_v_i8m1(v29, v22);
    const uint8_t* v31 = v14 + 18;
    const uint8_t* v32 = v31 + 0;
    const int8_t* v33 = (const int8_t*) v32;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v34 = __riscv_vle8_v_i8m1(v33, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v35 = __riscv_vand_vx_u8m1(v26, 0x0F, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v36 = __riscv_vsrl_vx_u8m1(v26, 0x04, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v37 = __riscv_vid_v_u16m2(v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v38 = __riscv_vadd_vx_u16m2(v37, 0, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v39 = __riscv_vmv_v_x_u16m2(v18, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v40 = __riscv_vsrl_vv_u16m2(v39, v38, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v41 = __riscv_vand_vx_u16m2(v40, 0x1, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v42 = __riscv_vsll_vx_u16m2(v41, 0x4, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v43 = __riscv_vncvt_x_x_w_u8m1(v42, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v44 = __riscv_vid_v_u16m2(v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v45 = __riscv_vadd_vx_u16m2(v44, 0, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v46 = __riscv_vmv_v_x_u16m2(v20, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v47 = __riscv_vsrl_vv_u16m2(v46, v45, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v48 = __riscv_vand_vx_u16m2(v47, 0x1, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v49 = __riscv_vsll_vx_u16m2(v48, 0x4, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v50 = __riscv_vncvt_x_x_w_u8m1(v49, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v51 = __riscv_vor_vv_u8m1(v35, v43, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v52 = __riscv_vor_vv_u8m1(v36, v50, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v53 = __riscv_vreinterpret_v_u8m1_i8m1(v51);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
    vint8m1_t v54 = __riscv_vsub_vx_i8m1(v53, 16, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v55 = __riscv_vreinterpret_v_u8m1_i8m1(v52);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
    vint8m1_t v56 = __riscv_vsub_vx_i8m1(v55, 16, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v57 = __riscv_vwmul_vv_i16m2(v54, v30, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v58 = __riscv_vwmacc_vv_i16m2(v57, v56, v34, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v59 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v60 = __riscv_vwredsum_vs_i16m2_i32m1(v58, v59, v22);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v61 = __riscv_vmv_x_s_i32m1_i32(v60);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v21 = v61;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v62 = v10 + 1;
    size_t v63 = v62 * 22;
    const uint8_t* v64 = v3 + v63;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v65 = v10 + 1;
    size_t v66 = v65 * 34;
    const uint8_t* v67 = v4 + v66;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v68 = (float)*(const _Float16 *)(v64);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v69 = (float)*(const _Float16 *)(v67);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v70 = v64 + 2;
    uint32_t v71 = (uint16_t)*(const uint16_t *)(v70);
    const uint8_t* v72 = v64 + 4;
    uint32_t v73 = (uint16_t)*(const uint16_t *)(v72);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v74;
    v74 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v75 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v76 = v64 + 6;
    const uint8_t* v77 = v76 + 0;
    const uint8_t* v78 = (const uint8_t*) v77;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v79 = __riscv_vle8_v_u8m1(v78, v75);
    const uint8_t* v80 = v67 + 2;
    const uint8_t* v81 = v80 + 0;
    const int8_t* v82 = (const int8_t*) v81;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v83 = __riscv_vle8_v_i8m1(v82, v75);
    const uint8_t* v84 = v67 + 18;
    const uint8_t* v85 = v84 + 0;
    const int8_t* v86 = (const int8_t*) v85;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v87 = __riscv_vle8_v_i8m1(v86, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v88 = __riscv_vand_vx_u8m1(v79, 0x0F, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v89 = __riscv_vsrl_vx_u8m1(v79, 0x04, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v90 = __riscv_vid_v_u16m2(v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v91 = __riscv_vadd_vx_u16m2(v90, 0, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v92 = __riscv_vmv_v_x_u16m2(v71, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v93 = __riscv_vsrl_vv_u16m2(v92, v91, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v94 = __riscv_vand_vx_u16m2(v93, 0x1, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v95 = __riscv_vsll_vx_u16m2(v94, 0x4, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v96 = __riscv_vncvt_x_x_w_u8m1(v95, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v97 = __riscv_vid_v_u16m2(v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v98 = __riscv_vadd_vx_u16m2(v97, 0, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v99 = __riscv_vmv_v_x_u16m2(v73, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v100 = __riscv_vsrl_vv_u16m2(v99, v98, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v101 = __riscv_vand_vx_u16m2(v100, 0x1, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v102 = __riscv_vsll_vx_u16m2(v101, 0x4, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v103 = __riscv_vncvt_x_x_w_u8m1(v102, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v104 = __riscv_vor_vv_u8m1(v88, v96, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v105 = __riscv_vor_vv_u8m1(v89, v103, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v106 = __riscv_vreinterpret_v_u8m1_i8m1(v104);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
    vint8m1_t v107 = __riscv_vsub_vx_i8m1(v106, 16, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v108 = __riscv_vreinterpret_v_u8m1_i8m1(v105);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
    vint8m1_t v109 = __riscv_vsub_vx_i8m1(v108, 16, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v110 = __riscv_vwmul_vv_i16m2(v107, v83, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v111 = __riscv_vwmacc_vv_i16m2(v110, v109, v87, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v112 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v113 = __riscv_vwredsum_vs_i16m2_i32m1(v111, v112, v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v114 = __riscv_vmv_x_s_i32m1_i32(v113);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v74 = v114;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v115 = v10 + 2;
    size_t v116 = v115 * 22;
    const uint8_t* v117 = v3 + v116;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v118 = v10 + 2;
    size_t v119 = v118 * 34;
    const uint8_t* v120 = v4 + v119;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v121 = (float)*(const _Float16 *)(v117);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v122 = (float)*(const _Float16 *)(v120);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v123 = v117 + 2;
    uint32_t v124 = (uint16_t)*(const uint16_t *)(v123);
    const uint8_t* v125 = v117 + 4;
    uint32_t v126 = (uint16_t)*(const uint16_t *)(v125);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v127;
    v127 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v128 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v129 = v117 + 6;
    const uint8_t* v130 = v129 + 0;
    const uint8_t* v131 = (const uint8_t*) v130;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v132 = __riscv_vle8_v_u8m1(v131, v128);
    const uint8_t* v133 = v120 + 2;
    const uint8_t* v134 = v133 + 0;
    const int8_t* v135 = (const int8_t*) v134;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v136 = __riscv_vle8_v_i8m1(v135, v128);
    const uint8_t* v137 = v120 + 18;
    const uint8_t* v138 = v137 + 0;
    const int8_t* v139 = (const int8_t*) v138;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v140 = __riscv_vle8_v_i8m1(v139, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v141 = __riscv_vand_vx_u8m1(v132, 0x0F, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v142 = __riscv_vsrl_vx_u8m1(v132, 0x04, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v143 = __riscv_vid_v_u16m2(v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v144 = __riscv_vadd_vx_u16m2(v143, 0, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v145 = __riscv_vmv_v_x_u16m2(v124, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v146 = __riscv_vsrl_vv_u16m2(v145, v144, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v147 = __riscv_vand_vx_u16m2(v146, 0x1, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v148 = __riscv_vsll_vx_u16m2(v147, 0x4, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v149 = __riscv_vncvt_x_x_w_u8m1(v148, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v150 = __riscv_vid_v_u16m2(v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v151 = __riscv_vadd_vx_u16m2(v150, 0, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v152 = __riscv_vmv_v_x_u16m2(v126, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v153 = __riscv_vsrl_vv_u16m2(v152, v151, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v154 = __riscv_vand_vx_u16m2(v153, 0x1, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v155 = __riscv_vsll_vx_u16m2(v154, 0x4, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v156 = __riscv_vncvt_x_x_w_u8m1(v155, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v157 = __riscv_vor_vv_u8m1(v141, v149, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v158 = __riscv_vor_vv_u8m1(v142, v156, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v159 = __riscv_vreinterpret_v_u8m1_i8m1(v157);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
    vint8m1_t v160 = __riscv_vsub_vx_i8m1(v159, 16, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v161 = __riscv_vreinterpret_v_u8m1_i8m1(v158);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
    vint8m1_t v162 = __riscv_vsub_vx_i8m1(v161, 16, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v163 = __riscv_vwmul_vv_i16m2(v160, v136, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v164 = __riscv_vwmacc_vv_i16m2(v163, v162, v140, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v165 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v166 = __riscv_vwredsum_vs_i16m2_i32m1(v164, v165, v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v167 = __riscv_vmv_x_s_i32m1_i32(v166);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v127 = v167;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v168 = v10 + 3;
    size_t v169 = v168 * 22;
    const uint8_t* v170 = v3 + v169;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v171 = v10 + 3;
    size_t v172 = v171 * 34;
    const uint8_t* v173 = v4 + v172;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v174 = (float)*(const _Float16 *)(v170);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v175 = (float)*(const _Float16 *)(v173);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v176 = v170 + 2;
    uint32_t v177 = (uint16_t)*(const uint16_t *)(v176);
    const uint8_t* v178 = v170 + 4;
    uint32_t v179 = (uint16_t)*(const uint16_t *)(v178);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v180;
    v180 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v181 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v182 = v170 + 6;
    const uint8_t* v183 = v182 + 0;
    const uint8_t* v184 = (const uint8_t*) v183;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v185 = __riscv_vle8_v_u8m1(v184, v181);
    const uint8_t* v186 = v173 + 2;
    const uint8_t* v187 = v186 + 0;
    const int8_t* v188 = (const int8_t*) v187;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v189 = __riscv_vle8_v_i8m1(v188, v181);
    const uint8_t* v190 = v173 + 18;
    const uint8_t* v191 = v190 + 0;
    const int8_t* v192 = (const int8_t*) v191;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v193 = __riscv_vle8_v_i8m1(v192, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v194 = __riscv_vand_vx_u8m1(v185, 0x0F, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v195 = __riscv_vsrl_vx_u8m1(v185, 0x04, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v196 = __riscv_vid_v_u16m2(v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v197 = __riscv_vadd_vx_u16m2(v196, 0, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v198 = __riscv_vmv_v_x_u16m2(v177, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v199 = __riscv_vsrl_vv_u16m2(v198, v197, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v200 = __riscv_vand_vx_u16m2(v199, 0x1, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v201 = __riscv_vsll_vx_u16m2(v200, 0x4, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v202 = __riscv_vncvt_x_x_w_u8m1(v201, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v203 = __riscv_vid_v_u16m2(v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v204 = __riscv_vadd_vx_u16m2(v203, 0, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v205 = __riscv_vmv_v_x_u16m2(v179, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v206 = __riscv_vsrl_vv_u16m2(v205, v204, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v207 = __riscv_vand_vx_u16m2(v206, 0x1, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v208 = __riscv_vsll_vx_u16m2(v207, 0x4, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v209 = __riscv_vncvt_x_x_w_u8m1(v208, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v210 = __riscv_vor_vv_u8m1(v194, v202, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v211 = __riscv_vor_vv_u8m1(v195, v209, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v212 = __riscv_vreinterpret_v_u8m1_i8m1(v210);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
    vint8m1_t v213 = __riscv_vsub_vx_i8m1(v212, 16, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v214 = __riscv_vreinterpret_v_u8m1_i8m1(v211);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
    vint8m1_t v215 = __riscv_vsub_vx_i8m1(v214, 16, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v216 = __riscv_vwmul_vv_i16m2(v213, v189, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v217 = __riscv_vwmacc_vv_i16m2(v216, v215, v193, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v218 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v219 = __riscv_vwredsum_vs_i16m2_i32m1(v217, v218, v181);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v220 = __riscv_vmv_x_s_i32m1_i32(v219);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v180 = v220;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v221 = v21;
    float v222 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v222 + (v15 * v16) * (float) v221;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v223 = v74;
    float v224 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v224 + (v68 * v69) * (float) v223;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v225 = v127;
    float v226 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v226 + (v121 * v122) * (float) v225;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v227 = v180;
    float v228 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v228 + (v174 * v175) * (float) v227;
  }
  for (size_t v229 = v9; v229 < v7; v229 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v230 = v229 * 22;
    const uint8_t* v231 = v3 + v230;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v232 = v229 * 34;
    const uint8_t* v233 = v4 + v232;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v234 = (float)*(const _Float16 *)(v231);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v235 = (float)*(const _Float16 *)(v233);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v236 = v231 + 2;
    uint32_t v237 = (uint16_t)*(const uint16_t *)(v236);
    const uint8_t* v238 = v231 + 4;
    uint32_t v239 = (uint16_t)*(const uint16_t *)(v238);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v240;
    v240 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v241 = __riscv_vsetvl_e8m1(16);
    for (size_t v242 = 0; v242 < 16; v242 += v241) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v243 = 16 - v242;
      size_t v244 = __riscv_vsetvl_e8m1(v243);
      const uint8_t* v245 = v231 + 6;
      const uint8_t* v246 = v245 + v242;
      const uint8_t* v247 = (const uint8_t*) v246;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v248 = __riscv_vle8_v_u8m1(v247, v244);
      const uint8_t* v249 = v233 + 2;
      const uint8_t* v250 = v249 + v242;
      const int8_t* v251 = (const int8_t*) v250;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v252 = __riscv_vle8_v_i8m1(v251, v244);
      const uint8_t* v253 = v233 + 18;
      const uint8_t* v254 = v253 + v242;
      const int8_t* v255 = (const int8_t*) v254;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v256 = __riscv_vle8_v_i8m1(v255, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v257 = __riscv_vand_vx_u8m1(v248, 0x0F, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v258 = __riscv_vsrl_vx_u8m1(v248, 0x04, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v259 = __riscv_vid_v_u16m2(v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v260 = __riscv_vadd_vx_u16m2(v259, v242, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v261 = __riscv_vmv_v_x_u16m2(v237, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v262 = __riscv_vsrl_vv_u16m2(v261, v260, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v263 = __riscv_vand_vx_u16m2(v262, 0x1, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v264 = __riscv_vsll_vx_u16m2(v263, 0x4, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v265 = __riscv_vncvt_x_x_w_u8m1(v264, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v266 = __riscv_vid_v_u16m2(v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v267 = __riscv_vadd_vx_u16m2(v266, v242, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v268 = __riscv_vmv_v_x_u16m2(v239, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v269 = __riscv_vsrl_vv_u16m2(v268, v267, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v270 = __riscv_vand_vx_u16m2(v269, 0x1, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v271 = __riscv_vsll_vx_u16m2(v270, 0x4, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v272 = __riscv_vncvt_x_x_w_u8m1(v271, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v273 = __riscv_vor_vv_u8m1(v257, v265, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v274 = __riscv_vor_vv_u8m1(v258, v272, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v275 = __riscv_vreinterpret_v_u8m1_i8m1(v273);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v276 = __riscv_vsub_vx_i8m1(v275, 16, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v277 = __riscv_vreinterpret_v_u8m1_i8m1(v274);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v278 = __riscv_vsub_vx_i8m1(v277, 16, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v279 = __riscv_vwmul_vv_i16m2(v276, v252, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v280 = __riscv_vwmacc_vv_i16m2(v279, v278, v256, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v281 = v240;
      vint32m1_t v282 = __riscv_vmv_v_x_i32m1(v281, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v283 = __riscv_vwredsum_vs_i16m2_i32m1(v280, v282, v244);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v284 = __riscv_vmv_x_s_i32m1_i32(v283);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v240 = v284;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v285 = v240;
    float v286 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v286 + (v234 * v235) * (float) v285;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v287 = v6;
  v2[0] = v287;
  return;
}


