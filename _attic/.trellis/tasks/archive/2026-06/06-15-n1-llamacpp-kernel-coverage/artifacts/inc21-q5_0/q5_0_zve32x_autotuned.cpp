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
  size_t v8 = v7 % 2;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 2) {
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
    for (size_t v23 = 0; v23 < 16; v23 += v22) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v24 = 16 - v23;
      size_t v25 = __riscv_vsetvl_e8m1(v24);
      const uint8_t* v26 = v12 + 6;
      const uint8_t* v27 = v26 + v23;
      const uint8_t* v28 = (const uint8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v29 = __riscv_vle8_v_u8m1(v28, v25);
      const uint8_t* v30 = v14 + 2;
      const uint8_t* v31 = v30 + v23;
      const int8_t* v32 = (const int8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v33 = __riscv_vle8_v_i8m1(v32, v25);
      const uint8_t* v34 = v14 + 18;
      const uint8_t* v35 = v34 + v23;
      const int8_t* v36 = (const int8_t*) v35;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v37 = __riscv_vle8_v_i8m1(v36, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v38 = __riscv_vand_vx_u8m1(v29, 0x0F, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v39 = __riscv_vsrl_vx_u8m1(v29, 0x04, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v40 = __riscv_vid_v_u16m2(v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v41 = __riscv_vadd_vx_u16m2(v40, v23, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v42 = __riscv_vmv_v_x_u16m2(v18, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v43 = __riscv_vsrl_vv_u16m2(v42, v41, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v44 = __riscv_vand_vx_u16m2(v43, 0x1, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v45 = __riscv_vsll_vx_u16m2(v44, 0x4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v46 = __riscv_vncvt_x_x_w_u8m1(v45, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v47 = __riscv_vid_v_u16m2(v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v48 = __riscv_vadd_vx_u16m2(v47, v23, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v49 = __riscv_vmv_v_x_u16m2(v20, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v50 = __riscv_vsrl_vv_u16m2(v49, v48, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v51 = __riscv_vand_vx_u16m2(v50, 0x1, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v52 = __riscv_vsll_vx_u16m2(v51, 0x4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v53 = __riscv_vncvt_x_x_w_u8m1(v52, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v54 = __riscv_vor_vv_u8m1(v38, v46, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v55 = __riscv_vor_vv_u8m1(v39, v53, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v56 = __riscv_vreinterpret_v_u8m1_i8m1(v54);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v57 = __riscv_vsub_vx_i8m1(v56, 16, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v58 = __riscv_vreinterpret_v_u8m1_i8m1(v55);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v59 = __riscv_vsub_vx_i8m1(v58, 16, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v60 = __riscv_vwmul_vv_i16m2(v57, v33, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v61 = __riscv_vwmacc_vv_i16m2(v60, v59, v37, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v62 = v21;
      vint32m1_t v63 = __riscv_vmv_v_x_i32m1(v62, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v64 = __riscv_vwredsum_vs_i16m2_i32m1(v61, v63, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v65 = __riscv_vmv_x_s_i32m1_i32(v64);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v21 = v65;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v66 = v10 + 1;
    size_t v67 = v66 * 22;
    const uint8_t* v68 = v3 + v67;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v69 = v10 + 1;
    size_t v70 = v69 * 34;
    const uint8_t* v71 = v4 + v70;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v72 = (float)*(const _Float16 *)(v68);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v73 = (float)*(const _Float16 *)(v71);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v74 = v68 + 2;
    uint32_t v75 = (uint16_t)*(const uint16_t *)(v74);
    const uint8_t* v76 = v68 + 4;
    uint32_t v77 = (uint16_t)*(const uint16_t *)(v76);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v78;
    v78 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v79 = __riscv_vsetvl_e8m1(16);
    for (size_t v80 = 0; v80 < 16; v80 += v79) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v81 = 16 - v80;
      size_t v82 = __riscv_vsetvl_e8m1(v81);
      const uint8_t* v83 = v68 + 6;
      const uint8_t* v84 = v83 + v80;
      const uint8_t* v85 = (const uint8_t*) v84;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v86 = __riscv_vle8_v_u8m1(v85, v82);
      const uint8_t* v87 = v71 + 2;
      const uint8_t* v88 = v87 + v80;
      const int8_t* v89 = (const int8_t*) v88;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v90 = __riscv_vle8_v_i8m1(v89, v82);
      const uint8_t* v91 = v71 + 18;
      const uint8_t* v92 = v91 + v80;
      const int8_t* v93 = (const int8_t*) v92;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v94 = __riscv_vle8_v_i8m1(v93, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v95 = __riscv_vand_vx_u8m1(v86, 0x0F, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v96 = __riscv_vsrl_vx_u8m1(v86, 0x04, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v97 = __riscv_vid_v_u16m2(v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v98 = __riscv_vadd_vx_u16m2(v97, v80, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v99 = __riscv_vmv_v_x_u16m2(v75, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v100 = __riscv_vsrl_vv_u16m2(v99, v98, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v101 = __riscv_vand_vx_u16m2(v100, 0x1, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v102 = __riscv_vsll_vx_u16m2(v101, 0x4, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v103 = __riscv_vncvt_x_x_w_u8m1(v102, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v104 = __riscv_vid_v_u16m2(v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v105 = __riscv_vadd_vx_u16m2(v104, v80, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v106 = __riscv_vmv_v_x_u16m2(v77, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v107 = __riscv_vsrl_vv_u16m2(v106, v105, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v108 = __riscv_vand_vx_u16m2(v107, 0x1, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v109 = __riscv_vsll_vx_u16m2(v108, 0x4, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v110 = __riscv_vncvt_x_x_w_u8m1(v109, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v111 = __riscv_vor_vv_u8m1(v95, v103, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v112 = __riscv_vor_vv_u8m1(v96, v110, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v113 = __riscv_vreinterpret_v_u8m1_i8m1(v111);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v114 = __riscv_vsub_vx_i8m1(v113, 16, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v115 = __riscv_vreinterpret_v_u8m1_i8m1(v112);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v116 = __riscv_vsub_vx_i8m1(v115, 16, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v117 = __riscv_vwmul_vv_i16m2(v114, v90, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v118 = __riscv_vwmacc_vv_i16m2(v117, v116, v94, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v119 = v78;
      vint32m1_t v120 = __riscv_vmv_v_x_i32m1(v119, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v121 = __riscv_vwredsum_vs_i16m2_i32m1(v118, v120, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v122 = __riscv_vmv_x_s_i32m1_i32(v121);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v78 = v122;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v123 = v21;
    float v124 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v124 + (v15 * v16) * (float) v123;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v125 = v78;
    float v126 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v126 + (v72 * v73) * (float) v125;
  }
  for (size_t v127 = v9; v127 < v7; v127 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v128 = v127 * 22;
    const uint8_t* v129 = v3 + v128;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v130 = v127 * 34;
    const uint8_t* v131 = v4 + v130;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v132 = (float)*(const _Float16 *)(v129);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v133 = (float)*(const _Float16 *)(v131);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v134 = v129 + 2;
    uint32_t v135 = (uint16_t)*(const uint16_t *)(v134);
    const uint8_t* v136 = v129 + 4;
    uint32_t v137 = (uint16_t)*(const uint16_t *)(v136);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v138;
    v138 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v139 = __riscv_vsetvl_e8m1(16);
    for (size_t v140 = 0; v140 < 16; v140 += v139) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v141 = 16 - v140;
      size_t v142 = __riscv_vsetvl_e8m1(v141);
      const uint8_t* v143 = v129 + 6;
      const uint8_t* v144 = v143 + v140;
      const uint8_t* v145 = (const uint8_t*) v144;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v146 = __riscv_vle8_v_u8m1(v145, v142);
      const uint8_t* v147 = v131 + 2;
      const uint8_t* v148 = v147 + v140;
      const int8_t* v149 = (const int8_t*) v148;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v150 = __riscv_vle8_v_i8m1(v149, v142);
      const uint8_t* v151 = v131 + 18;
      const uint8_t* v152 = v151 + v140;
      const int8_t* v153 = (const int8_t*) v152;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v154 = __riscv_vle8_v_i8m1(v153, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v155 = __riscv_vand_vx_u8m1(v146, 0x0F, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v156 = __riscv_vsrl_vx_u8m1(v146, 0x04, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v157 = __riscv_vid_v_u16m2(v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v158 = __riscv_vadd_vx_u16m2(v157, v140, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v159 = __riscv_vmv_v_x_u16m2(v135, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v160 = __riscv_vsrl_vv_u16m2(v159, v158, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v161 = __riscv_vand_vx_u16m2(v160, 0x1, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v162 = __riscv_vsll_vx_u16m2(v161, 0x4, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v163 = __riscv_vncvt_x_x_w_u8m1(v162, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v164 = __riscv_vid_v_u16m2(v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v165 = __riscv_vadd_vx_u16m2(v164, v140, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v166 = __riscv_vmv_v_x_u16m2(v137, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v167 = __riscv_vsrl_vv_u16m2(v166, v165, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v168 = __riscv_vand_vx_u16m2(v167, 0x1, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v169 = __riscv_vsll_vx_u16m2(v168, 0x4, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v170 = __riscv_vncvt_x_x_w_u8m1(v169, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v171 = __riscv_vor_vv_u8m1(v155, v163, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v172 = __riscv_vor_vv_u8m1(v156, v170, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v173 = __riscv_vreinterpret_v_u8m1_i8m1(v171);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v174 = __riscv_vsub_vx_i8m1(v173, 16, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v175 = __riscv_vreinterpret_v_u8m1_i8m1(v172);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v176 = __riscv_vsub_vx_i8m1(v175, 16, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v177 = __riscv_vwmul_vv_i16m2(v174, v150, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v178 = __riscv_vwmacc_vv_i16m2(v177, v176, v154, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v179 = v138;
      vint32m1_t v180 = __riscv_vmv_v_x_i32m1(v179, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v181 = __riscv_vwredsum_vs_i16m2_i32m1(v178, v180, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v182 = __riscv_vmv_x_s_i32m1_i32(v181);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v138 = v182;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v183 = v138;
    float v184 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v184 + (v132 * v133) * (float) v183;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v185 = v6;
  v2[0] = v185;
  return;
}


