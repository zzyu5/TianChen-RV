#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_repack_gemv_q5_0_q8_0_kernel_ggml_repack_gemv_q5_0_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count
  size_t v8 = v5 / 16;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
    size_t v10 = v9 * v7;
    size_t v11 = v10 * 352;
    const uint8_t* v12 = v3 + v11;
    vfloat32m2_t v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v14 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v13 = v14;
    vfloat32m2_t v15;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v16 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v15 = v16;
    for (size_t v17 = 0; v17 < v7; v17 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v18 = v17 * 352;
      const uint8_t* v19 = v12 + v18;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v20 = v17 * 34;
      const uint8_t* v21 = v4 + v20;
      vint16m1_t v22;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v23 = __riscv_vmv_v_x_i16m1(0, 8);
      v22 = v23;
      vint16m1_t v24;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v25 = __riscv_vmv_v_x_i16m1(0, 8);
      v24 = v25;
      vint16m1_t v26;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v27 = __riscv_vmv_v_x_i16m1(0, 8);
      v26 = v27;
      vint16m1_t v28;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v29 = __riscv_vmv_v_x_i16m1(0, 8);
      v28 = v29;
      for (size_t v30 = 0; v30 < 16; v30 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v31 = v30 * 16;
        size_t v32 = 32 + v31;
        size_t v33 = v32 + 8;
        const uint8_t* v34 = v19 + v32;
        const uint8_t* v35 = (const uint8_t*) v34;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v36 = __riscv_vle8_v_u8mf2(v35, 8);
        const uint8_t* v37 = v19 + v33;
        const uint8_t* v38 = (const uint8_t*) v37;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v39 = __riscv_vle8_v_u8mf2(v38, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_lo_addr
        size_t v40 = v30 * 2;
        size_t v41 = 288 + v40;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_hi_addr
        size_t v42 = 288 + 32;
        size_t v43 = v42 + v40;
        const uint8_t* v44 = v19 + v41;
        const uint16_t* v45 = (const uint16_t*) v44;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_mask_scalar
        int32_t v46 = (uint16_t)*(const uint16_t *)(v45);
        const uint8_t* v47 = v19 + v43;
        const uint16_t* v48 = (const uint16_t*) v47;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_mask_scalar
        int32_t v49 = (uint16_t)*(const uint16_t *)(v48);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
        vuint16m1_t v50 = __riscv_vmv_v_x_u16m1(v46, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
        vuint16m1_t v51 = __riscv_vid_v_u16m1(8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
        vuint16m1_t v52 = __riscv_vsrl_vv_u16m1(v50, v51, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
        vuint16m1_t v53 = __riscv_vand_vx_u16m1(v52, 1, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
        vuint16m1_t v54 = __riscv_vsll_vx_u16m1(v53, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
        vuint8mf2_t v55 = __riscv_vncvt_x_x_w_u8mf2(v54, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v56 = __riscv_vand_vx_u8mf2(v36, 15, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v57 = __riscv_vor_vv_u8mf2(v56, v55, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v58 = __riscv_vreinterpret_v_u8mf2_i8mf2(v57);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v59 = __riscv_vsub_vx_i8mf2(v58, 16, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
        vuint16m1_t v60 = __riscv_vmv_v_x_u16m1(v49, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
        vuint16m1_t v61 = __riscv_vid_v_u16m1(8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
        vuint16m1_t v62 = __riscv_vsrl_vv_u16m1(v60, v61, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
        vuint16m1_t v63 = __riscv_vand_vx_u16m1(v62, 1, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
        vuint16m1_t v64 = __riscv_vsll_vx_u16m1(v63, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
        vuint8mf2_t v65 = __riscv_vncvt_x_x_w_u8mf2(v64, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v66 = __riscv_vsrl_vx_u8mf2(v36, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v67 = __riscv_vor_vv_u8mf2(v66, v65, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v68 = __riscv_vreinterpret_v_u8mf2_i8mf2(v67);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v69 = __riscv_vsub_vx_i8mf2(v68, 16, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
        vuint16m1_t v70 = __riscv_vmv_v_x_u16m1(v46, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
        vuint16m1_t v71 = __riscv_vid_v_u16m1(8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m1
        vuint16m1_t v72 = __riscv_vadd_vx_u16m1(v71, 8, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
        vuint16m1_t v73 = __riscv_vsrl_vv_u16m1(v70, v72, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
        vuint16m1_t v74 = __riscv_vand_vx_u16m1(v73, 1, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
        vuint16m1_t v75 = __riscv_vsll_vx_u16m1(v74, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
        vuint8mf2_t v76 = __riscv_vncvt_x_x_w_u8mf2(v75, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v77 = __riscv_vand_vx_u8mf2(v39, 15, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v78 = __riscv_vor_vv_u8mf2(v77, v76, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v79 = __riscv_vreinterpret_v_u8mf2_i8mf2(v78);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v80 = __riscv_vsub_vx_i8mf2(v79, 16, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
        vuint16m1_t v81 = __riscv_vmv_v_x_u16m1(v49, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
        vuint16m1_t v82 = __riscv_vid_v_u16m1(8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m1
        vuint16m1_t v83 = __riscv_vadd_vx_u16m1(v82, 8, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
        vuint16m1_t v84 = __riscv_vsrl_vv_u16m1(v81, v83, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
        vuint16m1_t v85 = __riscv_vand_vx_u16m1(v84, 1, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
        vuint16m1_t v86 = __riscv_vsll_vx_u16m1(v85, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
        vuint8mf2_t v87 = __riscv_vncvt_x_x_w_u8mf2(v86, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v88 = __riscv_vsrl_vx_u8mf2(v39, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v89 = __riscv_vor_vv_u8mf2(v88, v87, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v90 = __riscv_vreinterpret_v_u8mf2_i8mf2(v89);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v91 = __riscv_vsub_vx_i8mf2(v90, 16, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v92 = 2 + v30;
        const uint8_t* v93 = v21 + v92;
        const int8_t* v94 = (const int8_t*) v93;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v95 = *(const int8_t *)(v94);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v96 = 2 + 16;
        size_t v97 = v96 + v30;
        const uint8_t* v98 = v21 + v97;
        const int8_t* v99 = (const int8_t*) v98;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v100 = *(const int8_t *)(v99);
        vint16m1_t v101 = v22;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v102 = __riscv_vwmacc_vx_i16m1(v101, v95, v59, 8);
        v22 = v102;
        vint16m1_t v103 = v24;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v104 = __riscv_vwmacc_vx_i16m1(v103, v100, v69, 8);
        v24 = v104;
        vint16m1_t v105 = v26;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v106 = __riscv_vwmacc_vx_i16m1(v105, v95, v80, 8);
        v26 = v106;
        vint16m1_t v107 = v28;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v108 = __riscv_vwmacc_vx_i16m1(v107, v100, v91, 8);
        v28 = v108;
      }
      vint16m1_t v109 = v22;
      vint16m1_t v110 = v24;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v111 = __riscv_vwadd_vv_i32m2(v109, v110, 8);
      vint16m1_t v112 = v26;
      vint16m1_t v113 = v28;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v114 = __riscv_vwadd_vv_i32m2(v112, v113, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v115 = (const _Float16*) v19;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v116 = __riscv_vle16_v_f16m1(v115, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v117 = v19 + 16;
      const _Float16* v118 = (const _Float16*) v117;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v119 = __riscv_vle16_v_f16m1(v118, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v120 = (const _Float16*) v21;
      _Float16 v121 = *(const _Float16 *)(v120);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v122 = __riscv_vfwmul_vf_f32m2(v116, v121, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v123 = __riscv_vfcvt_f_x_v_f32m2(v111, 8);
      vfloat32m2_t v124 = v13;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v125 = __riscv_vfmacc_vv_f32m2(v124, v123, v122, 8);
      v13 = v125;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v126 = __riscv_vfwmul_vf_f32m2(v119, v121, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v127 = __riscv_vfcvt_f_x_v_f32m2(v114, 8);
      vfloat32m2_t v128 = v15;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v129 = __riscv_vfmacc_vv_f32m2(v128, v127, v126, 8);
      v15 = v129;
    }
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v130 = v9 * 16;
    float* v131 = v2 + v130;
    vfloat32m2_t v132 = v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v131, v132, 8);
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v133 = v9 * 16;
    size_t v134 = v133 + 8;
    float* v135 = v2 + v134;
    vfloat32m2_t v136 = v15;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v135, v136, 8);
  }
  vint32m1_t v137 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


