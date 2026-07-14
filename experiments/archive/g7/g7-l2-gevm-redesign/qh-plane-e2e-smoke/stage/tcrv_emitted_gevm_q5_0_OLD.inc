#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v9 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v10 = v1 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count
  size_t v11 = v3 / 16;
  for (size_t v12 = 0; v12 < v11; v12 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
    size_t v13 = v12 * v10;
    size_t v14 = v13 * 352;
    const uint8_t* v15 = v4 + v14;
    vfloat32m2_t v16;
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v17 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v16 = v17;
    vfloat32m2_t v18;
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v19 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v18 = v19;
    for (size_t v20 = 0; v20 < v10; v20 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v21 = v20 * 352;
      const uint8_t* v22 = v15 + v21;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v23 = v20 * 34;
      const uint8_t* v24 = v6 + v23;
      vint16m1_t v25;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v26 = __riscv_vmv_v_x_i16m1(0, 8);
      v25 = v26;
      vint16m1_t v27;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v28 = __riscv_vmv_v_x_i16m1(0, 8);
      v27 = v28;
      vint16m1_t v29;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v30 = __riscv_vmv_v_x_i16m1(0, 8);
      v29 = v30;
      vint16m1_t v31;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v32 = __riscv_vmv_v_x_i16m1(0, 8);
      v31 = v32;
      for (size_t v33 = 0; v33 < 16; v33 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v34 = v33 * 16;
        size_t v35 = 32 + v34;
        size_t v36 = v35 + 8;
        const uint8_t* v37 = v22 + v35;
        const uint8_t* v38 = (const uint8_t*) v37;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v39 = __riscv_vle8_v_u8mf2(v38, 8);
        const uint8_t* v40 = v22 + v36;
        const uint8_t* v41 = (const uint8_t*) v40;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v42 = __riscv_vle8_v_u8mf2(v41, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_lo_addr
        size_t v43 = v33 * 2;
        size_t v44 = 288 + v43;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_hi_addr
        size_t v45 = 288 + 32;
        size_t v46 = v45 + v43;
        const uint8_t* v47 = v22 + v44;
        const uint16_t* v48 = (const uint16_t*) v47;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_mask_scalar
        int32_t v49 = (uint16_t)*(const uint16_t *)(v48);
        const uint8_t* v50 = v22 + v46;
        const uint16_t* v51 = (const uint16_t*) v50;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_mask_scalar
        int32_t v52 = (uint16_t)*(const uint16_t *)(v51);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
        vuint16m1_t v53 = __riscv_vmv_v_x_u16m1(v49, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
        vuint16m1_t v54 = __riscv_vid_v_u16m1(8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
        vuint16m1_t v55 = __riscv_vsrl_vv_u16m1(v53, v54, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
        vuint16m1_t v56 = __riscv_vand_vx_u16m1(v55, 1, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
        vuint16m1_t v57 = __riscv_vsll_vx_u16m1(v56, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
        vuint8mf2_t v58 = __riscv_vncvt_x_x_w_u8mf2(v57, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v59 = __riscv_vand_vx_u8mf2(v39, 15, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v60 = __riscv_vor_vv_u8mf2(v59, v58, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v61 = __riscv_vreinterpret_v_u8mf2_i8mf2(v60);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v62 = __riscv_vsub_vx_i8mf2(v61, 16, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
        vuint16m1_t v63 = __riscv_vmv_v_x_u16m1(v52, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
        vuint16m1_t v64 = __riscv_vid_v_u16m1(8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
        vuint16m1_t v65 = __riscv_vsrl_vv_u16m1(v63, v64, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
        vuint16m1_t v66 = __riscv_vand_vx_u16m1(v65, 1, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
        vuint16m1_t v67 = __riscv_vsll_vx_u16m1(v66, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
        vuint8mf2_t v68 = __riscv_vncvt_x_x_w_u8mf2(v67, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v69 = __riscv_vsrl_vx_u8mf2(v39, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v70 = __riscv_vor_vv_u8mf2(v69, v68, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v71 = __riscv_vreinterpret_v_u8mf2_i8mf2(v70);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v72 = __riscv_vsub_vx_i8mf2(v71, 16, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
        vuint16m1_t v73 = __riscv_vmv_v_x_u16m1(v49, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
        vuint16m1_t v74 = __riscv_vid_v_u16m1(8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m1
        vuint16m1_t v75 = __riscv_vadd_vx_u16m1(v74, 8, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
        vuint16m1_t v76 = __riscv_vsrl_vv_u16m1(v73, v75, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
        vuint16m1_t v77 = __riscv_vand_vx_u16m1(v76, 1, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
        vuint16m1_t v78 = __riscv_vsll_vx_u16m1(v77, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
        vuint8mf2_t v79 = __riscv_vncvt_x_x_w_u8mf2(v78, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v80 = __riscv_vand_vx_u8mf2(v42, 15, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v81 = __riscv_vor_vv_u8mf2(v80, v79, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v82 = __riscv_vreinterpret_v_u8mf2_i8mf2(v81);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v83 = __riscv_vsub_vx_i8mf2(v82, 16, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
        vuint16m1_t v84 = __riscv_vmv_v_x_u16m1(v52, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
        vuint16m1_t v85 = __riscv_vid_v_u16m1(8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m1
        vuint16m1_t v86 = __riscv_vadd_vx_u16m1(v85, 8, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
        vuint16m1_t v87 = __riscv_vsrl_vv_u16m1(v84, v86, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
        vuint16m1_t v88 = __riscv_vand_vx_u16m1(v87, 1, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
        vuint16m1_t v89 = __riscv_vsll_vx_u16m1(v88, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
        vuint8mf2_t v90 = __riscv_vncvt_x_x_w_u8mf2(v89, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v91 = __riscv_vsrl_vx_u8mf2(v42, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v92 = __riscv_vor_vv_u8mf2(v91, v90, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v93 = __riscv_vreinterpret_v_u8mf2_i8mf2(v92);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v94 = __riscv_vsub_vx_i8mf2(v93, 16, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v95 = 2 + v33;
        const uint8_t* v96 = v24 + v95;
        const int8_t* v97 = (const int8_t*) v96;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v98 = *(const int8_t *)(v97);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v99 = 2 + 16;
        size_t v100 = v99 + v33;
        const uint8_t* v101 = v24 + v100;
        const int8_t* v102 = (const int8_t*) v101;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v103 = *(const int8_t *)(v102);
        vint16m1_t v104 = v25;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v105 = __riscv_vwmacc_vx_i16m1(v104, v98, v62, 8);
        v25 = v105;
        vint16m1_t v106 = v27;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v107 = __riscv_vwmacc_vx_i16m1(v106, v103, v72, 8);
        v27 = v107;
        vint16m1_t v108 = v29;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v109 = __riscv_vwmacc_vx_i16m1(v108, v98, v83, 8);
        v29 = v109;
        vint16m1_t v110 = v31;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v111 = __riscv_vwmacc_vx_i16m1(v110, v103, v94, 8);
        v31 = v111;
      }
      vint16m1_t v112 = v25;
      vint16m1_t v113 = v27;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v114 = __riscv_vwadd_vv_i32m2(v112, v113, 8);
      vint16m1_t v115 = v29;
      vint16m1_t v116 = v31;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v117 = __riscv_vwadd_vv_i32m2(v115, v116, 8);
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v118 = (const _Float16*) v22;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v119 = __riscv_vle16_v_f16m1(v118, 8);
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v120 = v22 + 16;
      const _Float16* v121 = (const _Float16*) v120;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v122 = __riscv_vle16_v_f16m1(v121, 8);
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v123 = (const _Float16*) v24;
      _Float16 v124 = *(const _Float16 *)(v123);
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v125 = __riscv_vfwmul_vf_f32m2(v119, v124, 8);
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v126 = __riscv_vfcvt_f_x_v_f32m2(v114, 8);
      vfloat32m2_t v127 = v16;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v128 = __riscv_vfmacc_vv_f32m2(v127, v126, v125, 8);
      v16 = v128;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v129 = __riscv_vfwmul_vf_f32m2(v122, v124, 8);
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v130 = __riscv_vfcvt_f_x_v_f32m2(v117, 8);
      vfloat32m2_t v131 = v18;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v132 = __riscv_vfmacc_vv_f32m2(v131, v130, v129, 8);
      v18 = v132;
    }
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v133 = v12 * 16;
    float* v134 = v2 + v133;
    vfloat32m2_t v135 = v16;
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v134, v135, 8);
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v136 = v12 * 16;
    size_t v137 = v136 + 8;
    float* v138 = v2 + v137;
    vfloat32m2_t v139 = v18;
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v138, v139, 8);
  }
  return;
}


