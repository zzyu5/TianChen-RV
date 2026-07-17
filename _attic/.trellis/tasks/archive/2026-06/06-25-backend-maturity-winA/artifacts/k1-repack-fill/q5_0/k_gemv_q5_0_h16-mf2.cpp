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
    vfloat32m2_t v14 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
    v13 = v14;
    for (size_t v15 = 0; v15 < v7; v15 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v16 = v15 * 352;
      const uint8_t* v17 = v12 + v16;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v18 = v15 * 34;
      const uint8_t* v19 = v4 + v18;
      vint16m1_t v20;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v21 = __riscv_vmv_v_x_i16m1(0, 16);
      v20 = v21;
      vint16m1_t v22;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v23 = __riscv_vmv_v_x_i16m1(0, 16);
      v22 = v23;
      for (size_t v24 = 0; v24 < 16; v24 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v25 = v24 * 16;
        size_t v26 = 32 + v25;
        const uint8_t* v27 = v17 + v26;
        const uint8_t* v28 = (const uint8_t*) v27;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v29 = __riscv_vle8_v_u8mf2(v28, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_lo_addr
        size_t v30 = v24 * 2;
        size_t v31 = 288 + v30;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_hi_addr
        size_t v32 = 288 + 32;
        size_t v33 = v32 + v30;
        const uint8_t* v34 = v17 + v31;
        const uint16_t* v35 = (const uint16_t*) v34;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_mask_scalar
        int32_t v36 = (uint16_t)*(const uint16_t *)(v35);
        const uint8_t* v37 = v17 + v33;
        const uint16_t* v38 = (const uint16_t*) v37;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_mask_scalar
        int32_t v39 = (uint16_t)*(const uint16_t *)(v38);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
        vuint16m1_t v40 = __riscv_vmv_v_x_u16m1(v36, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
        vuint16m1_t v41 = __riscv_vid_v_u16m1(16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
        vuint16m1_t v42 = __riscv_vsrl_vv_u16m1(v40, v41, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
        vuint16m1_t v43 = __riscv_vand_vx_u16m1(v42, 1, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
        vuint16m1_t v44 = __riscv_vsll_vx_u16m1(v43, 4, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
        vuint8mf2_t v45 = __riscv_vncvt_x_x_w_u8mf2(v44, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v46 = __riscv_vand_vx_u8mf2(v29, 15, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v47 = __riscv_vor_vv_u8mf2(v46, v45, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v48 = __riscv_vreinterpret_v_u8mf2_i8mf2(v47);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v49 = __riscv_vsub_vx_i8mf2(v48, 16, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
        vuint16m1_t v50 = __riscv_vmv_v_x_u16m1(v39, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
        vuint16m1_t v51 = __riscv_vid_v_u16m1(16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
        vuint16m1_t v52 = __riscv_vsrl_vv_u16m1(v50, v51, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
        vuint16m1_t v53 = __riscv_vand_vx_u16m1(v52, 1, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
        vuint16m1_t v54 = __riscv_vsll_vx_u16m1(v53, 4, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
        vuint8mf2_t v55 = __riscv_vncvt_x_x_w_u8mf2(v54, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v56 = __riscv_vsrl_vx_u8mf2(v29, 4, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v57 = __riscv_vor_vv_u8mf2(v56, v55, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v58 = __riscv_vreinterpret_v_u8mf2_i8mf2(v57);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v59 = __riscv_vsub_vx_i8mf2(v58, 16, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v60 = 2 + v24;
        const uint8_t* v61 = v19 + v60;
        const int8_t* v62 = (const int8_t*) v61;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v63 = *(const int8_t *)(v62);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v64 = 2 + 16;
        size_t v65 = v64 + v24;
        const uint8_t* v66 = v19 + v65;
        const int8_t* v67 = (const int8_t*) v66;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v68 = *(const int8_t *)(v67);
        vint16m1_t v69 = v20;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v70 = __riscv_vwmacc_vx_i16m1(v69, v63, v49, 16);
        v20 = v70;
        vint16m1_t v71 = v22;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v72 = __riscv_vwmacc_vx_i16m1(v71, v68, v59, 16);
        v22 = v72;
      }
      vint16m1_t v73 = v20;
      vint16m1_t v74 = v22;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v75 = __riscv_vwadd_vv_i32m2(v73, v74, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v76 = (const _Float16*) v17;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v77 = __riscv_vle16_v_f16m1(v76, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v78 = (const _Float16*) v19;
      _Float16 v79 = *(const _Float16 *)(v78);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v80 = __riscv_vfwmul_vf_f32m2(v77, v79, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v81 = __riscv_vfcvt_f_x_v_f32m2(v75, 16);
      vfloat32m2_t v82 = v13;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v83 = __riscv_vfmacc_vv_f32m2(v82, v81, v80, 16);
      v13 = v83;
    }
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v84 = v9 * 16;
    float* v85 = v2 + v84;
    vfloat32m2_t v86 = v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q5_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v85, v86, 16);
  }
  vint32m1_t v87 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


