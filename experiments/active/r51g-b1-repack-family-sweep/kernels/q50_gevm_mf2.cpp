#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q50_gevm_mf2(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v9 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v10 = v1 / 32;
  // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=col_group_count
  size_t v11 = v3 / 16;
  for (size_t v12 = 0; v12 < v11; v12 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_group_base
    size_t v13 = v12 * v10;
    size_t v14 = v13 * 352;
    const uint8_t* v15 = v4 + v14;
    vfloat32m2_t v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v17 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v16 = v17;
    vfloat32m2_t v18;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v19 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v18 = v19;
    for (size_t v20 = 0; v20 < v10; v20 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
      size_t v21 = v20 * 352;
      const uint8_t* v22 = v15 + v21;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
      size_t v23 = v20 * 34;
      const uint8_t* v24 = v6 + v23;
      vint16m1_t v25;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v26 = __riscv_vmv_v_x_i16m1(0, 8);
      v25 = v26;
      vint16m1_t v27;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v28 = __riscv_vmv_v_x_i16m1(0, 8);
      v27 = v28;
      vint16m1_t v29;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v30 = __riscv_vmv_v_x_i16m1(0, 8);
      v29 = v30;
      vint16m1_t v31;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v32 = __riscv_vmv_v_x_i16m1(0, 8);
      v31 = v32;
      for (size_t v33 = 0; v33 < 16; v33 += 1) {
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v34 = v33 * 16;
        size_t v35 = 32 + v34;
        size_t v36 = v35 + 8;
        const uint8_t* v37 = v22 + v35;
        const uint8_t* v38 = (const uint8_t*) v37;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v39 = __riscv_vle8_v_u8mf2(v38, 8);
        const uint8_t* v40 = v22 + v36;
        const uint8_t* v41 = (const uint8_t*) v40;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v42 = __riscv_vle8_v_u8mf2(v41, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
        size_t v43 = v33 * 2;
        size_t v44 = 288 + v43;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
        size_t v45 = 288 + 32;
        size_t v46 = v45 + v43;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v47 = __riscv_vand_vx_u8mf2(v39, 15, 8);
        const uint8_t* v48 = v22 + v44;
        const uint8_t* v49 = (const uint8_t*) v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_bits
        vbool16_t v50 = __riscv_vlm_v_b16(v49, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmnand_mm_b16
        vbool16_t v51 = __riscv_vmnand_mm_b16(v50, v50, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v52 = __riscv_vreinterpret_v_u8mf2_i8mf2(v47);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2_mu
        vint8mf2_t v53 = __riscv_vsub_vx_i8mf2_mu(v51, v52, v52, 16, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v54 = __riscv_vsrl_vx_u8mf2(v39, 4, 8);
        const uint8_t* v55 = v22 + v46;
        const uint8_t* v56 = (const uint8_t*) v55;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_bits
        vbool16_t v57 = __riscv_vlm_v_b16(v56, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmnand_mm_b16
        vbool16_t v58 = __riscv_vmnand_mm_b16(v57, v57, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v59 = __riscv_vreinterpret_v_u8mf2_i8mf2(v54);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2_mu
        vint8mf2_t v60 = __riscv_vsub_vx_i8mf2_mu(v58, v59, v59, 16, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v61 = __riscv_vand_vx_u8mf2(v42, 15, 8);
        size_t v62 = v44 + 1;
        const uint8_t* v63 = v22 + v62;
        const uint8_t* v64 = (const uint8_t*) v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_bits
        vbool16_t v65 = __riscv_vlm_v_b16(v64, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmnand_mm_b16
        vbool16_t v66 = __riscv_vmnand_mm_b16(v65, v65, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v67 = __riscv_vreinterpret_v_u8mf2_i8mf2(v61);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2_mu
        vint8mf2_t v68 = __riscv_vsub_vx_i8mf2_mu(v66, v67, v67, 16, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v69 = __riscv_vsrl_vx_u8mf2(v42, 4, 8);
        size_t v70 = v46 + 1;
        const uint8_t* v71 = v22 + v70;
        const uint8_t* v72 = (const uint8_t*) v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_bits
        vbool16_t v73 = __riscv_vlm_v_b16(v72, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmnand_mm_b16
        vbool16_t v74 = __riscv_vmnand_mm_b16(v73, v73, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v75 = __riscv_vreinterpret_v_u8mf2_i8mf2(v69);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2_mu
        vint8mf2_t v76 = __riscv_vsub_vx_i8mf2_mu(v74, v75, v75, 16, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v77 = 2 + v33;
        const uint8_t* v78 = v24 + v77;
        const int8_t* v79 = (const int8_t*) v78;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v80 = *(const int8_t *)(v79);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v81 = 2 + 16;
        size_t v82 = v81 + v33;
        const uint8_t* v83 = v24 + v82;
        const int8_t* v84 = (const int8_t*) v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v85 = *(const int8_t *)(v84);
        vint16m1_t v86 = v25;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v87 = __riscv_vwmacc_vx_i16m1(v86, v80, v53, 8);
        v25 = v87;
        vint16m1_t v88 = v27;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v89 = __riscv_vwmacc_vx_i16m1(v88, v85, v60, 8);
        v27 = v89;
        vint16m1_t v90 = v29;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v91 = __riscv_vwmacc_vx_i16m1(v90, v80, v68, 8);
        v29 = v91;
        vint16m1_t v92 = v31;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v93 = __riscv_vwmacc_vx_i16m1(v92, v85, v76, 8);
        v31 = v93;
      }
      vint16m1_t v94 = v25;
      vint16m1_t v95 = v27;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v96 = __riscv_vwadd_vv_i32m2(v94, v95, 8);
      vint16m1_t v97 = v29;
      vint16m1_t v98 = v31;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v99 = __riscv_vwadd_vv_i32m2(v97, v98, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v100 = (const _Float16*) v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v101 = __riscv_vle16_v_f16m1(v100, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v102 = v22 + 16;
      const _Float16* v103 = (const _Float16*) v102;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v104 = __riscv_vle16_v_f16m1(v103, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v105 = (const _Float16*) v24;
      _Float16 v106 = *(const _Float16 *)(v105);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v107 = __riscv_vfwmul_vf_f32m2(v101, v106, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v108 = __riscv_vfcvt_f_x_v_f32m2(v96, 8);
      vfloat32m2_t v109 = v16;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v110 = __riscv_vfmacc_vv_f32m2(v109, v108, v107, 8);
      v16 = v110;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v111 = __riscv_vfwmul_vf_f32m2(v104, v106, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v112 = __riscv_vfcvt_f_x_v_f32m2(v99, 8);
      vfloat32m2_t v113 = v18;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v114 = __riscv_vfmacc_vv_f32m2(v113, v112, v111, 8);
      v18 = v114;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v115 = v12 * 16;
    float* v116 = v2 + v115;
    vfloat32m2_t v117 = v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v116, v117, 8);
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v118 = v12 * 16;
    size_t v119 = v118 + 8;
    float* v120 = v2 + v119;
    vfloat32m2_t v121 = v18;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v120, v121, 8);
  }
  return;
}


