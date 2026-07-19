#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q51_gevm_mf2(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
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
    size_t v14 = v13 * 384;
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
      size_t v21 = v20 * 384;
      const uint8_t* v22 = v15 + v21;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
      size_t v23 = v20 * 36;
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
        size_t v35 = 64 + v34;
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
        size_t v44 = 320 + v43;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
        size_t v45 = 320 + 32;
        size_t v46 = v45 + v43;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v47 = __riscv_vand_vx_u8mf2(v39, 15, 8);
        const uint8_t* v48 = v22 + v44;
        const uint8_t* v49 = (const uint8_t*) v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_bits
        vbool16_t v50 = __riscv_vlm_v_b16(v49, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8mf2_mu
        vuint8mf2_t v51 = __riscv_vadd_vx_u8mf2_mu(v50, v47, v47, 16, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v52 = __riscv_vreinterpret_v_u8mf2_i8mf2(v51);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v53 = __riscv_vsrl_vx_u8mf2(v39, 4, 8);
        const uint8_t* v54 = v22 + v46;
        const uint8_t* v55 = (const uint8_t*) v54;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_bits
        vbool16_t v56 = __riscv_vlm_v_b16(v55, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8mf2_mu
        vuint8mf2_t v57 = __riscv_vadd_vx_u8mf2_mu(v56, v53, v53, 16, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v58 = __riscv_vreinterpret_v_u8mf2_i8mf2(v57);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v59 = __riscv_vand_vx_u8mf2(v42, 15, 8);
        size_t v60 = v44 + 1;
        const uint8_t* v61 = v22 + v60;
        const uint8_t* v62 = (const uint8_t*) v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_bits
        vbool16_t v63 = __riscv_vlm_v_b16(v62, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8mf2_mu
        vuint8mf2_t v64 = __riscv_vadd_vx_u8mf2_mu(v63, v59, v59, 16, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v65 = __riscv_vreinterpret_v_u8mf2_i8mf2(v64);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v66 = __riscv_vsrl_vx_u8mf2(v42, 4, 8);
        size_t v67 = v46 + 1;
        const uint8_t* v68 = v22 + v67;
        const uint8_t* v69 = (const uint8_t*) v68;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_bits
        vbool16_t v70 = __riscv_vlm_v_b16(v69, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8mf2_mu
        vuint8mf2_t v71 = __riscv_vadd_vx_u8mf2_mu(v70, v66, v66, 16, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v72 = __riscv_vreinterpret_v_u8mf2_i8mf2(v71);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v73 = 4 + v33;
        const uint8_t* v74 = v24 + v73;
        const int8_t* v75 = (const int8_t*) v74;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v76 = *(const int8_t *)(v75);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v77 = 4 + 16;
        size_t v78 = v77 + v33;
        const uint8_t* v79 = v24 + v78;
        const int8_t* v80 = (const int8_t*) v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v81 = *(const int8_t *)(v80);
        vint16m1_t v82 = v25;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v83 = __riscv_vwmacc_vx_i16m1(v82, v76, v52, 8);
        v25 = v83;
        vint16m1_t v84 = v27;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v85 = __riscv_vwmacc_vx_i16m1(v84, v81, v58, 8);
        v27 = v85;
        vint16m1_t v86 = v29;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v87 = __riscv_vwmacc_vx_i16m1(v86, v76, v65, 8);
        v29 = v87;
        vint16m1_t v88 = v31;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v89 = __riscv_vwmacc_vx_i16m1(v88, v81, v72, 8);
        v31 = v89;
      }
      vint16m1_t v90 = v25;
      vint16m1_t v91 = v27;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v92 = __riscv_vwadd_vv_i32m2(v90, v91, 8);
      vint16m1_t v93 = v29;
      vint16m1_t v94 = v31;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v95 = __riscv_vwadd_vv_i32m2(v93, v94, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v96 = (const _Float16*) v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v97 = __riscv_vle16_v_f16m1(v96, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v98 = v22 + 16;
      const _Float16* v99 = (const _Float16*) v98;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v100 = __riscv_vle16_v_f16m1(v99, 8);
      const uint8_t* v101 = v22 + 32;
      const _Float16* v102 = (const _Float16*) v101;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v103 = __riscv_vle16_v_f16m1(v102, 8);
      const uint8_t* v104 = v22 + 48;
      const _Float16* v105 = (const _Float16*) v104;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v106 = __riscv_vle16_v_f16m1(v105, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v107 = (const _Float16*) v24;
      _Float16 v108 = *(const _Float16 *)(v107);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
      const uint8_t* v109 = v24 + 2;
      const _Float16* v110 = (const _Float16*) v109;
      _Float16 v111 = *(const _Float16 *)(v110);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v112 = __riscv_vfwmul_vf_f32m2(v97, v108, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v113 = __riscv_vfcvt_f_x_v_f32m2(v92, 8);
      vfloat32m2_t v114 = v16;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v115 = __riscv_vfmacc_vv_f32m2(v114, v113, v112, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v116 = __riscv_vfwmul_vf_f32m2(v103, v111, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
      vfloat32m2_t v117 = __riscv_vfadd_vv_f32m2(v115, v116, 8);
      v16 = v117;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v118 = __riscv_vfwmul_vf_f32m2(v100, v108, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v119 = __riscv_vfcvt_f_x_v_f32m2(v95, 8);
      vfloat32m2_t v120 = v18;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v121 = __riscv_vfmacc_vv_f32m2(v120, v119, v118, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v122 = __riscv_vfwmul_vf_f32m2(v106, v111, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
      vfloat32m2_t v123 = __riscv_vfadd_vv_f32m2(v121, v122, 8);
      v18 = v123;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v124 = v12 * 16;
    float* v125 = v2 + v124;
    vfloat32m2_t v126 = v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v125, v126, 8);
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v127 = v12 * 16;
    size_t v128 = v127 + 8;
    float* v129 = v2 + v128;
    vfloat32m2_t v130 = v18;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v129, v130, 8);
  }
  return;
}


