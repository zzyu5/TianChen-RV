#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q51_gevm_m1(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
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
    vfloat32m4_t v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v17 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
    v16 = v17;
    for (size_t v18 = 0; v18 < v10; v18 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
      size_t v19 = v18 * 384;
      const uint8_t* v20 = v15 + v19;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
      size_t v21 = v18 * 36;
      const uint8_t* v22 = v6 + v21;
      vint16m2_t v23;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
      vint16m2_t v24 = __riscv_vmv_v_x_i16m2(0, 16);
      v23 = v24;
      vint16m2_t v25;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
      vint16m2_t v26 = __riscv_vmv_v_x_i16m2(0, 16);
      v25 = v26;
      for (size_t v27 = 0; v27 < 16; v27 += 1) {
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v28 = v27 * 16;
        size_t v29 = 64 + v28;
        const uint8_t* v30 = v20 + v29;
        const uint8_t* v31 = (const uint8_t*) v30;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
        vuint8m1_t v32 = __riscv_vle8_v_u8m1(v31, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
        size_t v33 = v27 * 2;
        size_t v34 = 320 + v33;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
        size_t v35 = 320 + 32;
        size_t v36 = v35 + v33;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
        vuint8m1_t v37 = __riscv_vand_vx_u8m1(v32, 15, 16);
        const uint8_t* v38 = v20 + v34;
        const uint8_t* v39 = (const uint8_t*) v38;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_bits
        vbool8_t v40 = __riscv_vlm_v_b8(v39, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8m1_mu
        vuint8m1_t v41 = __riscv_vadd_vx_u8m1_mu(v40, v37, v37, 16, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
        vint8m1_t v42 = __riscv_vreinterpret_v_u8m1_i8m1(v41);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
        vuint8m1_t v43 = __riscv_vsrl_vx_u8m1(v32, 4, 16);
        const uint8_t* v44 = v20 + v36;
        const uint8_t* v45 = (const uint8_t*) v44;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_bits
        vbool8_t v46 = __riscv_vlm_v_b8(v45, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8m1_mu
        vuint8m1_t v47 = __riscv_vadd_vx_u8m1_mu(v46, v43, v43, 16, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
        vint8m1_t v48 = __riscv_vreinterpret_v_u8m1_i8m1(v47);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v49 = 4 + v27;
        const uint8_t* v50 = v22 + v49;
        const int8_t* v51 = (const int8_t*) v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v52 = *(const int8_t *)(v51);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v53 = 4 + 16;
        size_t v54 = v53 + v27;
        const uint8_t* v55 = v22 + v54;
        const int8_t* v56 = (const int8_t*) v55;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v57 = *(const int8_t *)(v56);
        vint16m2_t v58 = v23;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
        vint16m2_t v59 = __riscv_vwmacc_vx_i16m2(v58, v52, v42, 16);
        v23 = v59;
        vint16m2_t v60 = v25;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
        vint16m2_t v61 = __riscv_vwmacc_vx_i16m2(v60, v57, v48, 16);
        v25 = v61;
      }
      vint16m2_t v62 = v23;
      vint16m2_t v63 = v25;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
      vint32m4_t v64 = __riscv_vwadd_vv_i32m4(v62, v63, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v65 = (const _Float16*) v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
      vfloat16m2_t v66 = __riscv_vle16_v_f16m2(v65, 16);
      const uint8_t* v67 = v20 + 32;
      const _Float16* v68 = (const _Float16*) v67;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
      vfloat16m2_t v69 = __riscv_vle16_v_f16m2(v68, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v70 = (const _Float16*) v22;
      _Float16 v71 = *(const _Float16 *)(v70);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
      const uint8_t* v72 = v22 + 2;
      const _Float16* v73 = (const _Float16*) v72;
      _Float16 v74 = *(const _Float16 *)(v73);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
      vfloat32m4_t v75 = __riscv_vfwmul_vf_f32m4(v66, v71, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
      vfloat32m4_t v76 = __riscv_vfcvt_f_x_v_f32m4(v64, 16);
      vfloat32m4_t v77 = v16;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
      vfloat32m4_t v78 = __riscv_vfmacc_vv_f32m4(v77, v76, v75, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
      vfloat32m4_t v79 = __riscv_vfwmul_vf_f32m4(v69, v74, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m4
      vfloat32m4_t v80 = __riscv_vfadd_vv_f32m4(v78, v79, 16);
      v16 = v80;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v81 = v12 * 16;
    float* v82 = v2 + v81;
    vfloat32m4_t v83 = v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v82, v83, 16);
  }
  return;
}


