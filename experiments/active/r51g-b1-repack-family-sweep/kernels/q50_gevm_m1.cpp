#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q50_gevm_m1(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
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
    vfloat32m4_t v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v17 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
    v16 = v17;
    for (size_t v18 = 0; v18 < v10; v18 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
      size_t v19 = v18 * 352;
      const uint8_t* v20 = v15 + v19;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
      size_t v21 = v18 * 34;
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
        size_t v29 = 32 + v28;
        const uint8_t* v30 = v20 + v29;
        const uint8_t* v31 = (const uint8_t*) v30;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
        vuint8m1_t v32 = __riscv_vle8_v_u8m1(v31, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
        size_t v33 = v27 * 2;
        size_t v34 = 288 + v33;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
        size_t v35 = 288 + 32;
        size_t v36 = v35 + v33;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
        vuint8m1_t v37 = __riscv_vand_vx_u8m1(v32, 15, 16);
        const uint8_t* v38 = v20 + v34;
        const uint8_t* v39 = (const uint8_t*) v38;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_bits
        vbool8_t v40 = __riscv_vlm_v_b8(v39, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmnand_mm_b8
        vbool8_t v41 = __riscv_vmnand_mm_b8(v40, v40, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
        vint8m1_t v42 = __riscv_vreinterpret_v_u8m1_i8m1(v37);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1_mu
        vint8m1_t v43 = __riscv_vsub_vx_i8m1_mu(v41, v42, v42, 16, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
        vuint8m1_t v44 = __riscv_vsrl_vx_u8m1(v32, 4, 16);
        const uint8_t* v45 = v20 + v36;
        const uint8_t* v46 = (const uint8_t*) v45;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_bits
        vbool8_t v47 = __riscv_vlm_v_b8(v46, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmnand_mm_b8
        vbool8_t v48 = __riscv_vmnand_mm_b8(v47, v47, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
        vint8m1_t v49 = __riscv_vreinterpret_v_u8m1_i8m1(v44);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1_mu
        vint8m1_t v50 = __riscv_vsub_vx_i8m1_mu(v48, v49, v49, 16, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v51 = 2 + v27;
        const uint8_t* v52 = v22 + v51;
        const int8_t* v53 = (const int8_t*) v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v54 = *(const int8_t *)(v53);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v55 = 2 + 16;
        size_t v56 = v55 + v27;
        const uint8_t* v57 = v22 + v56;
        const int8_t* v58 = (const int8_t*) v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v59 = *(const int8_t *)(v58);
        vint16m2_t v60 = v23;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
        vint16m2_t v61 = __riscv_vwmacc_vx_i16m2(v60, v54, v43, 16);
        v23 = v61;
        vint16m2_t v62 = v25;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
        vint16m2_t v63 = __riscv_vwmacc_vx_i16m2(v62, v59, v50, 16);
        v25 = v63;
      }
      vint16m2_t v64 = v23;
      vint16m2_t v65 = v25;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
      vint32m4_t v66 = __riscv_vwadd_vv_i32m4(v64, v65, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v67 = (const _Float16*) v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
      vfloat16m2_t v68 = __riscv_vle16_v_f16m2(v67, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v69 = (const _Float16*) v22;
      _Float16 v70 = *(const _Float16 *)(v69);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
      vfloat32m4_t v71 = __riscv_vfwmul_vf_f32m4(v68, v70, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
      vfloat32m4_t v72 = __riscv_vfcvt_f_x_v_f32m4(v66, 16);
      vfloat32m4_t v73 = v16;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
      vfloat32m4_t v74 = __riscv_vfmacc_vv_f32m4(v73, v72, v71, 16);
      v16 = v74;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v75 = v12 * 16;
    float* v76 = v2 + v75;
    vfloat32m4_t v77 = v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v76, v77, 16);
  }
  return;
}


