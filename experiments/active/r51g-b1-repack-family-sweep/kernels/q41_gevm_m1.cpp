#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q41_gevm_m1(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
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
    size_t v14 = v13 * 320;
    const uint8_t* v15 = v4 + v14;
    vfloat32m4_t v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v17 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
    v16 = v17;
    for (size_t v18 = 0; v18 < v10; v18 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
      size_t v19 = v18 * 320;
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
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
        vuint8m1_t v33 = __riscv_vand_vx_u8m1(v32, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
        vint8m1_t v34 = __riscv_vreinterpret_v_u8m1_i8m1(v33);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
        vuint8m1_t v35 = __riscv_vsrl_vx_u8m1(v32, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
        vint8m1_t v36 = __riscv_vreinterpret_v_u8m1_i8m1(v35);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v37 = 4 + v27;
        const uint8_t* v38 = v22 + v37;
        const int8_t* v39 = (const int8_t*) v38;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v40 = *(const int8_t *)(v39);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v41 = 4 + 16;
        size_t v42 = v41 + v27;
        const uint8_t* v43 = v22 + v42;
        const int8_t* v44 = (const int8_t*) v43;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v45 = *(const int8_t *)(v44);
        vint16m2_t v46 = v23;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
        vint16m2_t v47 = __riscv_vwmacc_vx_i16m2(v46, v40, v34, 16);
        v23 = v47;
        vint16m2_t v48 = v25;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
        vint16m2_t v49 = __riscv_vwmacc_vx_i16m2(v48, v45, v36, 16);
        v25 = v49;
      }
      vint16m2_t v50 = v23;
      vint16m2_t v51 = v25;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
      vint32m4_t v52 = __riscv_vwadd_vv_i32m4(v50, v51, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v53 = (const _Float16*) v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
      vfloat16m2_t v54 = __riscv_vle16_v_f16m2(v53, 16);
      const uint8_t* v55 = v20 + 32;
      const _Float16* v56 = (const _Float16*) v55;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
      vfloat16m2_t v57 = __riscv_vle16_v_f16m2(v56, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v58 = (const _Float16*) v22;
      _Float16 v59 = *(const _Float16 *)(v58);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
      const uint8_t* v60 = v22 + 2;
      const _Float16* v61 = (const _Float16*) v60;
      _Float16 v62 = *(const _Float16 *)(v61);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
      vfloat32m4_t v63 = __riscv_vfwmul_vf_f32m4(v54, v59, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
      vfloat32m4_t v64 = __riscv_vfcvt_f_x_v_f32m4(v52, 16);
      vfloat32m4_t v65 = v16;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
      vfloat32m4_t v66 = __riscv_vfmacc_vv_f32m4(v65, v64, v63, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
      vfloat32m4_t v67 = __riscv_vfwmul_vf_f32m4(v57, v62, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m4
      vfloat32m4_t v68 = __riscv_vfadd_vv_f32m4(v66, v67, 16);
      v16 = v68;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v69 = v12 * 16;
    float* v70 = v2 + v69;
    vfloat32m4_t v71 = v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v70, v71, 16);
  }
  return;
}


