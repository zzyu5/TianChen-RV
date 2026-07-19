#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q8_gevm_m1(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
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
    size_t v14 = v13 * 544;
    const uint8_t* v15 = v4 + v14;
    vfloat32m4_t v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v17 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
    v16 = v17;
    for (size_t v18 = 0; v18 < v10; v18 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
      size_t v19 = v18 * 544;
      const uint8_t* v20 = v15 + v19;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
      size_t v21 = v18 * 34;
      const uint8_t* v22 = v6 + v21;
      vint32m4_t v23;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m4
      vint32m4_t v24 = __riscv_vmv_v_x_i32m4(0, 16);
      v23 = v24;
      for (size_t v25 = 0; v25 < 32; v25 += 1) {
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_quant_addr
        size_t v26 = v25 * 16;
        size_t v27 = 32 + v26;
        const uint8_t* v28 = v20 + v27;
        const int8_t* v29 = (const int8_t*) v28;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
        vint8m1_t v30 = __riscv_vle8_v_i8m1(v29, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v31 = 2 + v25;
        const uint8_t* v32 = v22 + v31;
        const int8_t* v33 = (const int8_t*) v32;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v34 = *(const int8_t *)(v33);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m2
        vint16m2_t v35 = __riscv_vwmul_vx_i16m2(v30, v34, 16);
        vint32m4_t v36 = v23;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m4
        vint32m4_t v37 = __riscv_vwadd_wv_i32m4(v36, v35, 16);
        v23 = v37;
      }
      vint32m4_t v38 = v23;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v39 = (const _Float16*) v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
      vfloat16m2_t v40 = __riscv_vle16_v_f16m2(v39, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v41 = (const _Float16*) v22;
      _Float16 v42 = *(const _Float16 *)(v41);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
      vfloat32m4_t v43 = __riscv_vfwmul_vf_f32m4(v40, v42, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
      vfloat32m4_t v44 = __riscv_vfcvt_f_x_v_f32m4(v38, 16);
      vfloat32m4_t v45 = v16;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
      vfloat32m4_t v46 = __riscv_vfmacc_vv_f32m4(v45, v44, v43, 16);
      v16 = v46;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v47 = v12 * 16;
    float* v48 = v2 + v47;
    vfloat32m4_t v49 = v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v48, v49, 16);
  }
  return;
}


