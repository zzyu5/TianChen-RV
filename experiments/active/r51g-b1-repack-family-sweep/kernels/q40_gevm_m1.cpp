#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q40_gevm_m1(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
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
    size_t v14 = v13 * 288;
    const uint8_t* v15 = v4 + v14;
    vfloat32m4_t v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v17 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
    v16 = v17;
    for (size_t v18 = 0; v18 < v10; v18 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
      size_t v19 = v18 * 288;
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
        const int8_t* v31 = (const int8_t*) v30;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
        vint8m1_t v32 = __riscv_vle8_v_i8m1(v31, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
        vint8m1_t v33 = __riscv_vsll_vx_i8m1(v32, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
        vint8m1_t v34 = __riscv_vsra_vx_i8m1(v33, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
        vint8m1_t v35 = __riscv_vsra_vx_i8m1(v32, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v36 = 2 + v27;
        const uint8_t* v37 = v22 + v36;
        const int8_t* v38 = (const int8_t*) v37;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v39 = *(const int8_t *)(v38);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v40 = 2 + 16;
        size_t v41 = v40 + v27;
        const uint8_t* v42 = v22 + v41;
        const int8_t* v43 = (const int8_t*) v42;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v44 = *(const int8_t *)(v43);
        vint16m2_t v45 = v23;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
        vint16m2_t v46 = __riscv_vwmacc_vx_i16m2(v45, v39, v34, 16);
        v23 = v46;
        vint16m2_t v47 = v25;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
        vint16m2_t v48 = __riscv_vwmacc_vx_i16m2(v47, v44, v35, 16);
        v25 = v48;
      }
      vint16m2_t v49 = v23;
      vint16m2_t v50 = v25;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
      vint32m4_t v51 = __riscv_vwadd_vv_i32m4(v49, v50, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v52 = (const _Float16*) v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
      vfloat16m2_t v53 = __riscv_vle16_v_f16m2(v52, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v54 = (const _Float16*) v22;
      _Float16 v55 = *(const _Float16 *)(v54);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
      vfloat32m4_t v56 = __riscv_vfwmul_vf_f32m4(v53, v55, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
      vfloat32m4_t v57 = __riscv_vfcvt_f_x_v_f32m4(v51, 16);
      vfloat32m4_t v58 = v16;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
      vfloat32m4_t v59 = __riscv_vfmacc_vv_f32m4(v58, v57, v56, 16);
      v16 = v59;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v60 = v12 * 16;
    float* v61 = v2 + v60;
    vfloat32m4_t v62 = v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v61, v62, 16);
  }
  return;
}


