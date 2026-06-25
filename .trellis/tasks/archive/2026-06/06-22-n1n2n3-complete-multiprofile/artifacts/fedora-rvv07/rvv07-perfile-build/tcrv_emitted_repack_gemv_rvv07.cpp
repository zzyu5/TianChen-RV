#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_ggml_repack_gemv_q4_0_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count
  size_t v8 = v5 / 16;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
    size_t v10 = v9 * v7;
    size_t v11 = v10 * 288;
    const uint8_t* v12 = v3 + v11;
    vfloat32m4_t v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
    vfloat32m4_t v14 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
    v13 = v14;
    for (size_t v15 = 0; v15 < v7; v15 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v16 = v15 * 288;
      const uint8_t* v17 = v12 + v16;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v18 = v15 * 34;
      const uint8_t* v19 = v4 + v18;
      vint16m2_t v20;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
      vint16m2_t v21 = __riscv_vmv_v_x_i16m2(0, 16);
      v20 = v21;
      vint16m2_t v22;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
      vint16m2_t v23 = __riscv_vmv_v_x_i16m2(0, 16);
      v22 = v23;
      for (size_t v24 = 0; v24 < 16; v24 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v25 = v24 * 16;
        size_t v26 = 32 + v25;
        const uint8_t* v27 = v17 + v26;
        const int8_t* v28 = (const int8_t*) v27;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
        vint8m1_t v29 = __riscv_vle8_v_i8m1(v28, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
        vint8m1_t v30 = __riscv_vsll_vx_i8m1(v29, 4, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
        vint8m1_t v31 = __riscv_vsra_vx_i8m1(v30, 4, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
        vint8m1_t v32 = __riscv_vsra_vx_i8m1(v29, 4, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v33 = 2 + v24;
        const uint8_t* v34 = v19 + v33;
        const int8_t* v35 = (const int8_t*) v34;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v36 = *(const int8_t *)(v35);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v37 = 2 + 16;
        size_t v38 = v37 + v24;
        const uint8_t* v39 = v19 + v38;
        const int8_t* v40 = (const int8_t*) v39;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v41 = *(const int8_t *)(v40);
        vint16m2_t v42 = v20;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
        vint16m2_t v43 = __riscv_vwmacc_vx_i16m2(v42, v36, v31, 16);
        v20 = v43;
        vint16m2_t v44 = v22;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
        vint16m2_t v45 = __riscv_vwmacc_vx_i16m2(v44, v41, v32, 16);
        v22 = v45;
      }
      vint16m2_t v46 = v20;
      vint16m2_t v47 = v22;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
      vint32m4_t v48 = __riscv_vwadd_vv_i32m4(v46, v47, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v49 = (const _Float16*) v17;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
      vfloat16m2_t v50 = __riscv_vle16_v_f16m2(v49, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v51 = (const _Float16*) v19;
      _Float16 v52 = *(const _Float16 *)(v51);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
      vfloat32m4_t v53 = __riscv_vfwmul_vf_f32m4(v50, v52, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
      vfloat32m4_t v54 = __riscv_vfcvt_f_x_v_f32m4(v48, 16);
      vfloat32m4_t v55 = v13;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
      vfloat32m4_t v56 = __riscv_vfmacc_vv_f32m4(v55, v54, v53, 16);
      v13 = v56;
    }
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v57 = v9 * 16;
    float* v58 = v2 + v57;
    vfloat32m4_t v59 = v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v58, v59, 16);
  }
  vint32m1_t v60 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


