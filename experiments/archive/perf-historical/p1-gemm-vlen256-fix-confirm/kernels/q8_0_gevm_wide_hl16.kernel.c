#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_repack_gemv_q8_0_q8_0_kernel_ggml_repack_gemv_q8_0_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count
  size_t v8 = v5 / 16;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
    size_t v10 = v9 * v7;
    size_t v11 = v10 * 544;
    const uint8_t* v12 = v3 + v11;
    vfloat32m2_t v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v14 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
    v13 = v14;
    for (size_t v15 = 0; v15 < v7; v15 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v16 = v15 * 544;
      const uint8_t* v17 = v12 + v16;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v18 = v15 * 34;
      const uint8_t* v19 = v4 + v18;
      vint32m2_t v20;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v21 = __riscv_vmv_v_x_i32m2(0, 16);
      v20 = v21;
      for (size_t v22 = 0; v22 < 32; v22 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_quant_addr
        size_t v23 = v22 * 16;
        size_t v24 = 32 + v23;
        const uint8_t* v25 = v17 + v24;
        const int8_t* v26 = (const int8_t*) v25;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v27 = __riscv_vle8_v_i8mf2(v26, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
        size_t v28 = 2 + v22;
        const uint8_t* v29 = v19 + v28;
        const int8_t* v30 = (const int8_t*) v29;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v31 = *(const int8_t *)(v30);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v32 = __riscv_vwmul_vx_i16m1(v27, v31, 16);
        vint32m2_t v33 = v20;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v34 = __riscv_vwadd_wv_i32m2(v33, v32, 16);
        v20 = v34;
      }
      vint32m2_t v35 = v20;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v36 = (const _Float16*) v17;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v37 = __riscv_vle16_v_f16m1(v36, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v38 = (const _Float16*) v19;
      _Float16 v39 = *(const _Float16 *)(v38);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v40 = __riscv_vfwmul_vf_f32m2(v37, v39, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v41 = __riscv_vfcvt_f_x_v_f32m2(v35, 16);
      vfloat32m2_t v42 = v13;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v43 = __riscv_vfmacc_vv_f32m2(v42, v41, v40, 16);
      v13 = v43;
    }
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v44 = v9 * 16;
    float* v45 = v2 + v44;
    vfloat32m2_t v46 = v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v45, v46, 16);
  }
  vint32m1_t v47 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


