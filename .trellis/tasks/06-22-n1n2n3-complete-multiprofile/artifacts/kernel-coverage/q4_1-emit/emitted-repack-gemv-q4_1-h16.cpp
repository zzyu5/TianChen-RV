#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_repack_gemv_q4_1_q8_1_kernel_ggml_repack_gemv_q4_1_q8_1(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count
  size_t v8 = v5 / 16;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
    size_t v10 = v9 * v7;
    size_t v11 = v10 * 320;
    const uint8_t* v12 = v3 + v11;
    vfloat32m2_t v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v14 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
    v13 = v14;
    for (size_t v15 = 0; v15 < v7; v15 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v16 = v15 * 320;
      const uint8_t* v17 = v12 + v16;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v18 = v15 * 36;
      const uint8_t* v19 = v4 + v18;
      vint16m1_t v20;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v21 = __riscv_vmv_v_x_i16m1(0, 16);
      v20 = v21;
      vint16m1_t v22;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v23 = __riscv_vmv_v_x_i16m1(0, 16);
      v22 = v23;
      for (size_t v24 = 0; v24 < 16; v24 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v25 = v24 * 16;
        size_t v26 = 64 + v25;
        const uint8_t* v27 = v17 + v26;
        const uint8_t* v28 = (const uint8_t*) v27;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v29 = __riscv_vle8_v_u8mf2(v28, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v30 = __riscv_vand_vx_u8mf2(v29, 0x0F, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v31 = __riscv_vreinterpret_v_u8mf2_i8mf2(v30);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v32 = __riscv_vsrl_vx_u8mf2(v29, 0x04, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v33 = __riscv_vreinterpret_v_u8mf2_i8mf2(v32);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v34 = 4 + v24;
        const uint8_t* v35 = v19 + v34;
        const int8_t* v36 = (const int8_t*) v35;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v37 = *(const int8_t *)(v36);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v38 = 4 + 16;
        size_t v39 = v38 + v24;
        const uint8_t* v40 = v19 + v39;
        const int8_t* v41 = (const int8_t*) v40;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v42 = *(const int8_t *)(v41);
        vint16m1_t v43 = v20;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v44 = __riscv_vwmacc_vx_i16m1(v43, v37, v31, 16);
        v20 = v44;
        vint16m1_t v45 = v22;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v46 = __riscv_vwmacc_vx_i16m1(v45, v42, v33, 16);
        v22 = v46;
      }
      vint16m1_t v47 = v20;
      vint16m1_t v48 = v22;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v49 = __riscv_vwadd_vv_i32m2(v47, v48, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v50 = (const _Float16*) v17;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v51 = __riscv_vle16_v_f16m1(v50, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v52 = v17 + 32;
      const _Float16* v53 = (const _Float16*) v52;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v54 = __riscv_vle16_v_f16m1(v53, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v55 = (const _Float16*) v19;
      _Float16 v56 = *(const _Float16 *)(v55);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_sum_scalar
      const uint8_t* v57 = v19 + 2;
      const _Float16* v58 = (const _Float16*) v57;
      _Float16 v59 = *(const _Float16 *)(v58);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v60 = __riscv_vfwmul_vf_f32m2(v51, v56, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v61 = __riscv_vfcvt_f_x_v_f32m2(v49, 16);
      vfloat32m2_t v62 = v13;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v63 = __riscv_vfmacc_vv_f32m2(v62, v61, v60, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v64 = __riscv_vfwmul_vf_f32m2(v54, v59, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
      vfloat32m2_t v65 = __riscv_vfadd_vv_f32m2(v63, v64, 16);
      v13 = v65;
    }
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v66 = v9 * 16;
    float* v67 = v2 + v66;
    vfloat32m2_t v68 = v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v67, v68, 16);
  }
  vint32m1_t v69 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


