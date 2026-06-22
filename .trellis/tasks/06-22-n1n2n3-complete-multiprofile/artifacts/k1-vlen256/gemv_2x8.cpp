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
    vfloat32m2_t v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v14 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v13 = v14;
    vfloat32m2_t v15;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v16 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v15 = v16;
    for (size_t v17 = 0; v17 < v7; v17 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v18 = v17 * 288;
      const uint8_t* v19 = v12 + v18;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v20 = v17 * 34;
      const uint8_t* v21 = v4 + v20;
      vint16m1_t v22;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v23 = __riscv_vmv_v_x_i16m1(0, 8);
      v22 = v23;
      vint16m1_t v24;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v25 = __riscv_vmv_v_x_i16m1(0, 8);
      v24 = v25;
      vint16m1_t v26;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v27 = __riscv_vmv_v_x_i16m1(0, 8);
      v26 = v27;
      vint16m1_t v28;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v29 = __riscv_vmv_v_x_i16m1(0, 8);
      v28 = v29;
      for (size_t v30 = 0; v30 < 16; v30 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v31 = v30 * 16;
        size_t v32 = 32 + v31;
        size_t v33 = v32 + 8;
        const uint8_t* v34 = v19 + v32;
        const int8_t* v35 = (const int8_t*) v34;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v36 = __riscv_vle8_v_i8mf2(v35, 8);
        const uint8_t* v37 = v19 + v33;
        const int8_t* v38 = (const int8_t*) v37;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v39 = __riscv_vle8_v_i8mf2(v38, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8mf2
        vint8mf2_t v40 = __riscv_vsll_vx_i8mf2(v36, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2
        vint8mf2_t v41 = __riscv_vsra_vx_i8mf2(v40, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2
        vint8mf2_t v42 = __riscv_vsra_vx_i8mf2(v36, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8mf2
        vint8mf2_t v43 = __riscv_vsll_vx_i8mf2(v39, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2
        vint8mf2_t v44 = __riscv_vsra_vx_i8mf2(v43, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2
        vint8mf2_t v45 = __riscv_vsra_vx_i8mf2(v39, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v46 = 2 + v30;
        const uint8_t* v47 = v21 + v46;
        const int8_t* v48 = (const int8_t*) v47;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v49 = *(const int8_t *)(v48);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v50 = 2 + 16;
        size_t v51 = v50 + v30;
        const uint8_t* v52 = v21 + v51;
        const int8_t* v53 = (const int8_t*) v52;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v54 = *(const int8_t *)(v53);
        vint16m1_t v55 = v22;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v56 = __riscv_vwmacc_vx_i16m1(v55, v49, v41, 8);
        v22 = v56;
        vint16m1_t v57 = v24;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v58 = __riscv_vwmacc_vx_i16m1(v57, v54, v42, 8);
        v24 = v58;
        vint16m1_t v59 = v26;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v60 = __riscv_vwmacc_vx_i16m1(v59, v49, v44, 8);
        v26 = v60;
        vint16m1_t v61 = v28;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v62 = __riscv_vwmacc_vx_i16m1(v61, v54, v45, 8);
        v28 = v62;
      }
      vint16m1_t v63 = v22;
      vint16m1_t v64 = v24;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v65 = __riscv_vwadd_vv_i32m2(v63, v64, 8);
      vint16m1_t v66 = v26;
      vint16m1_t v67 = v28;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v68 = __riscv_vwadd_vv_i32m2(v66, v67, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v69 = (const _Float16*) v19;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v70 = __riscv_vle16_v_f16m1(v69, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v71 = v19 + 16;
      const _Float16* v72 = (const _Float16*) v71;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v73 = __riscv_vle16_v_f16m1(v72, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v74 = (const _Float16*) v21;
      _Float16 v75 = *(const _Float16 *)(v74);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v76 = __riscv_vfwmul_vf_f32m2(v70, v75, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v77 = __riscv_vfcvt_f_x_v_f32m2(v65, 8);
      vfloat32m2_t v78 = v13;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v79 = __riscv_vfmacc_vv_f32m2(v78, v77, v76, 8);
      v13 = v79;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v80 = __riscv_vfwmul_vf_f32m2(v73, v75, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v81 = __riscv_vfcvt_f_x_v_f32m2(v68, 8);
      vfloat32m2_t v82 = v15;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v83 = __riscv_vfmacc_vv_f32m2(v82, v81, v80, 8);
      v15 = v83;
    }
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v84 = v9 * 16;
    float* v85 = v2 + v84;
    vfloat32m2_t v86 = v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v85, v86, 8);
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v87 = v9 * 16;
    size_t v88 = v87 + 8;
    float* v89 = v2 + v88;
    vfloat32m2_t v90 = v15;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v89, v90, 8);
  }
  vint32m1_t v91 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


