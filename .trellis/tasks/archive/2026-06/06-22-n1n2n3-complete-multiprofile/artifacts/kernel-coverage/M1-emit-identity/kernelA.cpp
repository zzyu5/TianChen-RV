#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v9 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v10 = v1 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count
  size_t v11 = v3 / 16;
  for (size_t v12 = 0; v12 < v11; v12 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
    size_t v13 = v12 * v10;
    size_t v14 = v13 * 288;
    const uint8_t* v15 = v4 + v14;
    vfloat32m2_t v16;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v17 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v16 = v17;
    vfloat32m2_t v18;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v19 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v18 = v19;
    for (size_t v20 = 0; v20 < v10; v20 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v21 = v20 * 288;
      const uint8_t* v22 = v15 + v21;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v23 = v20 * 34;
      const uint8_t* v24 = v6 + v23;
      vint16m1_t v25;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v26 = __riscv_vmv_v_x_i16m1(0, 8);
      v25 = v26;
      vint16m1_t v27;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v28 = __riscv_vmv_v_x_i16m1(0, 8);
      v27 = v28;
      vint16m1_t v29;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v30 = __riscv_vmv_v_x_i16m1(0, 8);
      v29 = v30;
      vint16m1_t v31;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v32 = __riscv_vmv_v_x_i16m1(0, 8);
      v31 = v32;
      for (size_t v33 = 0; v33 < 16; v33 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v34 = v33 * 16;
        size_t v35 = 32 + v34;
        size_t v36 = v35 + 8;
        const uint8_t* v37 = v22 + v35;
        const int8_t* v38 = (const int8_t*) v37;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v39 = __riscv_vle8_v_i8mf2(v38, 8);
        const uint8_t* v40 = v22 + v36;
        const int8_t* v41 = (const int8_t*) v40;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v42 = __riscv_vle8_v_i8mf2(v41, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8mf2
        vint8mf2_t v43 = __riscv_vsll_vx_i8mf2(v39, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2
        vint8mf2_t v44 = __riscv_vsra_vx_i8mf2(v43, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2
        vint8mf2_t v45 = __riscv_vsra_vx_i8mf2(v39, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8mf2
        vint8mf2_t v46 = __riscv_vsll_vx_i8mf2(v42, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2
        vint8mf2_t v47 = __riscv_vsra_vx_i8mf2(v46, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2
        vint8mf2_t v48 = __riscv_vsra_vx_i8mf2(v42, 4, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v49 = 2 + v33;
        const uint8_t* v50 = v24 + v49;
        const int8_t* v51 = (const int8_t*) v50;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v52 = *(const int8_t *)(v51);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v53 = 2 + 16;
        size_t v54 = v53 + v33;
        const uint8_t* v55 = v24 + v54;
        const int8_t* v56 = (const int8_t*) v55;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v57 = *(const int8_t *)(v56);
        vint16m1_t v58 = v25;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v59 = __riscv_vwmacc_vx_i16m1(v58, v52, v44, 8);
        v25 = v59;
        vint16m1_t v60 = v27;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v61 = __riscv_vwmacc_vx_i16m1(v60, v57, v45, 8);
        v27 = v61;
        vint16m1_t v62 = v29;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v63 = __riscv_vwmacc_vx_i16m1(v62, v52, v47, 8);
        v29 = v63;
        vint16m1_t v64 = v31;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v65 = __riscv_vwmacc_vx_i16m1(v64, v57, v48, 8);
        v31 = v65;
      }
      vint16m1_t v66 = v25;
      vint16m1_t v67 = v27;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v68 = __riscv_vwadd_vv_i32m2(v66, v67, 8);
      vint16m1_t v69 = v29;
      vint16m1_t v70 = v31;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v71 = __riscv_vwadd_vv_i32m2(v69, v70, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v72 = (const _Float16*) v22;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v73 = __riscv_vle16_v_f16m1(v72, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v74 = v22 + 16;
      const _Float16* v75 = (const _Float16*) v74;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v76 = __riscv_vle16_v_f16m1(v75, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v77 = (const _Float16*) v24;
      _Float16 v78 = *(const _Float16 *)(v77);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v79 = __riscv_vfwmul_vf_f32m2(v73, v78, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v80 = __riscv_vfcvt_f_x_v_f32m2(v68, 8);
      vfloat32m2_t v81 = v16;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v82 = __riscv_vfmacc_vv_f32m2(v81, v80, v79, 8);
      v16 = v82;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v83 = __riscv_vfwmul_vf_f32m2(v76, v78, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v84 = __riscv_vfcvt_f_x_v_f32m2(v71, 8);
      vfloat32m2_t v85 = v18;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v86 = __riscv_vfmacc_vv_f32m2(v85, v84, v83, 8);
      v18 = v86;
    }
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v87 = v12 * 16;
    float* v88 = v2 + v87;
    vfloat32m2_t v89 = v16;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v88, v89, 8);
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v90 = v12 * 16;
    size_t v91 = v90 + 8;
    float* v92 = v2 + v91;
    vfloat32m2_t v93 = v18;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v92, v93, 8);
  }
  vint32m1_t v94 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


