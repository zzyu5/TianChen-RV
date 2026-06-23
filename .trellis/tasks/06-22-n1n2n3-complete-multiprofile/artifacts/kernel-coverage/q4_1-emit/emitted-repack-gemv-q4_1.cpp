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
    vfloat32m2_t v14 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v13 = v14;
    vfloat32m2_t v15;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v16 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v15 = v16;
    for (size_t v17 = 0; v17 < v7; v17 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v18 = v17 * 320;
      const uint8_t* v19 = v12 + v18;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v20 = v17 * 36;
      const uint8_t* v21 = v4 + v20;
      vint16m1_t v22;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v23 = __riscv_vmv_v_x_i16m1(0, 8);
      v22 = v23;
      vint16m1_t v24;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v25 = __riscv_vmv_v_x_i16m1(0, 8);
      v24 = v25;
      vint16m1_t v26;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v27 = __riscv_vmv_v_x_i16m1(0, 8);
      v26 = v27;
      vint16m1_t v28;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v29 = __riscv_vmv_v_x_i16m1(0, 8);
      v28 = v29;
      for (size_t v30 = 0; v30 < 16; v30 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v31 = v30 * 16;
        size_t v32 = 64 + v31;
        size_t v33 = v32 + 8;
        const uint8_t* v34 = v19 + v32;
        const uint8_t* v35 = (const uint8_t*) v34;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v36 = __riscv_vle8_v_u8mf2(v35, 8);
        const uint8_t* v37 = v19 + v33;
        const uint8_t* v38 = (const uint8_t*) v37;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v39 = __riscv_vle8_v_u8mf2(v38, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v40 = __riscv_vand_vx_u8mf2(v36, 0x0F, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v41 = __riscv_vreinterpret_v_u8mf2_i8mf2(v40);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v42 = __riscv_vsrl_vx_u8mf2(v36, 0x04, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v43 = __riscv_vreinterpret_v_u8mf2_i8mf2(v42);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v44 = __riscv_vand_vx_u8mf2(v39, 0x0F, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v45 = __riscv_vreinterpret_v_u8mf2_i8mf2(v44);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v46 = __riscv_vsrl_vx_u8mf2(v39, 0x04, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v47 = __riscv_vreinterpret_v_u8mf2_i8mf2(v46);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v48 = 4 + v30;
        const uint8_t* v49 = v21 + v48;
        const int8_t* v50 = (const int8_t*) v49;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v51 = *(const int8_t *)(v50);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v52 = 4 + 16;
        size_t v53 = v52 + v30;
        const uint8_t* v54 = v21 + v53;
        const int8_t* v55 = (const int8_t*) v54;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v56 = *(const int8_t *)(v55);
        vint16m1_t v57 = v22;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v58 = __riscv_vwmacc_vx_i16m1(v57, v51, v41, 8);
        v22 = v58;
        vint16m1_t v59 = v24;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v60 = __riscv_vwmacc_vx_i16m1(v59, v56, v43, 8);
        v24 = v60;
        vint16m1_t v61 = v26;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v62 = __riscv_vwmacc_vx_i16m1(v61, v51, v45, 8);
        v26 = v62;
        vint16m1_t v63 = v28;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v64 = __riscv_vwmacc_vx_i16m1(v63, v56, v47, 8);
        v28 = v64;
      }
      vint16m1_t v65 = v22;
      vint16m1_t v66 = v24;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v67 = __riscv_vwadd_vv_i32m2(v65, v66, 8);
      vint16m1_t v68 = v26;
      vint16m1_t v69 = v28;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v70 = __riscv_vwadd_vv_i32m2(v68, v69, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v71 = (const _Float16*) v19;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v72 = __riscv_vle16_v_f16m1(v71, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v73 = v19 + 16;
      const _Float16* v74 = (const _Float16*) v73;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v75 = __riscv_vle16_v_f16m1(v74, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v76 = v19 + 32;
      const _Float16* v77 = (const _Float16*) v76;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v78 = __riscv_vle16_v_f16m1(v77, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v79 = v19 + 48;
      const _Float16* v80 = (const _Float16*) v79;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v81 = __riscv_vle16_v_f16m1(v80, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v82 = (const _Float16*) v21;
      _Float16 v83 = *(const _Float16 *)(v82);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_sum_scalar
      const uint8_t* v84 = v21 + 2;
      const _Float16* v85 = (const _Float16*) v84;
      _Float16 v86 = *(const _Float16 *)(v85);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v87 = __riscv_vfwmul_vf_f32m2(v72, v83, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v88 = __riscv_vfcvt_f_x_v_f32m2(v67, 8);
      vfloat32m2_t v89 = v13;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v90 = __riscv_vfmacc_vv_f32m2(v89, v88, v87, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v91 = __riscv_vfwmul_vf_f32m2(v78, v86, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
      vfloat32m2_t v92 = __riscv_vfadd_vv_f32m2(v90, v91, 8);
      v13 = v92;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v93 = __riscv_vfwmul_vf_f32m2(v75, v83, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v94 = __riscv_vfcvt_f_x_v_f32m2(v70, 8);
      vfloat32m2_t v95 = v15;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v96 = __riscv_vfmacc_vv_f32m2(v95, v94, v93, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v97 = __riscv_vfwmul_vf_f32m2(v81, v86, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
      vfloat32m2_t v98 = __riscv_vfadd_vv_f32m2(v96, v97, 8);
      v15 = v98;
    }
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v99 = v9 * 16;
    float* v100 = v2 + v99;
    vfloat32m2_t v101 = v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v100, v101, 8);
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v102 = v9 * 16;
    size_t v103 = v102 + 8;
    float* v104 = v2 + v103;
    vfloat32m2_t v105 = v15;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_1_q8_1 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v104, v105, 8);
  }
  vint32m1_t v106 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


