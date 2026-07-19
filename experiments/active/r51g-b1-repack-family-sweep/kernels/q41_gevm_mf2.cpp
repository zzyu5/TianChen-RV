#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q41_gevm_mf2(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
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
    vfloat32m2_t v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v17 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v16 = v17;
    vfloat32m2_t v18;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v19 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v18 = v19;
    for (size_t v20 = 0; v20 < v10; v20 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
      size_t v21 = v20 * 320;
      const uint8_t* v22 = v15 + v21;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
      size_t v23 = v20 * 36;
      const uint8_t* v24 = v6 + v23;
      vint16m1_t v25;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v26 = __riscv_vmv_v_x_i16m1(0, 8);
      v25 = v26;
      vint16m1_t v27;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v28 = __riscv_vmv_v_x_i16m1(0, 8);
      v27 = v28;
      vint16m1_t v29;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v30 = __riscv_vmv_v_x_i16m1(0, 8);
      v29 = v30;
      vint16m1_t v31;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v32 = __riscv_vmv_v_x_i16m1(0, 8);
      v31 = v32;
      for (size_t v33 = 0; v33 < 16; v33 += 1) {
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v34 = v33 * 16;
        size_t v35 = 64 + v34;
        size_t v36 = v35 + 8;
        const uint8_t* v37 = v22 + v35;
        const uint8_t* v38 = (const uint8_t*) v37;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v39 = __riscv_vle8_v_u8mf2(v38, 8);
        const uint8_t* v40 = v22 + v36;
        const uint8_t* v41 = (const uint8_t*) v40;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v42 = __riscv_vle8_v_u8mf2(v41, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v43 = __riscv_vand_vx_u8mf2(v39, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v44 = __riscv_vreinterpret_v_u8mf2_i8mf2(v43);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v45 = __riscv_vsrl_vx_u8mf2(v39, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v46 = __riscv_vreinterpret_v_u8mf2_i8mf2(v45);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v47 = __riscv_vand_vx_u8mf2(v42, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v48 = __riscv_vreinterpret_v_u8mf2_i8mf2(v47);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v49 = __riscv_vsrl_vx_u8mf2(v42, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v50 = __riscv_vreinterpret_v_u8mf2_i8mf2(v49);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
        size_t v51 = 4 + v33;
        const uint8_t* v52 = v24 + v51;
        const int8_t* v53 = (const int8_t*) v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v54 = *(const int8_t *)(v53);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
        size_t v55 = 4 + 16;
        size_t v56 = v55 + v33;
        const uint8_t* v57 = v24 + v56;
        const int8_t* v58 = (const int8_t*) v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v59 = *(const int8_t *)(v58);
        vint16m1_t v60 = v25;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v61 = __riscv_vwmacc_vx_i16m1(v60, v54, v44, 8);
        v25 = v61;
        vint16m1_t v62 = v27;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v63 = __riscv_vwmacc_vx_i16m1(v62, v59, v46, 8);
        v27 = v63;
        vint16m1_t v64 = v29;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v65 = __riscv_vwmacc_vx_i16m1(v64, v54, v48, 8);
        v29 = v65;
        vint16m1_t v66 = v31;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v67 = __riscv_vwmacc_vx_i16m1(v66, v59, v50, 8);
        v31 = v67;
      }
      vint16m1_t v68 = v25;
      vint16m1_t v69 = v27;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v70 = __riscv_vwadd_vv_i32m2(v68, v69, 8);
      vint16m1_t v71 = v29;
      vint16m1_t v72 = v31;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
      vint32m2_t v73 = __riscv_vwadd_vv_i32m2(v71, v72, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v74 = (const _Float16*) v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v75 = __riscv_vle16_v_f16m1(v74, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v76 = v22 + 16;
      const _Float16* v77 = (const _Float16*) v76;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v78 = __riscv_vle16_v_f16m1(v77, 8);
      const uint8_t* v79 = v22 + 32;
      const _Float16* v80 = (const _Float16*) v79;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v81 = __riscv_vle16_v_f16m1(v80, 8);
      const uint8_t* v82 = v22 + 48;
      const _Float16* v83 = (const _Float16*) v82;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v84 = __riscv_vle16_v_f16m1(v83, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v85 = (const _Float16*) v24;
      _Float16 v86 = *(const _Float16 *)(v85);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
      const uint8_t* v87 = v24 + 2;
      const _Float16* v88 = (const _Float16*) v87;
      _Float16 v89 = *(const _Float16 *)(v88);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v90 = __riscv_vfwmul_vf_f32m2(v75, v86, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v91 = __riscv_vfcvt_f_x_v_f32m2(v70, 8);
      vfloat32m2_t v92 = v16;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v93 = __riscv_vfmacc_vv_f32m2(v92, v91, v90, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v94 = __riscv_vfwmul_vf_f32m2(v81, v89, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
      vfloat32m2_t v95 = __riscv_vfadd_vv_f32m2(v93, v94, 8);
      v16 = v95;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v96 = __riscv_vfwmul_vf_f32m2(v78, v86, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v97 = __riscv_vfcvt_f_x_v_f32m2(v73, 8);
      vfloat32m2_t v98 = v18;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v99 = __riscv_vfmacc_vv_f32m2(v98, v97, v96, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v100 = __riscv_vfwmul_vf_f32m2(v84, v89, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
      vfloat32m2_t v101 = __riscv_vfadd_vv_f32m2(v99, v100, 8);
      v18 = v101;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v102 = v12 * 16;
    float* v103 = v2 + v102;
    vfloat32m2_t v104 = v16;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v103, v104, 8);
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v105 = v12 * 16;
    size_t v106 = v105 + 8;
    float* v107 = v2 + v106;
    vfloat32m2_t v108 = v18;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v107, v108, 8);
  }
  return;
}


