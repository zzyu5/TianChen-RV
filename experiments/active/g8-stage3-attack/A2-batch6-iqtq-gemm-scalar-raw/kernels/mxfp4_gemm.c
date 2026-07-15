#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemm_mxfp4_q8_0_kernel_ggml_repack_gemm_mxfp4_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5, size_t v6, size_t v7) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v8 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_mxfp4_repack_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v9 = v1 / 32;
  // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=row_group_count
  size_t v10 = v5 / 4;
  // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=col_group_count
  size_t v11 = v6 / 16;
  for (size_t v12 = 0; v12 < v10; v12 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_group_base
    size_t v13 = v12 * v9;
    size_t v14 = v13 * 136;
    const uint8_t* v15 = v4 + v14;
    for (size_t v16 = 0; v16 < v11; v16 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_group_base
      size_t v17 = v16 * v9;
      size_t v18 = v17 * 272;
      const uint8_t* v19 = v3 + v18;
      vfloat32m2_t v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v21 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
      v20 = v21;
      vfloat32m2_t v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v23 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
      v22 = v23;
      vfloat32m2_t v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v25 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
      v24 = v25;
      vfloat32m2_t v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v27 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
      v26 = v27;
      vfloat32m2_t v28;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v29 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
      v28 = v29;
      vfloat32m2_t v30;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v31 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
      v30 = v31;
      vfloat32m2_t v32;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v33 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
      v32 = v33;
      vfloat32m2_t v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v35 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
      v34 = v35;
      for (size_t v36 = 0; v36 < v9; v36 += 1) {
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
        size_t v37 = v36 * 272;
        const uint8_t* v38 = v19 + v37;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
        size_t v39 = v36 * 136;
        const uint8_t* v40 = v15 + v39;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const _Float16* v41 = (const _Float16*) v40;
        _Float16 v42 = *(const _Float16 *)(v41);
        float v43 = (float) v42;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v44 = v40 + 2;
        const _Float16* v45 = (const _Float16*) v44;
        _Float16 v46 = *(const _Float16 *)(v45);
        float v47 = (float) v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v48 = v40 + 4;
        const _Float16* v49 = (const _Float16*) v48;
        _Float16 v50 = *(const _Float16 *)(v49);
        float v51 = (float) v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v52 = v40 + 6;
        const _Float16* v53 = (const _Float16*) v52;
        _Float16 v54 = *(const _Float16 *)(v53);
        float v55 = (float) v54;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_e8m0_scale_addr
        const uint8_t* v56 = (const uint8_t*) v38;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_exponent_load
        vuint8mf2_t v57 = __riscv_vle8_v_u8mf2(v56, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_widen_u32
        vuint32m2_t v58 = __riscv_vzext_vf4_u32m2(v57, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_to_fp32_half_bits
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u32m2
        vuint32m2_t v59 = __riscv_vand_vx_u32m2(v58, 0x1F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u32m2
        vuint32m2_t v60 = __riscv_vmv_v_x_u32m2(0x00200000, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vv_u32m2
        vuint32m2_t v61 = __riscv_vsll_vv_u32m2(v60, v59, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_u32m2
        vuint32m2_t v62 = __riscv_vsub_vx_u32m2(v58, 1, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u32m2
        vuint32m2_t v63 = __riscv_vsll_vx_u32m2(v62, 23, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsltu_vx_u32m2_b16
        vbool16_t v64 = __riscv_vmsltu_vx_u32m2_b16(v58, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_u32m2
        vuint32m2_t v65 = __riscv_vmerge_vvm_u32m2(v63, v61, v64, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_reinterpret_f32
        vfloat32m2_t v66 = __riscv_vreinterpret_v_u32m2_f32m2(v65);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_e8m0_scale_addr
        const uint8_t* v67 = v38 + 8;
        const uint8_t* v68 = (const uint8_t*) v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_exponent_load
        vuint8mf2_t v69 = __riscv_vle8_v_u8mf2(v68, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_widen_u32
        vuint32m2_t v70 = __riscv_vzext_vf4_u32m2(v69, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_to_fp32_half_bits
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u32m2
        vuint32m2_t v71 = __riscv_vand_vx_u32m2(v70, 0x1F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u32m2
        vuint32m2_t v72 = __riscv_vmv_v_x_u32m2(0x00200000, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vv_u32m2
        vuint32m2_t v73 = __riscv_vsll_vv_u32m2(v72, v71, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_u32m2
        vuint32m2_t v74 = __riscv_vsub_vx_u32m2(v70, 1, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u32m2
        vuint32m2_t v75 = __riscv_vsll_vx_u32m2(v74, 23, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsltu_vx_u32m2_b16
        vbool16_t v76 = __riscv_vmsltu_vx_u32m2_b16(v70, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_u32m2
        vuint32m2_t v77 = __riscv_vmerge_vvm_u32m2(v75, v73, v76, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_reinterpret_f32
        vfloat32m2_t v78 = __riscv_vreinterpret_v_u32m2_f32m2(v77);
        vint32m2_t v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v80 = __riscv_vmv_v_x_i32m2(0, 8);
        v79 = v80;
        vint32m2_t v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v82 = __riscv_vmv_v_x_i32m2(0, 8);
        v81 = v82;
        vint32m2_t v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v84 = __riscv_vmv_v_x_i32m2(0, 8);
        v83 = v84;
        vint32m2_t v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v86 = __riscv_vmv_v_x_i32m2(0, 8);
        v85 = v86;
        vint32m2_t v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v88 = __riscv_vmv_v_x_i32m2(0, 8);
        v87 = v88;
        vint32m2_t v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v90 = __riscv_vmv_v_x_i32m2(0, 8);
        v89 = v90;
        vint32m2_t v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v92 = __riscv_vmv_v_x_i32m2(0, 8);
        v91 = v92;
        vint32m2_t v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v94 = __riscv_vmv_v_x_i32m2(0, 8);
        v93 = v94;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v95 = v38 + 16;
        const uint8_t* v96 = (const uint8_t*) v95;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v97 = __riscv_vle8_v_u8mf2(v96, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v98 = __riscv_vand_vx_u8mf2(v97, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v99 = __riscv_vzext_vf2_u16m1(v98, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v100 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v99, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v101 = __riscv_vsrl_vx_u8mf2(v97, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v102 = __riscv_vzext_vf2_u16m1(v101, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v103 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v102, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v104 = v38 + 24;
        const uint8_t* v105 = (const uint8_t*) v104;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v106 = __riscv_vle8_v_u8mf2(v105, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v107 = __riscv_vand_vx_u8mf2(v106, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v108 = __riscv_vzext_vf2_u16m1(v107, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v109 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v108, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v110 = __riscv_vsrl_vx_u8mf2(v106, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v111 = __riscv_vzext_vf2_u16m1(v110, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v112 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v111, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v113 = v40 + 8;
        const int8_t* v114 = (const int8_t*) v113;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v115 = *(const int8_t *)(v114);
        const uint8_t* v116 = v40 + 72;
        const int8_t* v117 = (const int8_t*) v116;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v118 = *(const int8_t *)(v117);
        vint32m2_t v119 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v120 = __riscv_vwmul_vx_i16m1(v100, v115, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v121 = __riscv_vwadd_wv_i32m2(v119, v120, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v122 = __riscv_vwmul_vx_i16m1(v103, v118, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v123 = __riscv_vwadd_wv_i32m2(v121, v122, 8);
        v79 = v123;
        vint32m2_t v124 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v125 = __riscv_vwmul_vx_i16m1(v109, v115, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v126 = __riscv_vwadd_wv_i32m2(v124, v125, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v127 = __riscv_vwmul_vx_i16m1(v112, v118, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v128 = __riscv_vwadd_wv_i32m2(v126, v127, 8);
        v81 = v128;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v129 = v40 + 9;
        const int8_t* v130 = (const int8_t*) v129;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v131 = *(const int8_t *)(v130);
        const uint8_t* v132 = v40 + 73;
        const int8_t* v133 = (const int8_t*) v132;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v134 = *(const int8_t *)(v133);
        vint32m2_t v135 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v136 = __riscv_vwmul_vx_i16m1(v100, v131, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v137 = __riscv_vwadd_wv_i32m2(v135, v136, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v138 = __riscv_vwmul_vx_i16m1(v103, v134, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v139 = __riscv_vwadd_wv_i32m2(v137, v138, 8);
        v83 = v139;
        vint32m2_t v140 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v141 = __riscv_vwmul_vx_i16m1(v109, v131, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v142 = __riscv_vwadd_wv_i32m2(v140, v141, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v143 = __riscv_vwmul_vx_i16m1(v112, v134, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v144 = __riscv_vwadd_wv_i32m2(v142, v143, 8);
        v85 = v144;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v145 = v40 + 10;
        const int8_t* v146 = (const int8_t*) v145;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v147 = *(const int8_t *)(v146);
        const uint8_t* v148 = v40 + 74;
        const int8_t* v149 = (const int8_t*) v148;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v150 = *(const int8_t *)(v149);
        vint32m2_t v151 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v152 = __riscv_vwmul_vx_i16m1(v100, v147, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v153 = __riscv_vwadd_wv_i32m2(v151, v152, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v154 = __riscv_vwmul_vx_i16m1(v103, v150, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v155 = __riscv_vwadd_wv_i32m2(v153, v154, 8);
        v87 = v155;
        vint32m2_t v156 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v157 = __riscv_vwmul_vx_i16m1(v109, v147, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v158 = __riscv_vwadd_wv_i32m2(v156, v157, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v159 = __riscv_vwmul_vx_i16m1(v112, v150, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v160 = __riscv_vwadd_wv_i32m2(v158, v159, 8);
        v89 = v160;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v161 = v40 + 11;
        const int8_t* v162 = (const int8_t*) v161;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v163 = *(const int8_t *)(v162);
        const uint8_t* v164 = v40 + 75;
        const int8_t* v165 = (const int8_t*) v164;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v166 = *(const int8_t *)(v165);
        vint32m2_t v167 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v168 = __riscv_vwmul_vx_i16m1(v100, v163, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v169 = __riscv_vwadd_wv_i32m2(v167, v168, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v170 = __riscv_vwmul_vx_i16m1(v103, v166, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v171 = __riscv_vwadd_wv_i32m2(v169, v170, 8);
        v91 = v171;
        vint32m2_t v172 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v173 = __riscv_vwmul_vx_i16m1(v109, v163, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v174 = __riscv_vwadd_wv_i32m2(v172, v173, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v175 = __riscv_vwmul_vx_i16m1(v112, v166, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v176 = __riscv_vwadd_wv_i32m2(v174, v175, 8);
        v93 = v176;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v177 = v38 + 32;
        const uint8_t* v178 = (const uint8_t*) v177;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v179 = __riscv_vle8_v_u8mf2(v178, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v180 = __riscv_vand_vx_u8mf2(v179, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v181 = __riscv_vzext_vf2_u16m1(v180, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v182 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v181, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v183 = __riscv_vsrl_vx_u8mf2(v179, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v184 = __riscv_vzext_vf2_u16m1(v183, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v185 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v184, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v186 = v38 + 40;
        const uint8_t* v187 = (const uint8_t*) v186;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v188 = __riscv_vle8_v_u8mf2(v187, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v189 = __riscv_vand_vx_u8mf2(v188, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v190 = __riscv_vzext_vf2_u16m1(v189, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v191 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v190, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v192 = __riscv_vsrl_vx_u8mf2(v188, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v193 = __riscv_vzext_vf2_u16m1(v192, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v194 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v193, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v195 = v40 + 12;
        const int8_t* v196 = (const int8_t*) v195;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v197 = *(const int8_t *)(v196);
        const uint8_t* v198 = v40 + 76;
        const int8_t* v199 = (const int8_t*) v198;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v200 = *(const int8_t *)(v199);
        vint32m2_t v201 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v202 = __riscv_vwmul_vx_i16m1(v182, v197, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v203 = __riscv_vwadd_wv_i32m2(v201, v202, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v204 = __riscv_vwmul_vx_i16m1(v185, v200, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v205 = __riscv_vwadd_wv_i32m2(v203, v204, 8);
        v79 = v205;
        vint32m2_t v206 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v207 = __riscv_vwmul_vx_i16m1(v191, v197, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v208 = __riscv_vwadd_wv_i32m2(v206, v207, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v209 = __riscv_vwmul_vx_i16m1(v194, v200, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v210 = __riscv_vwadd_wv_i32m2(v208, v209, 8);
        v81 = v210;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v211 = v40 + 13;
        const int8_t* v212 = (const int8_t*) v211;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v213 = *(const int8_t *)(v212);
        const uint8_t* v214 = v40 + 77;
        const int8_t* v215 = (const int8_t*) v214;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v216 = *(const int8_t *)(v215);
        vint32m2_t v217 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v218 = __riscv_vwmul_vx_i16m1(v182, v213, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v219 = __riscv_vwadd_wv_i32m2(v217, v218, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v220 = __riscv_vwmul_vx_i16m1(v185, v216, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v221 = __riscv_vwadd_wv_i32m2(v219, v220, 8);
        v83 = v221;
        vint32m2_t v222 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v223 = __riscv_vwmul_vx_i16m1(v191, v213, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v224 = __riscv_vwadd_wv_i32m2(v222, v223, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v225 = __riscv_vwmul_vx_i16m1(v194, v216, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v226 = __riscv_vwadd_wv_i32m2(v224, v225, 8);
        v85 = v226;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v227 = v40 + 14;
        const int8_t* v228 = (const int8_t*) v227;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v229 = *(const int8_t *)(v228);
        const uint8_t* v230 = v40 + 78;
        const int8_t* v231 = (const int8_t*) v230;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v232 = *(const int8_t *)(v231);
        vint32m2_t v233 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v234 = __riscv_vwmul_vx_i16m1(v182, v229, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v235 = __riscv_vwadd_wv_i32m2(v233, v234, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v236 = __riscv_vwmul_vx_i16m1(v185, v232, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v237 = __riscv_vwadd_wv_i32m2(v235, v236, 8);
        v87 = v237;
        vint32m2_t v238 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v239 = __riscv_vwmul_vx_i16m1(v191, v229, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v240 = __riscv_vwadd_wv_i32m2(v238, v239, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v241 = __riscv_vwmul_vx_i16m1(v194, v232, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v242 = __riscv_vwadd_wv_i32m2(v240, v241, 8);
        v89 = v242;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v243 = v40 + 15;
        const int8_t* v244 = (const int8_t*) v243;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v245 = *(const int8_t *)(v244);
        const uint8_t* v246 = v40 + 79;
        const int8_t* v247 = (const int8_t*) v246;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v248 = *(const int8_t *)(v247);
        vint32m2_t v249 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v250 = __riscv_vwmul_vx_i16m1(v182, v245, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v251 = __riscv_vwadd_wv_i32m2(v249, v250, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v252 = __riscv_vwmul_vx_i16m1(v185, v248, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v253 = __riscv_vwadd_wv_i32m2(v251, v252, 8);
        v91 = v253;
        vint32m2_t v254 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v255 = __riscv_vwmul_vx_i16m1(v191, v245, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v256 = __riscv_vwadd_wv_i32m2(v254, v255, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v257 = __riscv_vwmul_vx_i16m1(v194, v248, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v258 = __riscv_vwadd_wv_i32m2(v256, v257, 8);
        v93 = v258;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v259 = v38 + 48;
        const uint8_t* v260 = (const uint8_t*) v259;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v261 = __riscv_vle8_v_u8mf2(v260, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v262 = __riscv_vand_vx_u8mf2(v261, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v263 = __riscv_vzext_vf2_u16m1(v262, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v264 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v263, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v265 = __riscv_vsrl_vx_u8mf2(v261, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v266 = __riscv_vzext_vf2_u16m1(v265, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v267 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v266, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v268 = v38 + 56;
        const uint8_t* v269 = (const uint8_t*) v268;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v270 = __riscv_vle8_v_u8mf2(v269, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v271 = __riscv_vand_vx_u8mf2(v270, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v272 = __riscv_vzext_vf2_u16m1(v271, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v273 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v272, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v274 = __riscv_vsrl_vx_u8mf2(v270, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v275 = __riscv_vzext_vf2_u16m1(v274, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v276 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v275, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v277 = v40 + 16;
        const int8_t* v278 = (const int8_t*) v277;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v279 = *(const int8_t *)(v278);
        const uint8_t* v280 = v40 + 80;
        const int8_t* v281 = (const int8_t*) v280;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v282 = *(const int8_t *)(v281);
        vint32m2_t v283 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v284 = __riscv_vwmul_vx_i16m1(v264, v279, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v285 = __riscv_vwadd_wv_i32m2(v283, v284, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v286 = __riscv_vwmul_vx_i16m1(v267, v282, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v287 = __riscv_vwadd_wv_i32m2(v285, v286, 8);
        v79 = v287;
        vint32m2_t v288 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v289 = __riscv_vwmul_vx_i16m1(v273, v279, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v290 = __riscv_vwadd_wv_i32m2(v288, v289, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v291 = __riscv_vwmul_vx_i16m1(v276, v282, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v292 = __riscv_vwadd_wv_i32m2(v290, v291, 8);
        v81 = v292;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v293 = v40 + 17;
        const int8_t* v294 = (const int8_t*) v293;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v295 = *(const int8_t *)(v294);
        const uint8_t* v296 = v40 + 81;
        const int8_t* v297 = (const int8_t*) v296;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v298 = *(const int8_t *)(v297);
        vint32m2_t v299 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v300 = __riscv_vwmul_vx_i16m1(v264, v295, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v301 = __riscv_vwadd_wv_i32m2(v299, v300, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v302 = __riscv_vwmul_vx_i16m1(v267, v298, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v303 = __riscv_vwadd_wv_i32m2(v301, v302, 8);
        v83 = v303;
        vint32m2_t v304 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v305 = __riscv_vwmul_vx_i16m1(v273, v295, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v306 = __riscv_vwadd_wv_i32m2(v304, v305, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v307 = __riscv_vwmul_vx_i16m1(v276, v298, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v308 = __riscv_vwadd_wv_i32m2(v306, v307, 8);
        v85 = v308;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v309 = v40 + 18;
        const int8_t* v310 = (const int8_t*) v309;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v311 = *(const int8_t *)(v310);
        const uint8_t* v312 = v40 + 82;
        const int8_t* v313 = (const int8_t*) v312;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v314 = *(const int8_t *)(v313);
        vint32m2_t v315 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v316 = __riscv_vwmul_vx_i16m1(v264, v311, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v317 = __riscv_vwadd_wv_i32m2(v315, v316, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v318 = __riscv_vwmul_vx_i16m1(v267, v314, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v319 = __riscv_vwadd_wv_i32m2(v317, v318, 8);
        v87 = v319;
        vint32m2_t v320 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v321 = __riscv_vwmul_vx_i16m1(v273, v311, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v322 = __riscv_vwadd_wv_i32m2(v320, v321, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v323 = __riscv_vwmul_vx_i16m1(v276, v314, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v324 = __riscv_vwadd_wv_i32m2(v322, v323, 8);
        v89 = v324;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v325 = v40 + 19;
        const int8_t* v326 = (const int8_t*) v325;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v327 = *(const int8_t *)(v326);
        const uint8_t* v328 = v40 + 83;
        const int8_t* v329 = (const int8_t*) v328;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v330 = *(const int8_t *)(v329);
        vint32m2_t v331 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v332 = __riscv_vwmul_vx_i16m1(v264, v327, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v333 = __riscv_vwadd_wv_i32m2(v331, v332, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v334 = __riscv_vwmul_vx_i16m1(v267, v330, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v335 = __riscv_vwadd_wv_i32m2(v333, v334, 8);
        v91 = v335;
        vint32m2_t v336 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v337 = __riscv_vwmul_vx_i16m1(v273, v327, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v338 = __riscv_vwadd_wv_i32m2(v336, v337, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v339 = __riscv_vwmul_vx_i16m1(v276, v330, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v340 = __riscv_vwadd_wv_i32m2(v338, v339, 8);
        v93 = v340;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v341 = v38 + 64;
        const uint8_t* v342 = (const uint8_t*) v341;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v343 = __riscv_vle8_v_u8mf2(v342, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v344 = __riscv_vand_vx_u8mf2(v343, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v345 = __riscv_vzext_vf2_u16m1(v344, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v346 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v345, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v347 = __riscv_vsrl_vx_u8mf2(v343, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v348 = __riscv_vzext_vf2_u16m1(v347, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v349 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v348, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v350 = v38 + 72;
        const uint8_t* v351 = (const uint8_t*) v350;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v352 = __riscv_vle8_v_u8mf2(v351, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v353 = __riscv_vand_vx_u8mf2(v352, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v354 = __riscv_vzext_vf2_u16m1(v353, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v355 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v354, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v356 = __riscv_vsrl_vx_u8mf2(v352, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v357 = __riscv_vzext_vf2_u16m1(v356, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v358 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v357, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v359 = v40 + 20;
        const int8_t* v360 = (const int8_t*) v359;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v361 = *(const int8_t *)(v360);
        const uint8_t* v362 = v40 + 84;
        const int8_t* v363 = (const int8_t*) v362;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v364 = *(const int8_t *)(v363);
        vint32m2_t v365 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v366 = __riscv_vwmul_vx_i16m1(v346, v361, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v367 = __riscv_vwadd_wv_i32m2(v365, v366, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v368 = __riscv_vwmul_vx_i16m1(v349, v364, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v369 = __riscv_vwadd_wv_i32m2(v367, v368, 8);
        v79 = v369;
        vint32m2_t v370 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v371 = __riscv_vwmul_vx_i16m1(v355, v361, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v372 = __riscv_vwadd_wv_i32m2(v370, v371, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v373 = __riscv_vwmul_vx_i16m1(v358, v364, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v374 = __riscv_vwadd_wv_i32m2(v372, v373, 8);
        v81 = v374;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v375 = v40 + 21;
        const int8_t* v376 = (const int8_t*) v375;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v377 = *(const int8_t *)(v376);
        const uint8_t* v378 = v40 + 85;
        const int8_t* v379 = (const int8_t*) v378;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v380 = *(const int8_t *)(v379);
        vint32m2_t v381 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v382 = __riscv_vwmul_vx_i16m1(v346, v377, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v383 = __riscv_vwadd_wv_i32m2(v381, v382, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v384 = __riscv_vwmul_vx_i16m1(v349, v380, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v385 = __riscv_vwadd_wv_i32m2(v383, v384, 8);
        v83 = v385;
        vint32m2_t v386 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v387 = __riscv_vwmul_vx_i16m1(v355, v377, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v388 = __riscv_vwadd_wv_i32m2(v386, v387, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v389 = __riscv_vwmul_vx_i16m1(v358, v380, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v390 = __riscv_vwadd_wv_i32m2(v388, v389, 8);
        v85 = v390;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v391 = v40 + 22;
        const int8_t* v392 = (const int8_t*) v391;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v393 = *(const int8_t *)(v392);
        const uint8_t* v394 = v40 + 86;
        const int8_t* v395 = (const int8_t*) v394;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v396 = *(const int8_t *)(v395);
        vint32m2_t v397 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v398 = __riscv_vwmul_vx_i16m1(v346, v393, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v399 = __riscv_vwadd_wv_i32m2(v397, v398, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v400 = __riscv_vwmul_vx_i16m1(v349, v396, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v401 = __riscv_vwadd_wv_i32m2(v399, v400, 8);
        v87 = v401;
        vint32m2_t v402 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v403 = __riscv_vwmul_vx_i16m1(v355, v393, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v404 = __riscv_vwadd_wv_i32m2(v402, v403, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v405 = __riscv_vwmul_vx_i16m1(v358, v396, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v406 = __riscv_vwadd_wv_i32m2(v404, v405, 8);
        v89 = v406;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v407 = v40 + 23;
        const int8_t* v408 = (const int8_t*) v407;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v409 = *(const int8_t *)(v408);
        const uint8_t* v410 = v40 + 87;
        const int8_t* v411 = (const int8_t*) v410;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v412 = *(const int8_t *)(v411);
        vint32m2_t v413 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v414 = __riscv_vwmul_vx_i16m1(v346, v409, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v415 = __riscv_vwadd_wv_i32m2(v413, v414, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v416 = __riscv_vwmul_vx_i16m1(v349, v412, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v417 = __riscv_vwadd_wv_i32m2(v415, v416, 8);
        v91 = v417;
        vint32m2_t v418 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v419 = __riscv_vwmul_vx_i16m1(v355, v409, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v420 = __riscv_vwadd_wv_i32m2(v418, v419, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v421 = __riscv_vwmul_vx_i16m1(v358, v412, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v422 = __riscv_vwadd_wv_i32m2(v420, v421, 8);
        v93 = v422;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v423 = v38 + 80;
        const uint8_t* v424 = (const uint8_t*) v423;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v425 = __riscv_vle8_v_u8mf2(v424, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v426 = __riscv_vand_vx_u8mf2(v425, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v427 = __riscv_vzext_vf2_u16m1(v426, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v428 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v427, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v429 = __riscv_vsrl_vx_u8mf2(v425, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v430 = __riscv_vzext_vf2_u16m1(v429, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v431 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v430, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v432 = v38 + 88;
        const uint8_t* v433 = (const uint8_t*) v432;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v434 = __riscv_vle8_v_u8mf2(v433, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v435 = __riscv_vand_vx_u8mf2(v434, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v436 = __riscv_vzext_vf2_u16m1(v435, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v437 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v436, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v438 = __riscv_vsrl_vx_u8mf2(v434, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v439 = __riscv_vzext_vf2_u16m1(v438, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v440 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v439, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v441 = v40 + 24;
        const int8_t* v442 = (const int8_t*) v441;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v443 = *(const int8_t *)(v442);
        const uint8_t* v444 = v40 + 88;
        const int8_t* v445 = (const int8_t*) v444;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v446 = *(const int8_t *)(v445);
        vint32m2_t v447 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v448 = __riscv_vwmul_vx_i16m1(v428, v443, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v449 = __riscv_vwadd_wv_i32m2(v447, v448, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v450 = __riscv_vwmul_vx_i16m1(v431, v446, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v451 = __riscv_vwadd_wv_i32m2(v449, v450, 8);
        v79 = v451;
        vint32m2_t v452 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v453 = __riscv_vwmul_vx_i16m1(v437, v443, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v454 = __riscv_vwadd_wv_i32m2(v452, v453, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v455 = __riscv_vwmul_vx_i16m1(v440, v446, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v456 = __riscv_vwadd_wv_i32m2(v454, v455, 8);
        v81 = v456;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v457 = v40 + 25;
        const int8_t* v458 = (const int8_t*) v457;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v459 = *(const int8_t *)(v458);
        const uint8_t* v460 = v40 + 89;
        const int8_t* v461 = (const int8_t*) v460;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v462 = *(const int8_t *)(v461);
        vint32m2_t v463 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v464 = __riscv_vwmul_vx_i16m1(v428, v459, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v465 = __riscv_vwadd_wv_i32m2(v463, v464, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v466 = __riscv_vwmul_vx_i16m1(v431, v462, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v467 = __riscv_vwadd_wv_i32m2(v465, v466, 8);
        v83 = v467;
        vint32m2_t v468 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v469 = __riscv_vwmul_vx_i16m1(v437, v459, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v470 = __riscv_vwadd_wv_i32m2(v468, v469, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v471 = __riscv_vwmul_vx_i16m1(v440, v462, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v472 = __riscv_vwadd_wv_i32m2(v470, v471, 8);
        v85 = v472;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v473 = v40 + 26;
        const int8_t* v474 = (const int8_t*) v473;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v475 = *(const int8_t *)(v474);
        const uint8_t* v476 = v40 + 90;
        const int8_t* v477 = (const int8_t*) v476;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v478 = *(const int8_t *)(v477);
        vint32m2_t v479 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v480 = __riscv_vwmul_vx_i16m1(v428, v475, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v481 = __riscv_vwadd_wv_i32m2(v479, v480, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v482 = __riscv_vwmul_vx_i16m1(v431, v478, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v483 = __riscv_vwadd_wv_i32m2(v481, v482, 8);
        v87 = v483;
        vint32m2_t v484 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v485 = __riscv_vwmul_vx_i16m1(v437, v475, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v486 = __riscv_vwadd_wv_i32m2(v484, v485, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v487 = __riscv_vwmul_vx_i16m1(v440, v478, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v488 = __riscv_vwadd_wv_i32m2(v486, v487, 8);
        v89 = v488;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v489 = v40 + 27;
        const int8_t* v490 = (const int8_t*) v489;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v491 = *(const int8_t *)(v490);
        const uint8_t* v492 = v40 + 91;
        const int8_t* v493 = (const int8_t*) v492;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v494 = *(const int8_t *)(v493);
        vint32m2_t v495 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v496 = __riscv_vwmul_vx_i16m1(v428, v491, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v497 = __riscv_vwadd_wv_i32m2(v495, v496, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v498 = __riscv_vwmul_vx_i16m1(v431, v494, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v499 = __riscv_vwadd_wv_i32m2(v497, v498, 8);
        v91 = v499;
        vint32m2_t v500 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v501 = __riscv_vwmul_vx_i16m1(v437, v491, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v502 = __riscv_vwadd_wv_i32m2(v500, v501, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v503 = __riscv_vwmul_vx_i16m1(v440, v494, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v504 = __riscv_vwadd_wv_i32m2(v502, v503, 8);
        v93 = v504;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v505 = v38 + 96;
        const uint8_t* v506 = (const uint8_t*) v505;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v507 = __riscv_vle8_v_u8mf2(v506, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v508 = __riscv_vand_vx_u8mf2(v507, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v509 = __riscv_vzext_vf2_u16m1(v508, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v510 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v509, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v511 = __riscv_vsrl_vx_u8mf2(v507, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v512 = __riscv_vzext_vf2_u16m1(v511, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v513 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v512, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v514 = v38 + 104;
        const uint8_t* v515 = (const uint8_t*) v514;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v516 = __riscv_vle8_v_u8mf2(v515, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v517 = __riscv_vand_vx_u8mf2(v516, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v518 = __riscv_vzext_vf2_u16m1(v517, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v519 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v518, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v520 = __riscv_vsrl_vx_u8mf2(v516, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v521 = __riscv_vzext_vf2_u16m1(v520, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v522 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v521, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v523 = v40 + 28;
        const int8_t* v524 = (const int8_t*) v523;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v525 = *(const int8_t *)(v524);
        const uint8_t* v526 = v40 + 92;
        const int8_t* v527 = (const int8_t*) v526;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v528 = *(const int8_t *)(v527);
        vint32m2_t v529 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v530 = __riscv_vwmul_vx_i16m1(v510, v525, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v531 = __riscv_vwadd_wv_i32m2(v529, v530, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v532 = __riscv_vwmul_vx_i16m1(v513, v528, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v533 = __riscv_vwadd_wv_i32m2(v531, v532, 8);
        v79 = v533;
        vint32m2_t v534 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v535 = __riscv_vwmul_vx_i16m1(v519, v525, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v536 = __riscv_vwadd_wv_i32m2(v534, v535, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v537 = __riscv_vwmul_vx_i16m1(v522, v528, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v538 = __riscv_vwadd_wv_i32m2(v536, v537, 8);
        v81 = v538;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v539 = v40 + 29;
        const int8_t* v540 = (const int8_t*) v539;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v541 = *(const int8_t *)(v540);
        const uint8_t* v542 = v40 + 93;
        const int8_t* v543 = (const int8_t*) v542;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v544 = *(const int8_t *)(v543);
        vint32m2_t v545 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v546 = __riscv_vwmul_vx_i16m1(v510, v541, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v547 = __riscv_vwadd_wv_i32m2(v545, v546, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v548 = __riscv_vwmul_vx_i16m1(v513, v544, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v549 = __riscv_vwadd_wv_i32m2(v547, v548, 8);
        v83 = v549;
        vint32m2_t v550 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v551 = __riscv_vwmul_vx_i16m1(v519, v541, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v552 = __riscv_vwadd_wv_i32m2(v550, v551, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v553 = __riscv_vwmul_vx_i16m1(v522, v544, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v554 = __riscv_vwadd_wv_i32m2(v552, v553, 8);
        v85 = v554;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v555 = v40 + 30;
        const int8_t* v556 = (const int8_t*) v555;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v557 = *(const int8_t *)(v556);
        const uint8_t* v558 = v40 + 94;
        const int8_t* v559 = (const int8_t*) v558;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v560 = *(const int8_t *)(v559);
        vint32m2_t v561 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v562 = __riscv_vwmul_vx_i16m1(v510, v557, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v563 = __riscv_vwadd_wv_i32m2(v561, v562, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v564 = __riscv_vwmul_vx_i16m1(v513, v560, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v565 = __riscv_vwadd_wv_i32m2(v563, v564, 8);
        v87 = v565;
        vint32m2_t v566 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v567 = __riscv_vwmul_vx_i16m1(v519, v557, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v568 = __riscv_vwadd_wv_i32m2(v566, v567, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v569 = __riscv_vwmul_vx_i16m1(v522, v560, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v570 = __riscv_vwadd_wv_i32m2(v568, v569, 8);
        v89 = v570;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v571 = v40 + 31;
        const int8_t* v572 = (const int8_t*) v571;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v573 = *(const int8_t *)(v572);
        const uint8_t* v574 = v40 + 95;
        const int8_t* v575 = (const int8_t*) v574;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v576 = *(const int8_t *)(v575);
        vint32m2_t v577 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v578 = __riscv_vwmul_vx_i16m1(v510, v573, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v579 = __riscv_vwadd_wv_i32m2(v577, v578, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v580 = __riscv_vwmul_vx_i16m1(v513, v576, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v581 = __riscv_vwadd_wv_i32m2(v579, v580, 8);
        v91 = v581;
        vint32m2_t v582 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v583 = __riscv_vwmul_vx_i16m1(v519, v573, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v584 = __riscv_vwadd_wv_i32m2(v582, v583, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v585 = __riscv_vwmul_vx_i16m1(v522, v576, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v586 = __riscv_vwadd_wv_i32m2(v584, v585, 8);
        v93 = v586;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v587 = v38 + 112;
        const uint8_t* v588 = (const uint8_t*) v587;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v589 = __riscv_vle8_v_u8mf2(v588, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v590 = __riscv_vand_vx_u8mf2(v589, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v591 = __riscv_vzext_vf2_u16m1(v590, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v592 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v591, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v593 = __riscv_vsrl_vx_u8mf2(v589, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v594 = __riscv_vzext_vf2_u16m1(v593, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v595 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v594, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v596 = v38 + 120;
        const uint8_t* v597 = (const uint8_t*) v596;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v598 = __riscv_vle8_v_u8mf2(v597, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v599 = __riscv_vand_vx_u8mf2(v598, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v600 = __riscv_vzext_vf2_u16m1(v599, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v601 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v600, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v602 = __riscv_vsrl_vx_u8mf2(v598, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v603 = __riscv_vzext_vf2_u16m1(v602, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v604 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v603, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v605 = v40 + 32;
        const int8_t* v606 = (const int8_t*) v605;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v607 = *(const int8_t *)(v606);
        const uint8_t* v608 = v40 + 96;
        const int8_t* v609 = (const int8_t*) v608;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v610 = *(const int8_t *)(v609);
        vint32m2_t v611 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v612 = __riscv_vwmul_vx_i16m1(v592, v607, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v613 = __riscv_vwadd_wv_i32m2(v611, v612, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v614 = __riscv_vwmul_vx_i16m1(v595, v610, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v615 = __riscv_vwadd_wv_i32m2(v613, v614, 8);
        v79 = v615;
        vint32m2_t v616 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v617 = __riscv_vwmul_vx_i16m1(v601, v607, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v618 = __riscv_vwadd_wv_i32m2(v616, v617, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v619 = __riscv_vwmul_vx_i16m1(v604, v610, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v620 = __riscv_vwadd_wv_i32m2(v618, v619, 8);
        v81 = v620;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v621 = v40 + 33;
        const int8_t* v622 = (const int8_t*) v621;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v623 = *(const int8_t *)(v622);
        const uint8_t* v624 = v40 + 97;
        const int8_t* v625 = (const int8_t*) v624;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v626 = *(const int8_t *)(v625);
        vint32m2_t v627 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v628 = __riscv_vwmul_vx_i16m1(v592, v623, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v629 = __riscv_vwadd_wv_i32m2(v627, v628, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v630 = __riscv_vwmul_vx_i16m1(v595, v626, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v631 = __riscv_vwadd_wv_i32m2(v629, v630, 8);
        v83 = v631;
        vint32m2_t v632 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v633 = __riscv_vwmul_vx_i16m1(v601, v623, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v634 = __riscv_vwadd_wv_i32m2(v632, v633, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v635 = __riscv_vwmul_vx_i16m1(v604, v626, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v636 = __riscv_vwadd_wv_i32m2(v634, v635, 8);
        v85 = v636;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v637 = v40 + 34;
        const int8_t* v638 = (const int8_t*) v637;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v639 = *(const int8_t *)(v638);
        const uint8_t* v640 = v40 + 98;
        const int8_t* v641 = (const int8_t*) v640;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v642 = *(const int8_t *)(v641);
        vint32m2_t v643 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v644 = __riscv_vwmul_vx_i16m1(v592, v639, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v645 = __riscv_vwadd_wv_i32m2(v643, v644, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v646 = __riscv_vwmul_vx_i16m1(v595, v642, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v647 = __riscv_vwadd_wv_i32m2(v645, v646, 8);
        v87 = v647;
        vint32m2_t v648 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v649 = __riscv_vwmul_vx_i16m1(v601, v639, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v650 = __riscv_vwadd_wv_i32m2(v648, v649, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v651 = __riscv_vwmul_vx_i16m1(v604, v642, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v652 = __riscv_vwadd_wv_i32m2(v650, v651, 8);
        v89 = v652;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v653 = v40 + 35;
        const int8_t* v654 = (const int8_t*) v653;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v655 = *(const int8_t *)(v654);
        const uint8_t* v656 = v40 + 99;
        const int8_t* v657 = (const int8_t*) v656;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v658 = *(const int8_t *)(v657);
        vint32m2_t v659 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v660 = __riscv_vwmul_vx_i16m1(v592, v655, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v661 = __riscv_vwadd_wv_i32m2(v659, v660, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v662 = __riscv_vwmul_vx_i16m1(v595, v658, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v663 = __riscv_vwadd_wv_i32m2(v661, v662, 8);
        v91 = v663;
        vint32m2_t v664 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v665 = __riscv_vwmul_vx_i16m1(v601, v655, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v666 = __riscv_vwadd_wv_i32m2(v664, v665, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v667 = __riscv_vwmul_vx_i16m1(v604, v658, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v668 = __riscv_vwadd_wv_i32m2(v666, v667, 8);
        v93 = v668;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v669 = v38 + 128;
        const uint8_t* v670 = (const uint8_t*) v669;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v671 = __riscv_vle8_v_u8mf2(v670, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v672 = __riscv_vand_vx_u8mf2(v671, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v673 = __riscv_vzext_vf2_u16m1(v672, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v674 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v673, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v675 = __riscv_vsrl_vx_u8mf2(v671, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v676 = __riscv_vzext_vf2_u16m1(v675, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v677 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v676, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v678 = v38 + 136;
        const uint8_t* v679 = (const uint8_t*) v678;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v680 = __riscv_vle8_v_u8mf2(v679, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v681 = __riscv_vand_vx_u8mf2(v680, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v682 = __riscv_vzext_vf2_u16m1(v681, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v683 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v682, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v684 = __riscv_vsrl_vx_u8mf2(v680, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v685 = __riscv_vzext_vf2_u16m1(v684, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v686 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v685, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v687 = v40 + 36;
        const int8_t* v688 = (const int8_t*) v687;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v689 = *(const int8_t *)(v688);
        const uint8_t* v690 = v40 + 100;
        const int8_t* v691 = (const int8_t*) v690;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v692 = *(const int8_t *)(v691);
        vint32m2_t v693 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v694 = __riscv_vwmul_vx_i16m1(v674, v689, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v695 = __riscv_vwadd_wv_i32m2(v693, v694, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v696 = __riscv_vwmul_vx_i16m1(v677, v692, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v697 = __riscv_vwadd_wv_i32m2(v695, v696, 8);
        v79 = v697;
        vint32m2_t v698 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v699 = __riscv_vwmul_vx_i16m1(v683, v689, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v700 = __riscv_vwadd_wv_i32m2(v698, v699, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v701 = __riscv_vwmul_vx_i16m1(v686, v692, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v702 = __riscv_vwadd_wv_i32m2(v700, v701, 8);
        v81 = v702;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v703 = v40 + 37;
        const int8_t* v704 = (const int8_t*) v703;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v705 = *(const int8_t *)(v704);
        const uint8_t* v706 = v40 + 101;
        const int8_t* v707 = (const int8_t*) v706;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v708 = *(const int8_t *)(v707);
        vint32m2_t v709 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v710 = __riscv_vwmul_vx_i16m1(v674, v705, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v711 = __riscv_vwadd_wv_i32m2(v709, v710, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v712 = __riscv_vwmul_vx_i16m1(v677, v708, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v713 = __riscv_vwadd_wv_i32m2(v711, v712, 8);
        v83 = v713;
        vint32m2_t v714 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v715 = __riscv_vwmul_vx_i16m1(v683, v705, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v716 = __riscv_vwadd_wv_i32m2(v714, v715, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v717 = __riscv_vwmul_vx_i16m1(v686, v708, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v718 = __riscv_vwadd_wv_i32m2(v716, v717, 8);
        v85 = v718;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v719 = v40 + 38;
        const int8_t* v720 = (const int8_t*) v719;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v721 = *(const int8_t *)(v720);
        const uint8_t* v722 = v40 + 102;
        const int8_t* v723 = (const int8_t*) v722;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v724 = *(const int8_t *)(v723);
        vint32m2_t v725 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v726 = __riscv_vwmul_vx_i16m1(v674, v721, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v727 = __riscv_vwadd_wv_i32m2(v725, v726, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v728 = __riscv_vwmul_vx_i16m1(v677, v724, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v729 = __riscv_vwadd_wv_i32m2(v727, v728, 8);
        v87 = v729;
        vint32m2_t v730 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v731 = __riscv_vwmul_vx_i16m1(v683, v721, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v732 = __riscv_vwadd_wv_i32m2(v730, v731, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v733 = __riscv_vwmul_vx_i16m1(v686, v724, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v734 = __riscv_vwadd_wv_i32m2(v732, v733, 8);
        v89 = v734;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v735 = v40 + 39;
        const int8_t* v736 = (const int8_t*) v735;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v737 = *(const int8_t *)(v736);
        const uint8_t* v738 = v40 + 103;
        const int8_t* v739 = (const int8_t*) v738;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v740 = *(const int8_t *)(v739);
        vint32m2_t v741 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v742 = __riscv_vwmul_vx_i16m1(v674, v737, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v743 = __riscv_vwadd_wv_i32m2(v741, v742, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v744 = __riscv_vwmul_vx_i16m1(v677, v740, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v745 = __riscv_vwadd_wv_i32m2(v743, v744, 8);
        v91 = v745;
        vint32m2_t v746 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v747 = __riscv_vwmul_vx_i16m1(v683, v737, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v748 = __riscv_vwadd_wv_i32m2(v746, v747, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v749 = __riscv_vwmul_vx_i16m1(v686, v740, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v750 = __riscv_vwadd_wv_i32m2(v748, v749, 8);
        v93 = v750;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v751 = v38 + 144;
        const uint8_t* v752 = (const uint8_t*) v751;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v753 = __riscv_vle8_v_u8mf2(v752, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v754 = __riscv_vand_vx_u8mf2(v753, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v755 = __riscv_vzext_vf2_u16m1(v754, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v756 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v755, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v757 = __riscv_vsrl_vx_u8mf2(v753, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v758 = __riscv_vzext_vf2_u16m1(v757, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v759 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v758, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v760 = v38 + 152;
        const uint8_t* v761 = (const uint8_t*) v760;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v762 = __riscv_vle8_v_u8mf2(v761, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v763 = __riscv_vand_vx_u8mf2(v762, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v764 = __riscv_vzext_vf2_u16m1(v763, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v765 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v764, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v766 = __riscv_vsrl_vx_u8mf2(v762, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v767 = __riscv_vzext_vf2_u16m1(v766, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v768 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v767, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v769 = v40 + 40;
        const int8_t* v770 = (const int8_t*) v769;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v771 = *(const int8_t *)(v770);
        const uint8_t* v772 = v40 + 104;
        const int8_t* v773 = (const int8_t*) v772;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v774 = *(const int8_t *)(v773);
        vint32m2_t v775 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v776 = __riscv_vwmul_vx_i16m1(v756, v771, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v777 = __riscv_vwadd_wv_i32m2(v775, v776, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v778 = __riscv_vwmul_vx_i16m1(v759, v774, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v779 = __riscv_vwadd_wv_i32m2(v777, v778, 8);
        v79 = v779;
        vint32m2_t v780 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v781 = __riscv_vwmul_vx_i16m1(v765, v771, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v782 = __riscv_vwadd_wv_i32m2(v780, v781, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v783 = __riscv_vwmul_vx_i16m1(v768, v774, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v784 = __riscv_vwadd_wv_i32m2(v782, v783, 8);
        v81 = v784;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v785 = v40 + 41;
        const int8_t* v786 = (const int8_t*) v785;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v787 = *(const int8_t *)(v786);
        const uint8_t* v788 = v40 + 105;
        const int8_t* v789 = (const int8_t*) v788;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v790 = *(const int8_t *)(v789);
        vint32m2_t v791 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v792 = __riscv_vwmul_vx_i16m1(v756, v787, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v793 = __riscv_vwadd_wv_i32m2(v791, v792, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v794 = __riscv_vwmul_vx_i16m1(v759, v790, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v795 = __riscv_vwadd_wv_i32m2(v793, v794, 8);
        v83 = v795;
        vint32m2_t v796 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v797 = __riscv_vwmul_vx_i16m1(v765, v787, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v798 = __riscv_vwadd_wv_i32m2(v796, v797, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v799 = __riscv_vwmul_vx_i16m1(v768, v790, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v800 = __riscv_vwadd_wv_i32m2(v798, v799, 8);
        v85 = v800;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v801 = v40 + 42;
        const int8_t* v802 = (const int8_t*) v801;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v803 = *(const int8_t *)(v802);
        const uint8_t* v804 = v40 + 106;
        const int8_t* v805 = (const int8_t*) v804;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v806 = *(const int8_t *)(v805);
        vint32m2_t v807 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v808 = __riscv_vwmul_vx_i16m1(v756, v803, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v809 = __riscv_vwadd_wv_i32m2(v807, v808, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v810 = __riscv_vwmul_vx_i16m1(v759, v806, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v811 = __riscv_vwadd_wv_i32m2(v809, v810, 8);
        v87 = v811;
        vint32m2_t v812 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v813 = __riscv_vwmul_vx_i16m1(v765, v803, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v814 = __riscv_vwadd_wv_i32m2(v812, v813, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v815 = __riscv_vwmul_vx_i16m1(v768, v806, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v816 = __riscv_vwadd_wv_i32m2(v814, v815, 8);
        v89 = v816;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v817 = v40 + 43;
        const int8_t* v818 = (const int8_t*) v817;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v819 = *(const int8_t *)(v818);
        const uint8_t* v820 = v40 + 107;
        const int8_t* v821 = (const int8_t*) v820;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v822 = *(const int8_t *)(v821);
        vint32m2_t v823 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v824 = __riscv_vwmul_vx_i16m1(v756, v819, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v825 = __riscv_vwadd_wv_i32m2(v823, v824, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v826 = __riscv_vwmul_vx_i16m1(v759, v822, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v827 = __riscv_vwadd_wv_i32m2(v825, v826, 8);
        v91 = v827;
        vint32m2_t v828 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v829 = __riscv_vwmul_vx_i16m1(v765, v819, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v830 = __riscv_vwadd_wv_i32m2(v828, v829, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v831 = __riscv_vwmul_vx_i16m1(v768, v822, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v832 = __riscv_vwadd_wv_i32m2(v830, v831, 8);
        v93 = v832;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v833 = v38 + 160;
        const uint8_t* v834 = (const uint8_t*) v833;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v835 = __riscv_vle8_v_u8mf2(v834, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v836 = __riscv_vand_vx_u8mf2(v835, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v837 = __riscv_vzext_vf2_u16m1(v836, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v838 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v837, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v839 = __riscv_vsrl_vx_u8mf2(v835, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v840 = __riscv_vzext_vf2_u16m1(v839, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v841 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v840, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v842 = v38 + 168;
        const uint8_t* v843 = (const uint8_t*) v842;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v844 = __riscv_vle8_v_u8mf2(v843, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v845 = __riscv_vand_vx_u8mf2(v844, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v846 = __riscv_vzext_vf2_u16m1(v845, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v847 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v846, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v848 = __riscv_vsrl_vx_u8mf2(v844, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v849 = __riscv_vzext_vf2_u16m1(v848, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v850 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v849, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v851 = v40 + 44;
        const int8_t* v852 = (const int8_t*) v851;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v853 = *(const int8_t *)(v852);
        const uint8_t* v854 = v40 + 108;
        const int8_t* v855 = (const int8_t*) v854;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v856 = *(const int8_t *)(v855);
        vint32m2_t v857 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v858 = __riscv_vwmul_vx_i16m1(v838, v853, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v859 = __riscv_vwadd_wv_i32m2(v857, v858, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v860 = __riscv_vwmul_vx_i16m1(v841, v856, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v861 = __riscv_vwadd_wv_i32m2(v859, v860, 8);
        v79 = v861;
        vint32m2_t v862 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v863 = __riscv_vwmul_vx_i16m1(v847, v853, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v864 = __riscv_vwadd_wv_i32m2(v862, v863, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v865 = __riscv_vwmul_vx_i16m1(v850, v856, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v866 = __riscv_vwadd_wv_i32m2(v864, v865, 8);
        v81 = v866;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v867 = v40 + 45;
        const int8_t* v868 = (const int8_t*) v867;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v869 = *(const int8_t *)(v868);
        const uint8_t* v870 = v40 + 109;
        const int8_t* v871 = (const int8_t*) v870;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v872 = *(const int8_t *)(v871);
        vint32m2_t v873 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v874 = __riscv_vwmul_vx_i16m1(v838, v869, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v875 = __riscv_vwadd_wv_i32m2(v873, v874, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v876 = __riscv_vwmul_vx_i16m1(v841, v872, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v877 = __riscv_vwadd_wv_i32m2(v875, v876, 8);
        v83 = v877;
        vint32m2_t v878 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v879 = __riscv_vwmul_vx_i16m1(v847, v869, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v880 = __riscv_vwadd_wv_i32m2(v878, v879, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v881 = __riscv_vwmul_vx_i16m1(v850, v872, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v882 = __riscv_vwadd_wv_i32m2(v880, v881, 8);
        v85 = v882;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v883 = v40 + 46;
        const int8_t* v884 = (const int8_t*) v883;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v885 = *(const int8_t *)(v884);
        const uint8_t* v886 = v40 + 110;
        const int8_t* v887 = (const int8_t*) v886;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v888 = *(const int8_t *)(v887);
        vint32m2_t v889 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v890 = __riscv_vwmul_vx_i16m1(v838, v885, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v891 = __riscv_vwadd_wv_i32m2(v889, v890, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v892 = __riscv_vwmul_vx_i16m1(v841, v888, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v893 = __riscv_vwadd_wv_i32m2(v891, v892, 8);
        v87 = v893;
        vint32m2_t v894 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v895 = __riscv_vwmul_vx_i16m1(v847, v885, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v896 = __riscv_vwadd_wv_i32m2(v894, v895, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v897 = __riscv_vwmul_vx_i16m1(v850, v888, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v898 = __riscv_vwadd_wv_i32m2(v896, v897, 8);
        v89 = v898;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v899 = v40 + 47;
        const int8_t* v900 = (const int8_t*) v899;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v901 = *(const int8_t *)(v900);
        const uint8_t* v902 = v40 + 111;
        const int8_t* v903 = (const int8_t*) v902;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v904 = *(const int8_t *)(v903);
        vint32m2_t v905 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v906 = __riscv_vwmul_vx_i16m1(v838, v901, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v907 = __riscv_vwadd_wv_i32m2(v905, v906, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v908 = __riscv_vwmul_vx_i16m1(v841, v904, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v909 = __riscv_vwadd_wv_i32m2(v907, v908, 8);
        v91 = v909;
        vint32m2_t v910 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v911 = __riscv_vwmul_vx_i16m1(v847, v901, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v912 = __riscv_vwadd_wv_i32m2(v910, v911, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v913 = __riscv_vwmul_vx_i16m1(v850, v904, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v914 = __riscv_vwadd_wv_i32m2(v912, v913, 8);
        v93 = v914;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v915 = v38 + 176;
        const uint8_t* v916 = (const uint8_t*) v915;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v917 = __riscv_vle8_v_u8mf2(v916, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v918 = __riscv_vand_vx_u8mf2(v917, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v919 = __riscv_vzext_vf2_u16m1(v918, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v920 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v919, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v921 = __riscv_vsrl_vx_u8mf2(v917, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v922 = __riscv_vzext_vf2_u16m1(v921, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v923 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v922, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v924 = v38 + 184;
        const uint8_t* v925 = (const uint8_t*) v924;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v926 = __riscv_vle8_v_u8mf2(v925, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v927 = __riscv_vand_vx_u8mf2(v926, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v928 = __riscv_vzext_vf2_u16m1(v927, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v929 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v928, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v930 = __riscv_vsrl_vx_u8mf2(v926, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v931 = __riscv_vzext_vf2_u16m1(v930, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v932 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v931, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v933 = v40 + 48;
        const int8_t* v934 = (const int8_t*) v933;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v935 = *(const int8_t *)(v934);
        const uint8_t* v936 = v40 + 112;
        const int8_t* v937 = (const int8_t*) v936;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v938 = *(const int8_t *)(v937);
        vint32m2_t v939 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v940 = __riscv_vwmul_vx_i16m1(v920, v935, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v941 = __riscv_vwadd_wv_i32m2(v939, v940, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v942 = __riscv_vwmul_vx_i16m1(v923, v938, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v943 = __riscv_vwadd_wv_i32m2(v941, v942, 8);
        v79 = v943;
        vint32m2_t v944 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v945 = __riscv_vwmul_vx_i16m1(v929, v935, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v946 = __riscv_vwadd_wv_i32m2(v944, v945, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v947 = __riscv_vwmul_vx_i16m1(v932, v938, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v948 = __riscv_vwadd_wv_i32m2(v946, v947, 8);
        v81 = v948;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v949 = v40 + 49;
        const int8_t* v950 = (const int8_t*) v949;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v951 = *(const int8_t *)(v950);
        const uint8_t* v952 = v40 + 113;
        const int8_t* v953 = (const int8_t*) v952;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v954 = *(const int8_t *)(v953);
        vint32m2_t v955 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v956 = __riscv_vwmul_vx_i16m1(v920, v951, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v957 = __riscv_vwadd_wv_i32m2(v955, v956, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v958 = __riscv_vwmul_vx_i16m1(v923, v954, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v959 = __riscv_vwadd_wv_i32m2(v957, v958, 8);
        v83 = v959;
        vint32m2_t v960 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v961 = __riscv_vwmul_vx_i16m1(v929, v951, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v962 = __riscv_vwadd_wv_i32m2(v960, v961, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v963 = __riscv_vwmul_vx_i16m1(v932, v954, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v964 = __riscv_vwadd_wv_i32m2(v962, v963, 8);
        v85 = v964;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v965 = v40 + 50;
        const int8_t* v966 = (const int8_t*) v965;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v967 = *(const int8_t *)(v966);
        const uint8_t* v968 = v40 + 114;
        const int8_t* v969 = (const int8_t*) v968;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v970 = *(const int8_t *)(v969);
        vint32m2_t v971 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v972 = __riscv_vwmul_vx_i16m1(v920, v967, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v973 = __riscv_vwadd_wv_i32m2(v971, v972, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v974 = __riscv_vwmul_vx_i16m1(v923, v970, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v975 = __riscv_vwadd_wv_i32m2(v973, v974, 8);
        v87 = v975;
        vint32m2_t v976 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v977 = __riscv_vwmul_vx_i16m1(v929, v967, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v978 = __riscv_vwadd_wv_i32m2(v976, v977, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v979 = __riscv_vwmul_vx_i16m1(v932, v970, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v980 = __riscv_vwadd_wv_i32m2(v978, v979, 8);
        v89 = v980;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v981 = v40 + 51;
        const int8_t* v982 = (const int8_t*) v981;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v983 = *(const int8_t *)(v982);
        const uint8_t* v984 = v40 + 115;
        const int8_t* v985 = (const int8_t*) v984;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v986 = *(const int8_t *)(v985);
        vint32m2_t v987 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v988 = __riscv_vwmul_vx_i16m1(v920, v983, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v989 = __riscv_vwadd_wv_i32m2(v987, v988, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v990 = __riscv_vwmul_vx_i16m1(v923, v986, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v991 = __riscv_vwadd_wv_i32m2(v989, v990, 8);
        v91 = v991;
        vint32m2_t v992 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v993 = __riscv_vwmul_vx_i16m1(v929, v983, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v994 = __riscv_vwadd_wv_i32m2(v992, v993, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v995 = __riscv_vwmul_vx_i16m1(v932, v986, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v996 = __riscv_vwadd_wv_i32m2(v994, v995, 8);
        v93 = v996;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v997 = v38 + 192;
        const uint8_t* v998 = (const uint8_t*) v997;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v999 = __riscv_vle8_v_u8mf2(v998, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1000 = __riscv_vand_vx_u8mf2(v999, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1001 = __riscv_vzext_vf2_u16m1(v1000, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1002 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1001, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1003 = __riscv_vsrl_vx_u8mf2(v999, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1004 = __riscv_vzext_vf2_u16m1(v1003, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1005 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1004, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1006 = v38 + 200;
        const uint8_t* v1007 = (const uint8_t*) v1006;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1008 = __riscv_vle8_v_u8mf2(v1007, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1009 = __riscv_vand_vx_u8mf2(v1008, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1010 = __riscv_vzext_vf2_u16m1(v1009, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1011 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1010, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1012 = __riscv_vsrl_vx_u8mf2(v1008, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1013 = __riscv_vzext_vf2_u16m1(v1012, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1014 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1013, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1015 = v40 + 52;
        const int8_t* v1016 = (const int8_t*) v1015;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1017 = *(const int8_t *)(v1016);
        const uint8_t* v1018 = v40 + 116;
        const int8_t* v1019 = (const int8_t*) v1018;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1020 = *(const int8_t *)(v1019);
        vint32m2_t v1021 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1022 = __riscv_vwmul_vx_i16m1(v1002, v1017, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1023 = __riscv_vwadd_wv_i32m2(v1021, v1022, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1024 = __riscv_vwmul_vx_i16m1(v1005, v1020, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1025 = __riscv_vwadd_wv_i32m2(v1023, v1024, 8);
        v79 = v1025;
        vint32m2_t v1026 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1027 = __riscv_vwmul_vx_i16m1(v1011, v1017, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1028 = __riscv_vwadd_wv_i32m2(v1026, v1027, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1029 = __riscv_vwmul_vx_i16m1(v1014, v1020, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1030 = __riscv_vwadd_wv_i32m2(v1028, v1029, 8);
        v81 = v1030;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1031 = v40 + 53;
        const int8_t* v1032 = (const int8_t*) v1031;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1033 = *(const int8_t *)(v1032);
        const uint8_t* v1034 = v40 + 117;
        const int8_t* v1035 = (const int8_t*) v1034;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1036 = *(const int8_t *)(v1035);
        vint32m2_t v1037 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1038 = __riscv_vwmul_vx_i16m1(v1002, v1033, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1039 = __riscv_vwadd_wv_i32m2(v1037, v1038, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1040 = __riscv_vwmul_vx_i16m1(v1005, v1036, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1041 = __riscv_vwadd_wv_i32m2(v1039, v1040, 8);
        v83 = v1041;
        vint32m2_t v1042 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1043 = __riscv_vwmul_vx_i16m1(v1011, v1033, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1044 = __riscv_vwadd_wv_i32m2(v1042, v1043, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1045 = __riscv_vwmul_vx_i16m1(v1014, v1036, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1046 = __riscv_vwadd_wv_i32m2(v1044, v1045, 8);
        v85 = v1046;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1047 = v40 + 54;
        const int8_t* v1048 = (const int8_t*) v1047;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1049 = *(const int8_t *)(v1048);
        const uint8_t* v1050 = v40 + 118;
        const int8_t* v1051 = (const int8_t*) v1050;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1052 = *(const int8_t *)(v1051);
        vint32m2_t v1053 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1054 = __riscv_vwmul_vx_i16m1(v1002, v1049, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1055 = __riscv_vwadd_wv_i32m2(v1053, v1054, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1056 = __riscv_vwmul_vx_i16m1(v1005, v1052, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1057 = __riscv_vwadd_wv_i32m2(v1055, v1056, 8);
        v87 = v1057;
        vint32m2_t v1058 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1059 = __riscv_vwmul_vx_i16m1(v1011, v1049, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1060 = __riscv_vwadd_wv_i32m2(v1058, v1059, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1061 = __riscv_vwmul_vx_i16m1(v1014, v1052, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1062 = __riscv_vwadd_wv_i32m2(v1060, v1061, 8);
        v89 = v1062;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1063 = v40 + 55;
        const int8_t* v1064 = (const int8_t*) v1063;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1065 = *(const int8_t *)(v1064);
        const uint8_t* v1066 = v40 + 119;
        const int8_t* v1067 = (const int8_t*) v1066;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1068 = *(const int8_t *)(v1067);
        vint32m2_t v1069 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1070 = __riscv_vwmul_vx_i16m1(v1002, v1065, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1071 = __riscv_vwadd_wv_i32m2(v1069, v1070, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1072 = __riscv_vwmul_vx_i16m1(v1005, v1068, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1073 = __riscv_vwadd_wv_i32m2(v1071, v1072, 8);
        v91 = v1073;
        vint32m2_t v1074 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1075 = __riscv_vwmul_vx_i16m1(v1011, v1065, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1076 = __riscv_vwadd_wv_i32m2(v1074, v1075, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1077 = __riscv_vwmul_vx_i16m1(v1014, v1068, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1078 = __riscv_vwadd_wv_i32m2(v1076, v1077, 8);
        v93 = v1078;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1079 = v38 + 208;
        const uint8_t* v1080 = (const uint8_t*) v1079;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1081 = __riscv_vle8_v_u8mf2(v1080, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1082 = __riscv_vand_vx_u8mf2(v1081, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1083 = __riscv_vzext_vf2_u16m1(v1082, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1084 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1083, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1085 = __riscv_vsrl_vx_u8mf2(v1081, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1086 = __riscv_vzext_vf2_u16m1(v1085, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1087 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1086, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1088 = v38 + 216;
        const uint8_t* v1089 = (const uint8_t*) v1088;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1090 = __riscv_vle8_v_u8mf2(v1089, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1091 = __riscv_vand_vx_u8mf2(v1090, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1092 = __riscv_vzext_vf2_u16m1(v1091, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1093 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1092, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1094 = __riscv_vsrl_vx_u8mf2(v1090, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1095 = __riscv_vzext_vf2_u16m1(v1094, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1096 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1095, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1097 = v40 + 56;
        const int8_t* v1098 = (const int8_t*) v1097;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1099 = *(const int8_t *)(v1098);
        const uint8_t* v1100 = v40 + 120;
        const int8_t* v1101 = (const int8_t*) v1100;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1102 = *(const int8_t *)(v1101);
        vint32m2_t v1103 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1104 = __riscv_vwmul_vx_i16m1(v1084, v1099, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1105 = __riscv_vwadd_wv_i32m2(v1103, v1104, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1106 = __riscv_vwmul_vx_i16m1(v1087, v1102, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1107 = __riscv_vwadd_wv_i32m2(v1105, v1106, 8);
        v79 = v1107;
        vint32m2_t v1108 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1109 = __riscv_vwmul_vx_i16m1(v1093, v1099, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1110 = __riscv_vwadd_wv_i32m2(v1108, v1109, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1111 = __riscv_vwmul_vx_i16m1(v1096, v1102, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1112 = __riscv_vwadd_wv_i32m2(v1110, v1111, 8);
        v81 = v1112;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1113 = v40 + 57;
        const int8_t* v1114 = (const int8_t*) v1113;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1115 = *(const int8_t *)(v1114);
        const uint8_t* v1116 = v40 + 121;
        const int8_t* v1117 = (const int8_t*) v1116;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1118 = *(const int8_t *)(v1117);
        vint32m2_t v1119 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1120 = __riscv_vwmul_vx_i16m1(v1084, v1115, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1121 = __riscv_vwadd_wv_i32m2(v1119, v1120, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1122 = __riscv_vwmul_vx_i16m1(v1087, v1118, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1123 = __riscv_vwadd_wv_i32m2(v1121, v1122, 8);
        v83 = v1123;
        vint32m2_t v1124 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1125 = __riscv_vwmul_vx_i16m1(v1093, v1115, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1126 = __riscv_vwadd_wv_i32m2(v1124, v1125, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1127 = __riscv_vwmul_vx_i16m1(v1096, v1118, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1128 = __riscv_vwadd_wv_i32m2(v1126, v1127, 8);
        v85 = v1128;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1129 = v40 + 58;
        const int8_t* v1130 = (const int8_t*) v1129;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1131 = *(const int8_t *)(v1130);
        const uint8_t* v1132 = v40 + 122;
        const int8_t* v1133 = (const int8_t*) v1132;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1134 = *(const int8_t *)(v1133);
        vint32m2_t v1135 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1136 = __riscv_vwmul_vx_i16m1(v1084, v1131, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1137 = __riscv_vwadd_wv_i32m2(v1135, v1136, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1138 = __riscv_vwmul_vx_i16m1(v1087, v1134, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1139 = __riscv_vwadd_wv_i32m2(v1137, v1138, 8);
        v87 = v1139;
        vint32m2_t v1140 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1141 = __riscv_vwmul_vx_i16m1(v1093, v1131, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1142 = __riscv_vwadd_wv_i32m2(v1140, v1141, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1143 = __riscv_vwmul_vx_i16m1(v1096, v1134, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1144 = __riscv_vwadd_wv_i32m2(v1142, v1143, 8);
        v89 = v1144;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1145 = v40 + 59;
        const int8_t* v1146 = (const int8_t*) v1145;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1147 = *(const int8_t *)(v1146);
        const uint8_t* v1148 = v40 + 123;
        const int8_t* v1149 = (const int8_t*) v1148;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1150 = *(const int8_t *)(v1149);
        vint32m2_t v1151 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1152 = __riscv_vwmul_vx_i16m1(v1084, v1147, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1153 = __riscv_vwadd_wv_i32m2(v1151, v1152, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1154 = __riscv_vwmul_vx_i16m1(v1087, v1150, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1155 = __riscv_vwadd_wv_i32m2(v1153, v1154, 8);
        v91 = v1155;
        vint32m2_t v1156 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1157 = __riscv_vwmul_vx_i16m1(v1093, v1147, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1158 = __riscv_vwadd_wv_i32m2(v1156, v1157, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1159 = __riscv_vwmul_vx_i16m1(v1096, v1150, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1160 = __riscv_vwadd_wv_i32m2(v1158, v1159, 8);
        v93 = v1160;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1161 = v38 + 224;
        const uint8_t* v1162 = (const uint8_t*) v1161;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1163 = __riscv_vle8_v_u8mf2(v1162, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1164 = __riscv_vand_vx_u8mf2(v1163, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1165 = __riscv_vzext_vf2_u16m1(v1164, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1166 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1165, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1167 = __riscv_vsrl_vx_u8mf2(v1163, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1168 = __riscv_vzext_vf2_u16m1(v1167, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1169 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1168, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1170 = v38 + 232;
        const uint8_t* v1171 = (const uint8_t*) v1170;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1172 = __riscv_vle8_v_u8mf2(v1171, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1173 = __riscv_vand_vx_u8mf2(v1172, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1174 = __riscv_vzext_vf2_u16m1(v1173, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1175 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1174, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1176 = __riscv_vsrl_vx_u8mf2(v1172, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1177 = __riscv_vzext_vf2_u16m1(v1176, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1178 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1177, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1179 = v40 + 60;
        const int8_t* v1180 = (const int8_t*) v1179;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1181 = *(const int8_t *)(v1180);
        const uint8_t* v1182 = v40 + 124;
        const int8_t* v1183 = (const int8_t*) v1182;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1184 = *(const int8_t *)(v1183);
        vint32m2_t v1185 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1186 = __riscv_vwmul_vx_i16m1(v1166, v1181, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1187 = __riscv_vwadd_wv_i32m2(v1185, v1186, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1188 = __riscv_vwmul_vx_i16m1(v1169, v1184, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1189 = __riscv_vwadd_wv_i32m2(v1187, v1188, 8);
        v79 = v1189;
        vint32m2_t v1190 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1191 = __riscv_vwmul_vx_i16m1(v1175, v1181, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1192 = __riscv_vwadd_wv_i32m2(v1190, v1191, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1193 = __riscv_vwmul_vx_i16m1(v1178, v1184, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1194 = __riscv_vwadd_wv_i32m2(v1192, v1193, 8);
        v81 = v1194;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1195 = v40 + 61;
        const int8_t* v1196 = (const int8_t*) v1195;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1197 = *(const int8_t *)(v1196);
        const uint8_t* v1198 = v40 + 125;
        const int8_t* v1199 = (const int8_t*) v1198;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1200 = *(const int8_t *)(v1199);
        vint32m2_t v1201 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1202 = __riscv_vwmul_vx_i16m1(v1166, v1197, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1203 = __riscv_vwadd_wv_i32m2(v1201, v1202, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1204 = __riscv_vwmul_vx_i16m1(v1169, v1200, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1205 = __riscv_vwadd_wv_i32m2(v1203, v1204, 8);
        v83 = v1205;
        vint32m2_t v1206 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1207 = __riscv_vwmul_vx_i16m1(v1175, v1197, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1208 = __riscv_vwadd_wv_i32m2(v1206, v1207, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1209 = __riscv_vwmul_vx_i16m1(v1178, v1200, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1210 = __riscv_vwadd_wv_i32m2(v1208, v1209, 8);
        v85 = v1210;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1211 = v40 + 62;
        const int8_t* v1212 = (const int8_t*) v1211;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1213 = *(const int8_t *)(v1212);
        const uint8_t* v1214 = v40 + 126;
        const int8_t* v1215 = (const int8_t*) v1214;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1216 = *(const int8_t *)(v1215);
        vint32m2_t v1217 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1218 = __riscv_vwmul_vx_i16m1(v1166, v1213, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1219 = __riscv_vwadd_wv_i32m2(v1217, v1218, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1220 = __riscv_vwmul_vx_i16m1(v1169, v1216, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1221 = __riscv_vwadd_wv_i32m2(v1219, v1220, 8);
        v87 = v1221;
        vint32m2_t v1222 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1223 = __riscv_vwmul_vx_i16m1(v1175, v1213, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1224 = __riscv_vwadd_wv_i32m2(v1222, v1223, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1225 = __riscv_vwmul_vx_i16m1(v1178, v1216, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1226 = __riscv_vwadd_wv_i32m2(v1224, v1225, 8);
        v89 = v1226;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1227 = v40 + 63;
        const int8_t* v1228 = (const int8_t*) v1227;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1229 = *(const int8_t *)(v1228);
        const uint8_t* v1230 = v40 + 127;
        const int8_t* v1231 = (const int8_t*) v1230;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1232 = *(const int8_t *)(v1231);
        vint32m2_t v1233 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1234 = __riscv_vwmul_vx_i16m1(v1166, v1229, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1235 = __riscv_vwadd_wv_i32m2(v1233, v1234, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1236 = __riscv_vwmul_vx_i16m1(v1169, v1232, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1237 = __riscv_vwadd_wv_i32m2(v1235, v1236, 8);
        v91 = v1237;
        vint32m2_t v1238 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1239 = __riscv_vwmul_vx_i16m1(v1175, v1229, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1240 = __riscv_vwadd_wv_i32m2(v1238, v1239, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1241 = __riscv_vwmul_vx_i16m1(v1178, v1232, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1242 = __riscv_vwadd_wv_i32m2(v1240, v1241, 8);
        v93 = v1242;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1243 = v38 + 240;
        const uint8_t* v1244 = (const uint8_t*) v1243;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1245 = __riscv_vle8_v_u8mf2(v1244, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1246 = __riscv_vand_vx_u8mf2(v1245, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1247 = __riscv_vzext_vf2_u16m1(v1246, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1248 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1247, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1249 = __riscv_vsrl_vx_u8mf2(v1245, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1250 = __riscv_vzext_vf2_u16m1(v1249, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1251 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1250, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1252 = v38 + 248;
        const uint8_t* v1253 = (const uint8_t*) v1252;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1254 = __riscv_vle8_v_u8mf2(v1253, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1255 = __riscv_vand_vx_u8mf2(v1254, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1256 = __riscv_vzext_vf2_u16m1(v1255, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1257 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1256, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1258 = __riscv_vsrl_vx_u8mf2(v1254, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1259 = __riscv_vzext_vf2_u16m1(v1258, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1260 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1259, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1261 = v40 + 64;
        const int8_t* v1262 = (const int8_t*) v1261;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1263 = *(const int8_t *)(v1262);
        const uint8_t* v1264 = v40 + 128;
        const int8_t* v1265 = (const int8_t*) v1264;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1266 = *(const int8_t *)(v1265);
        vint32m2_t v1267 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1268 = __riscv_vwmul_vx_i16m1(v1248, v1263, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1269 = __riscv_vwadd_wv_i32m2(v1267, v1268, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1270 = __riscv_vwmul_vx_i16m1(v1251, v1266, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1271 = __riscv_vwadd_wv_i32m2(v1269, v1270, 8);
        v79 = v1271;
        vint32m2_t v1272 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1273 = __riscv_vwmul_vx_i16m1(v1257, v1263, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1274 = __riscv_vwadd_wv_i32m2(v1272, v1273, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1275 = __riscv_vwmul_vx_i16m1(v1260, v1266, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1276 = __riscv_vwadd_wv_i32m2(v1274, v1275, 8);
        v81 = v1276;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1277 = v40 + 65;
        const int8_t* v1278 = (const int8_t*) v1277;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1279 = *(const int8_t *)(v1278);
        const uint8_t* v1280 = v40 + 129;
        const int8_t* v1281 = (const int8_t*) v1280;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1282 = *(const int8_t *)(v1281);
        vint32m2_t v1283 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1284 = __riscv_vwmul_vx_i16m1(v1248, v1279, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1285 = __riscv_vwadd_wv_i32m2(v1283, v1284, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1286 = __riscv_vwmul_vx_i16m1(v1251, v1282, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1287 = __riscv_vwadd_wv_i32m2(v1285, v1286, 8);
        v83 = v1287;
        vint32m2_t v1288 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1289 = __riscv_vwmul_vx_i16m1(v1257, v1279, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1290 = __riscv_vwadd_wv_i32m2(v1288, v1289, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1291 = __riscv_vwmul_vx_i16m1(v1260, v1282, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1292 = __riscv_vwadd_wv_i32m2(v1290, v1291, 8);
        v85 = v1292;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1293 = v40 + 66;
        const int8_t* v1294 = (const int8_t*) v1293;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1295 = *(const int8_t *)(v1294);
        const uint8_t* v1296 = v40 + 130;
        const int8_t* v1297 = (const int8_t*) v1296;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1298 = *(const int8_t *)(v1297);
        vint32m2_t v1299 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1300 = __riscv_vwmul_vx_i16m1(v1248, v1295, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1301 = __riscv_vwadd_wv_i32m2(v1299, v1300, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1302 = __riscv_vwmul_vx_i16m1(v1251, v1298, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1303 = __riscv_vwadd_wv_i32m2(v1301, v1302, 8);
        v87 = v1303;
        vint32m2_t v1304 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1305 = __riscv_vwmul_vx_i16m1(v1257, v1295, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1306 = __riscv_vwadd_wv_i32m2(v1304, v1305, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1307 = __riscv_vwmul_vx_i16m1(v1260, v1298, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1308 = __riscv_vwadd_wv_i32m2(v1306, v1307, 8);
        v89 = v1308;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1309 = v40 + 67;
        const int8_t* v1310 = (const int8_t*) v1309;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1311 = *(const int8_t *)(v1310);
        const uint8_t* v1312 = v40 + 131;
        const int8_t* v1313 = (const int8_t*) v1312;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1314 = *(const int8_t *)(v1313);
        vint32m2_t v1315 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1316 = __riscv_vwmul_vx_i16m1(v1248, v1311, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1317 = __riscv_vwadd_wv_i32m2(v1315, v1316, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1318 = __riscv_vwmul_vx_i16m1(v1251, v1314, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1319 = __riscv_vwadd_wv_i32m2(v1317, v1318, 8);
        v91 = v1319;
        vint32m2_t v1320 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1321 = __riscv_vwmul_vx_i16m1(v1257, v1311, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1322 = __riscv_vwadd_wv_i32m2(v1320, v1321, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1323 = __riscv_vwmul_vx_i16m1(v1260, v1314, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1324 = __riscv_vwadd_wv_i32m2(v1322, v1323, 8);
        v93 = v1324;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1325 = v38 + 256;
        const uint8_t* v1326 = (const uint8_t*) v1325;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1327 = __riscv_vle8_v_u8mf2(v1326, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1328 = __riscv_vand_vx_u8mf2(v1327, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1329 = __riscv_vzext_vf2_u16m1(v1328, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1330 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1329, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1331 = __riscv_vsrl_vx_u8mf2(v1327, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1332 = __riscv_vzext_vf2_u16m1(v1331, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1333 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1332, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1334 = v38 + 264;
        const uint8_t* v1335 = (const uint8_t*) v1334;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1336 = __riscv_vle8_v_u8mf2(v1335, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1337 = __riscv_vand_vx_u8mf2(v1336, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1338 = __riscv_vzext_vf2_u16m1(v1337, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1339 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1338, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1340 = __riscv_vsrl_vx_u8mf2(v1336, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1341 = __riscv_vzext_vf2_u16m1(v1340, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1342 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v1341, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1343 = v40 + 68;
        const int8_t* v1344 = (const int8_t*) v1343;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1345 = *(const int8_t *)(v1344);
        const uint8_t* v1346 = v40 + 132;
        const int8_t* v1347 = (const int8_t*) v1346;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1348 = *(const int8_t *)(v1347);
        vint32m2_t v1349 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1350 = __riscv_vwmul_vx_i16m1(v1330, v1345, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1351 = __riscv_vwadd_wv_i32m2(v1349, v1350, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1352 = __riscv_vwmul_vx_i16m1(v1333, v1348, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1353 = __riscv_vwadd_wv_i32m2(v1351, v1352, 8);
        v79 = v1353;
        vint32m2_t v1354 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1355 = __riscv_vwmul_vx_i16m1(v1339, v1345, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1356 = __riscv_vwadd_wv_i32m2(v1354, v1355, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1357 = __riscv_vwmul_vx_i16m1(v1342, v1348, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1358 = __riscv_vwadd_wv_i32m2(v1356, v1357, 8);
        v81 = v1358;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1359 = v40 + 69;
        const int8_t* v1360 = (const int8_t*) v1359;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1361 = *(const int8_t *)(v1360);
        const uint8_t* v1362 = v40 + 133;
        const int8_t* v1363 = (const int8_t*) v1362;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1364 = *(const int8_t *)(v1363);
        vint32m2_t v1365 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1366 = __riscv_vwmul_vx_i16m1(v1330, v1361, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1367 = __riscv_vwadd_wv_i32m2(v1365, v1366, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1368 = __riscv_vwmul_vx_i16m1(v1333, v1364, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1369 = __riscv_vwadd_wv_i32m2(v1367, v1368, 8);
        v83 = v1369;
        vint32m2_t v1370 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1371 = __riscv_vwmul_vx_i16m1(v1339, v1361, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1372 = __riscv_vwadd_wv_i32m2(v1370, v1371, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1373 = __riscv_vwmul_vx_i16m1(v1342, v1364, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1374 = __riscv_vwadd_wv_i32m2(v1372, v1373, 8);
        v85 = v1374;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1375 = v40 + 70;
        const int8_t* v1376 = (const int8_t*) v1375;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1377 = *(const int8_t *)(v1376);
        const uint8_t* v1378 = v40 + 134;
        const int8_t* v1379 = (const int8_t*) v1378;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1380 = *(const int8_t *)(v1379);
        vint32m2_t v1381 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1382 = __riscv_vwmul_vx_i16m1(v1330, v1377, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1383 = __riscv_vwadd_wv_i32m2(v1381, v1382, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1384 = __riscv_vwmul_vx_i16m1(v1333, v1380, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1385 = __riscv_vwadd_wv_i32m2(v1383, v1384, 8);
        v87 = v1385;
        vint32m2_t v1386 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1387 = __riscv_vwmul_vx_i16m1(v1339, v1377, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1388 = __riscv_vwadd_wv_i32m2(v1386, v1387, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1389 = __riscv_vwmul_vx_i16m1(v1342, v1380, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1390 = __riscv_vwadd_wv_i32m2(v1388, v1389, 8);
        v89 = v1390;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1391 = v40 + 71;
        const int8_t* v1392 = (const int8_t*) v1391;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1393 = *(const int8_t *)(v1392);
        const uint8_t* v1394 = v40 + 135;
        const int8_t* v1395 = (const int8_t*) v1394;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1396 = *(const int8_t *)(v1395);
        vint32m2_t v1397 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1398 = __riscv_vwmul_vx_i16m1(v1330, v1393, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1399 = __riscv_vwadd_wv_i32m2(v1397, v1398, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1400 = __riscv_vwmul_vx_i16m1(v1333, v1396, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1401 = __riscv_vwadd_wv_i32m2(v1399, v1400, 8);
        v91 = v1401;
        vint32m2_t v1402 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1403 = __riscv_vwmul_vx_i16m1(v1339, v1393, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1404 = __riscv_vwadd_wv_i32m2(v1402, v1403, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1405 = __riscv_vwmul_vx_i16m1(v1342, v1396, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1406 = __riscv_vwadd_wv_i32m2(v1404, v1405, 8);
        v93 = v1406;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1407 = __riscv_vfmul_vf_f32m2(v66, v43, 8);
        vint32m2_t v1408 = v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1409 = __riscv_vfcvt_f_x_v_f32m2(v1408, 8);
        vfloat32m2_t v1410 = v20;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1411 = __riscv_vfmacc_vv_f32m2(v1410, v1409, v1407, 8);
        v20 = v1411;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1412 = __riscv_vfmul_vf_f32m2(v78, v43, 8);
        vint32m2_t v1413 = v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1414 = __riscv_vfcvt_f_x_v_f32m2(v1413, 8);
        vfloat32m2_t v1415 = v22;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1416 = __riscv_vfmacc_vv_f32m2(v1415, v1414, v1412, 8);
        v22 = v1416;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1417 = __riscv_vfmul_vf_f32m2(v66, v47, 8);
        vint32m2_t v1418 = v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1419 = __riscv_vfcvt_f_x_v_f32m2(v1418, 8);
        vfloat32m2_t v1420 = v24;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1421 = __riscv_vfmacc_vv_f32m2(v1420, v1419, v1417, 8);
        v24 = v1421;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1422 = __riscv_vfmul_vf_f32m2(v78, v47, 8);
        vint32m2_t v1423 = v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1424 = __riscv_vfcvt_f_x_v_f32m2(v1423, 8);
        vfloat32m2_t v1425 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1426 = __riscv_vfmacc_vv_f32m2(v1425, v1424, v1422, 8);
        v26 = v1426;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1427 = __riscv_vfmul_vf_f32m2(v66, v51, 8);
        vint32m2_t v1428 = v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1429 = __riscv_vfcvt_f_x_v_f32m2(v1428, 8);
        vfloat32m2_t v1430 = v28;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1431 = __riscv_vfmacc_vv_f32m2(v1430, v1429, v1427, 8);
        v28 = v1431;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1432 = __riscv_vfmul_vf_f32m2(v78, v51, 8);
        vint32m2_t v1433 = v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1434 = __riscv_vfcvt_f_x_v_f32m2(v1433, 8);
        vfloat32m2_t v1435 = v30;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1436 = __riscv_vfmacc_vv_f32m2(v1435, v1434, v1432, 8);
        v30 = v1436;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1437 = __riscv_vfmul_vf_f32m2(v66, v55, 8);
        vint32m2_t v1438 = v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1439 = __riscv_vfcvt_f_x_v_f32m2(v1438, 8);
        vfloat32m2_t v1440 = v32;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1441 = __riscv_vfmacc_vv_f32m2(v1440, v1439, v1437, 8);
        v32 = v1441;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1442 = __riscv_vfmul_vf_f32m2(v78, v55, 8);
        vint32m2_t v1443 = v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1444 = __riscv_vfcvt_f_x_v_f32m2(v1443, 8);
        vfloat32m2_t v1445 = v34;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1446 = __riscv_vfmacc_vv_f32m2(v1445, v1444, v1442, 8);
        v34 = v1446;
      }
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1447 = v12 * 4;
      size_t v1448 = v1447 + 0;
      size_t v1449 = v1448 * v7;
      size_t v1450 = v16 * 16;
      size_t v1451 = v1449 + v1450;
      float* v1452 = v2 + v1451;
      vfloat32m2_t v1453 = v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1452, v1453, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1454 = v12 * 4;
      size_t v1455 = v1454 + 0;
      size_t v1456 = v1455 * v7;
      size_t v1457 = v16 * 16;
      size_t v1458 = v1456 + v1457;
      size_t v1459 = v1458 + 8;
      float* v1460 = v2 + v1459;
      vfloat32m2_t v1461 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1460, v1461, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1462 = v12 * 4;
      size_t v1463 = v1462 + 1;
      size_t v1464 = v1463 * v7;
      size_t v1465 = v16 * 16;
      size_t v1466 = v1464 + v1465;
      float* v1467 = v2 + v1466;
      vfloat32m2_t v1468 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1467, v1468, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1469 = v12 * 4;
      size_t v1470 = v1469 + 1;
      size_t v1471 = v1470 * v7;
      size_t v1472 = v16 * 16;
      size_t v1473 = v1471 + v1472;
      size_t v1474 = v1473 + 8;
      float* v1475 = v2 + v1474;
      vfloat32m2_t v1476 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1475, v1476, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1477 = v12 * 4;
      size_t v1478 = v1477 + 2;
      size_t v1479 = v1478 * v7;
      size_t v1480 = v16 * 16;
      size_t v1481 = v1479 + v1480;
      float* v1482 = v2 + v1481;
      vfloat32m2_t v1483 = v28;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1482, v1483, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1484 = v12 * 4;
      size_t v1485 = v1484 + 2;
      size_t v1486 = v1485 * v7;
      size_t v1487 = v16 * 16;
      size_t v1488 = v1486 + v1487;
      size_t v1489 = v1488 + 8;
      float* v1490 = v2 + v1489;
      vfloat32m2_t v1491 = v30;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1490, v1491, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1492 = v12 * 4;
      size_t v1493 = v1492 + 3;
      size_t v1494 = v1493 * v7;
      size_t v1495 = v16 * 16;
      size_t v1496 = v1494 + v1495;
      float* v1497 = v2 + v1496;
      vfloat32m2_t v1498 = v32;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1497, v1498, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1499 = v12 * 4;
      size_t v1500 = v1499 + 3;
      size_t v1501 = v1500 * v7;
      size_t v1502 = v16 * 16;
      size_t v1503 = v1501 + v1502;
      size_t v1504 = v1503 + 8;
      float* v1505 = v2 + v1504;
      vfloat32m2_t v1506 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1505, v1506, 8);
    }
  }
  return;
}


