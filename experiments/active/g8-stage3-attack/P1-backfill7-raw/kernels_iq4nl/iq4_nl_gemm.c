#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5, size_t v6, size_t v7) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v8 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_iq4_nl_repack_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
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
      size_t v18 = v17 * 288;
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
        size_t v37 = v36 * 288;
        const uint8_t* v38 = v19 + v37;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
        size_t v39 = v36 * 136;
        const uint8_t* v40 = v15 + v39;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const _Float16* v41 = (const _Float16*) v40;
        _Float16 v42 = *(const _Float16 *)(v41);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v43 = v40 + 2;
        const _Float16* v44 = (const _Float16*) v43;
        _Float16 v45 = *(const _Float16 *)(v44);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v46 = v40 + 4;
        const _Float16* v47 = (const _Float16*) v46;
        _Float16 v48 = *(const _Float16 *)(v47);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v49 = v40 + 6;
        const _Float16* v50 = (const _Float16*) v49;
        _Float16 v51 = *(const _Float16 *)(v50);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const _Float16* v52 = (const _Float16*) v38;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v53 = __riscv_vle16_v_f16m1(v52, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const uint8_t* v54 = v38 + 16;
        const _Float16* v55 = (const _Float16*) v54;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v56 = __riscv_vle16_v_f16m1(v55, 8);
        vint32m2_t v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v58 = __riscv_vmv_v_x_i32m2(0, 8);
        v57 = v58;
        vint32m2_t v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v60 = __riscv_vmv_v_x_i32m2(0, 8);
        v59 = v60;
        vint32m2_t v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v62 = __riscv_vmv_v_x_i32m2(0, 8);
        v61 = v62;
        vint32m2_t v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v64 = __riscv_vmv_v_x_i32m2(0, 8);
        v63 = v64;
        vint32m2_t v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v66 = __riscv_vmv_v_x_i32m2(0, 8);
        v65 = v66;
        vint32m2_t v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v68 = __riscv_vmv_v_x_i32m2(0, 8);
        v67 = v68;
        vint32m2_t v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v70 = __riscv_vmv_v_x_i32m2(0, 8);
        v69 = v70;
        vint32m2_t v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v72 = __riscv_vmv_v_x_i32m2(0, 8);
        v71 = v72;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v73 = v38 + 32;
        const uint8_t* v74 = (const uint8_t*) v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v75 = __riscv_vle8_v_u8mf2(v74, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v76 = __riscv_vand_vx_u8mf2(v75, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v77 = __riscv_vzext_vf2_u16m1(v76, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v78 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v77, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v79 = __riscv_vsrl_vx_u8mf2(v75, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v80 = __riscv_vzext_vf2_u16m1(v79, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v81 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v80, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v82 = v38 + 40;
        const uint8_t* v83 = (const uint8_t*) v82;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v84 = __riscv_vle8_v_u8mf2(v83, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v85 = __riscv_vand_vx_u8mf2(v84, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v86 = __riscv_vzext_vf2_u16m1(v85, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v87 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v86, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v88 = __riscv_vsrl_vx_u8mf2(v84, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v89 = __riscv_vzext_vf2_u16m1(v88, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v90 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v89, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v91 = v40 + 8;
        const int8_t* v92 = (const int8_t*) v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v93 = *(const int8_t *)(v92);
        const uint8_t* v94 = v40 + 72;
        const int8_t* v95 = (const int8_t*) v94;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v96 = *(const int8_t *)(v95);
        vint32m2_t v97 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v98 = __riscv_vwmul_vx_i16m1(v78, v93, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v99 = __riscv_vwadd_wv_i32m2(v97, v98, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v100 = __riscv_vwmul_vx_i16m1(v81, v96, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v101 = __riscv_vwadd_wv_i32m2(v99, v100, 8);
        v57 = v101;
        vint32m2_t v102 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v103 = __riscv_vwmul_vx_i16m1(v87, v93, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v104 = __riscv_vwadd_wv_i32m2(v102, v103, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v105 = __riscv_vwmul_vx_i16m1(v90, v96, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v106 = __riscv_vwadd_wv_i32m2(v104, v105, 8);
        v59 = v106;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v107 = v40 + 9;
        const int8_t* v108 = (const int8_t*) v107;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v109 = *(const int8_t *)(v108);
        const uint8_t* v110 = v40 + 73;
        const int8_t* v111 = (const int8_t*) v110;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v112 = *(const int8_t *)(v111);
        vint32m2_t v113 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v114 = __riscv_vwmul_vx_i16m1(v78, v109, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v115 = __riscv_vwadd_wv_i32m2(v113, v114, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v116 = __riscv_vwmul_vx_i16m1(v81, v112, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v117 = __riscv_vwadd_wv_i32m2(v115, v116, 8);
        v61 = v117;
        vint32m2_t v118 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v119 = __riscv_vwmul_vx_i16m1(v87, v109, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v120 = __riscv_vwadd_wv_i32m2(v118, v119, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v121 = __riscv_vwmul_vx_i16m1(v90, v112, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v122 = __riscv_vwadd_wv_i32m2(v120, v121, 8);
        v63 = v122;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v123 = v40 + 10;
        const int8_t* v124 = (const int8_t*) v123;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v125 = *(const int8_t *)(v124);
        const uint8_t* v126 = v40 + 74;
        const int8_t* v127 = (const int8_t*) v126;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v128 = *(const int8_t *)(v127);
        vint32m2_t v129 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v130 = __riscv_vwmul_vx_i16m1(v78, v125, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v131 = __riscv_vwadd_wv_i32m2(v129, v130, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v132 = __riscv_vwmul_vx_i16m1(v81, v128, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v133 = __riscv_vwadd_wv_i32m2(v131, v132, 8);
        v65 = v133;
        vint32m2_t v134 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v135 = __riscv_vwmul_vx_i16m1(v87, v125, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v136 = __riscv_vwadd_wv_i32m2(v134, v135, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v137 = __riscv_vwmul_vx_i16m1(v90, v128, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v138 = __riscv_vwadd_wv_i32m2(v136, v137, 8);
        v67 = v138;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v139 = v40 + 11;
        const int8_t* v140 = (const int8_t*) v139;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v141 = *(const int8_t *)(v140);
        const uint8_t* v142 = v40 + 75;
        const int8_t* v143 = (const int8_t*) v142;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v144 = *(const int8_t *)(v143);
        vint32m2_t v145 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v146 = __riscv_vwmul_vx_i16m1(v78, v141, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v147 = __riscv_vwadd_wv_i32m2(v145, v146, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v148 = __riscv_vwmul_vx_i16m1(v81, v144, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v149 = __riscv_vwadd_wv_i32m2(v147, v148, 8);
        v69 = v149;
        vint32m2_t v150 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v151 = __riscv_vwmul_vx_i16m1(v87, v141, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v152 = __riscv_vwadd_wv_i32m2(v150, v151, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v153 = __riscv_vwmul_vx_i16m1(v90, v144, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v154 = __riscv_vwadd_wv_i32m2(v152, v153, 8);
        v71 = v154;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v155 = v38 + 48;
        const uint8_t* v156 = (const uint8_t*) v155;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v157 = __riscv_vle8_v_u8mf2(v156, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v158 = __riscv_vand_vx_u8mf2(v157, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v159 = __riscv_vzext_vf2_u16m1(v158, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v160 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v159, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v161 = __riscv_vsrl_vx_u8mf2(v157, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v162 = __riscv_vzext_vf2_u16m1(v161, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v163 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v162, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v164 = v38 + 56;
        const uint8_t* v165 = (const uint8_t*) v164;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v166 = __riscv_vle8_v_u8mf2(v165, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v167 = __riscv_vand_vx_u8mf2(v166, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v168 = __riscv_vzext_vf2_u16m1(v167, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v169 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v168, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v170 = __riscv_vsrl_vx_u8mf2(v166, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v171 = __riscv_vzext_vf2_u16m1(v170, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v172 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v171, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v173 = v40 + 12;
        const int8_t* v174 = (const int8_t*) v173;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v175 = *(const int8_t *)(v174);
        const uint8_t* v176 = v40 + 76;
        const int8_t* v177 = (const int8_t*) v176;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v178 = *(const int8_t *)(v177);
        vint32m2_t v179 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v180 = __riscv_vwmul_vx_i16m1(v160, v175, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v181 = __riscv_vwadd_wv_i32m2(v179, v180, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v182 = __riscv_vwmul_vx_i16m1(v163, v178, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v183 = __riscv_vwadd_wv_i32m2(v181, v182, 8);
        v57 = v183;
        vint32m2_t v184 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v185 = __riscv_vwmul_vx_i16m1(v169, v175, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v186 = __riscv_vwadd_wv_i32m2(v184, v185, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v187 = __riscv_vwmul_vx_i16m1(v172, v178, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v188 = __riscv_vwadd_wv_i32m2(v186, v187, 8);
        v59 = v188;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v189 = v40 + 13;
        const int8_t* v190 = (const int8_t*) v189;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v191 = *(const int8_t *)(v190);
        const uint8_t* v192 = v40 + 77;
        const int8_t* v193 = (const int8_t*) v192;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v194 = *(const int8_t *)(v193);
        vint32m2_t v195 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v196 = __riscv_vwmul_vx_i16m1(v160, v191, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v197 = __riscv_vwadd_wv_i32m2(v195, v196, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v198 = __riscv_vwmul_vx_i16m1(v163, v194, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v199 = __riscv_vwadd_wv_i32m2(v197, v198, 8);
        v61 = v199;
        vint32m2_t v200 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v201 = __riscv_vwmul_vx_i16m1(v169, v191, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v202 = __riscv_vwadd_wv_i32m2(v200, v201, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v203 = __riscv_vwmul_vx_i16m1(v172, v194, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v204 = __riscv_vwadd_wv_i32m2(v202, v203, 8);
        v63 = v204;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v205 = v40 + 14;
        const int8_t* v206 = (const int8_t*) v205;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v207 = *(const int8_t *)(v206);
        const uint8_t* v208 = v40 + 78;
        const int8_t* v209 = (const int8_t*) v208;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v210 = *(const int8_t *)(v209);
        vint32m2_t v211 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v212 = __riscv_vwmul_vx_i16m1(v160, v207, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v213 = __riscv_vwadd_wv_i32m2(v211, v212, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v214 = __riscv_vwmul_vx_i16m1(v163, v210, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v215 = __riscv_vwadd_wv_i32m2(v213, v214, 8);
        v65 = v215;
        vint32m2_t v216 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v217 = __riscv_vwmul_vx_i16m1(v169, v207, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v218 = __riscv_vwadd_wv_i32m2(v216, v217, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v219 = __riscv_vwmul_vx_i16m1(v172, v210, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v220 = __riscv_vwadd_wv_i32m2(v218, v219, 8);
        v67 = v220;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v221 = v40 + 15;
        const int8_t* v222 = (const int8_t*) v221;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v223 = *(const int8_t *)(v222);
        const uint8_t* v224 = v40 + 79;
        const int8_t* v225 = (const int8_t*) v224;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v226 = *(const int8_t *)(v225);
        vint32m2_t v227 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v228 = __riscv_vwmul_vx_i16m1(v160, v223, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v229 = __riscv_vwadd_wv_i32m2(v227, v228, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v230 = __riscv_vwmul_vx_i16m1(v163, v226, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v231 = __riscv_vwadd_wv_i32m2(v229, v230, 8);
        v69 = v231;
        vint32m2_t v232 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v233 = __riscv_vwmul_vx_i16m1(v169, v223, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v234 = __riscv_vwadd_wv_i32m2(v232, v233, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v235 = __riscv_vwmul_vx_i16m1(v172, v226, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v236 = __riscv_vwadd_wv_i32m2(v234, v235, 8);
        v71 = v236;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v237 = v38 + 64;
        const uint8_t* v238 = (const uint8_t*) v237;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v239 = __riscv_vle8_v_u8mf2(v238, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v240 = __riscv_vand_vx_u8mf2(v239, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v241 = __riscv_vzext_vf2_u16m1(v240, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v242 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v241, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v243 = __riscv_vsrl_vx_u8mf2(v239, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v244 = __riscv_vzext_vf2_u16m1(v243, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v245 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v244, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v246 = v38 + 72;
        const uint8_t* v247 = (const uint8_t*) v246;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v248 = __riscv_vle8_v_u8mf2(v247, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v249 = __riscv_vand_vx_u8mf2(v248, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v250 = __riscv_vzext_vf2_u16m1(v249, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v251 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v250, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v252 = __riscv_vsrl_vx_u8mf2(v248, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v253 = __riscv_vzext_vf2_u16m1(v252, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v254 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v253, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v255 = v40 + 16;
        const int8_t* v256 = (const int8_t*) v255;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v257 = *(const int8_t *)(v256);
        const uint8_t* v258 = v40 + 80;
        const int8_t* v259 = (const int8_t*) v258;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v260 = *(const int8_t *)(v259);
        vint32m2_t v261 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v262 = __riscv_vwmul_vx_i16m1(v242, v257, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v263 = __riscv_vwadd_wv_i32m2(v261, v262, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v264 = __riscv_vwmul_vx_i16m1(v245, v260, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v265 = __riscv_vwadd_wv_i32m2(v263, v264, 8);
        v57 = v265;
        vint32m2_t v266 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v267 = __riscv_vwmul_vx_i16m1(v251, v257, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v268 = __riscv_vwadd_wv_i32m2(v266, v267, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v269 = __riscv_vwmul_vx_i16m1(v254, v260, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v270 = __riscv_vwadd_wv_i32m2(v268, v269, 8);
        v59 = v270;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v271 = v40 + 17;
        const int8_t* v272 = (const int8_t*) v271;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v273 = *(const int8_t *)(v272);
        const uint8_t* v274 = v40 + 81;
        const int8_t* v275 = (const int8_t*) v274;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v276 = *(const int8_t *)(v275);
        vint32m2_t v277 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v278 = __riscv_vwmul_vx_i16m1(v242, v273, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v279 = __riscv_vwadd_wv_i32m2(v277, v278, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v280 = __riscv_vwmul_vx_i16m1(v245, v276, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v281 = __riscv_vwadd_wv_i32m2(v279, v280, 8);
        v61 = v281;
        vint32m2_t v282 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v283 = __riscv_vwmul_vx_i16m1(v251, v273, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v284 = __riscv_vwadd_wv_i32m2(v282, v283, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v285 = __riscv_vwmul_vx_i16m1(v254, v276, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v286 = __riscv_vwadd_wv_i32m2(v284, v285, 8);
        v63 = v286;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v287 = v40 + 18;
        const int8_t* v288 = (const int8_t*) v287;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v289 = *(const int8_t *)(v288);
        const uint8_t* v290 = v40 + 82;
        const int8_t* v291 = (const int8_t*) v290;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v292 = *(const int8_t *)(v291);
        vint32m2_t v293 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v294 = __riscv_vwmul_vx_i16m1(v242, v289, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v295 = __riscv_vwadd_wv_i32m2(v293, v294, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v296 = __riscv_vwmul_vx_i16m1(v245, v292, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v297 = __riscv_vwadd_wv_i32m2(v295, v296, 8);
        v65 = v297;
        vint32m2_t v298 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v299 = __riscv_vwmul_vx_i16m1(v251, v289, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v300 = __riscv_vwadd_wv_i32m2(v298, v299, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v301 = __riscv_vwmul_vx_i16m1(v254, v292, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v302 = __riscv_vwadd_wv_i32m2(v300, v301, 8);
        v67 = v302;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v303 = v40 + 19;
        const int8_t* v304 = (const int8_t*) v303;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v305 = *(const int8_t *)(v304);
        const uint8_t* v306 = v40 + 83;
        const int8_t* v307 = (const int8_t*) v306;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v308 = *(const int8_t *)(v307);
        vint32m2_t v309 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v310 = __riscv_vwmul_vx_i16m1(v242, v305, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v311 = __riscv_vwadd_wv_i32m2(v309, v310, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v312 = __riscv_vwmul_vx_i16m1(v245, v308, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v313 = __riscv_vwadd_wv_i32m2(v311, v312, 8);
        v69 = v313;
        vint32m2_t v314 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v315 = __riscv_vwmul_vx_i16m1(v251, v305, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v316 = __riscv_vwadd_wv_i32m2(v314, v315, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v317 = __riscv_vwmul_vx_i16m1(v254, v308, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v318 = __riscv_vwadd_wv_i32m2(v316, v317, 8);
        v71 = v318;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v319 = v38 + 80;
        const uint8_t* v320 = (const uint8_t*) v319;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v321 = __riscv_vle8_v_u8mf2(v320, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v322 = __riscv_vand_vx_u8mf2(v321, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v323 = __riscv_vzext_vf2_u16m1(v322, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v324 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v323, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v325 = __riscv_vsrl_vx_u8mf2(v321, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v326 = __riscv_vzext_vf2_u16m1(v325, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v327 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v326, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v328 = v38 + 88;
        const uint8_t* v329 = (const uint8_t*) v328;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v330 = __riscv_vle8_v_u8mf2(v329, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v331 = __riscv_vand_vx_u8mf2(v330, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v332 = __riscv_vzext_vf2_u16m1(v331, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v333 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v332, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v334 = __riscv_vsrl_vx_u8mf2(v330, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v335 = __riscv_vzext_vf2_u16m1(v334, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v336 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v335, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v337 = v40 + 20;
        const int8_t* v338 = (const int8_t*) v337;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v339 = *(const int8_t *)(v338);
        const uint8_t* v340 = v40 + 84;
        const int8_t* v341 = (const int8_t*) v340;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v342 = *(const int8_t *)(v341);
        vint32m2_t v343 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v344 = __riscv_vwmul_vx_i16m1(v324, v339, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v345 = __riscv_vwadd_wv_i32m2(v343, v344, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v346 = __riscv_vwmul_vx_i16m1(v327, v342, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v347 = __riscv_vwadd_wv_i32m2(v345, v346, 8);
        v57 = v347;
        vint32m2_t v348 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v349 = __riscv_vwmul_vx_i16m1(v333, v339, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v350 = __riscv_vwadd_wv_i32m2(v348, v349, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v351 = __riscv_vwmul_vx_i16m1(v336, v342, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v352 = __riscv_vwadd_wv_i32m2(v350, v351, 8);
        v59 = v352;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v353 = v40 + 21;
        const int8_t* v354 = (const int8_t*) v353;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v355 = *(const int8_t *)(v354);
        const uint8_t* v356 = v40 + 85;
        const int8_t* v357 = (const int8_t*) v356;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v358 = *(const int8_t *)(v357);
        vint32m2_t v359 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v360 = __riscv_vwmul_vx_i16m1(v324, v355, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v361 = __riscv_vwadd_wv_i32m2(v359, v360, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v362 = __riscv_vwmul_vx_i16m1(v327, v358, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v363 = __riscv_vwadd_wv_i32m2(v361, v362, 8);
        v61 = v363;
        vint32m2_t v364 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v365 = __riscv_vwmul_vx_i16m1(v333, v355, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v366 = __riscv_vwadd_wv_i32m2(v364, v365, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v367 = __riscv_vwmul_vx_i16m1(v336, v358, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v368 = __riscv_vwadd_wv_i32m2(v366, v367, 8);
        v63 = v368;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v369 = v40 + 22;
        const int8_t* v370 = (const int8_t*) v369;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v371 = *(const int8_t *)(v370);
        const uint8_t* v372 = v40 + 86;
        const int8_t* v373 = (const int8_t*) v372;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v374 = *(const int8_t *)(v373);
        vint32m2_t v375 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v376 = __riscv_vwmul_vx_i16m1(v324, v371, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v377 = __riscv_vwadd_wv_i32m2(v375, v376, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v378 = __riscv_vwmul_vx_i16m1(v327, v374, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v379 = __riscv_vwadd_wv_i32m2(v377, v378, 8);
        v65 = v379;
        vint32m2_t v380 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v381 = __riscv_vwmul_vx_i16m1(v333, v371, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v382 = __riscv_vwadd_wv_i32m2(v380, v381, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v383 = __riscv_vwmul_vx_i16m1(v336, v374, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v384 = __riscv_vwadd_wv_i32m2(v382, v383, 8);
        v67 = v384;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v385 = v40 + 23;
        const int8_t* v386 = (const int8_t*) v385;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v387 = *(const int8_t *)(v386);
        const uint8_t* v388 = v40 + 87;
        const int8_t* v389 = (const int8_t*) v388;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v390 = *(const int8_t *)(v389);
        vint32m2_t v391 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v392 = __riscv_vwmul_vx_i16m1(v324, v387, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v393 = __riscv_vwadd_wv_i32m2(v391, v392, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v394 = __riscv_vwmul_vx_i16m1(v327, v390, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v395 = __riscv_vwadd_wv_i32m2(v393, v394, 8);
        v69 = v395;
        vint32m2_t v396 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v397 = __riscv_vwmul_vx_i16m1(v333, v387, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v398 = __riscv_vwadd_wv_i32m2(v396, v397, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v399 = __riscv_vwmul_vx_i16m1(v336, v390, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v400 = __riscv_vwadd_wv_i32m2(v398, v399, 8);
        v71 = v400;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v401 = v38 + 96;
        const uint8_t* v402 = (const uint8_t*) v401;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v403 = __riscv_vle8_v_u8mf2(v402, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v404 = __riscv_vand_vx_u8mf2(v403, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v405 = __riscv_vzext_vf2_u16m1(v404, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v406 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v405, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v407 = __riscv_vsrl_vx_u8mf2(v403, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v408 = __riscv_vzext_vf2_u16m1(v407, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v409 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v408, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v410 = v38 + 104;
        const uint8_t* v411 = (const uint8_t*) v410;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v412 = __riscv_vle8_v_u8mf2(v411, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v413 = __riscv_vand_vx_u8mf2(v412, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v414 = __riscv_vzext_vf2_u16m1(v413, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v415 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v414, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v416 = __riscv_vsrl_vx_u8mf2(v412, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v417 = __riscv_vzext_vf2_u16m1(v416, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v418 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v417, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v419 = v40 + 24;
        const int8_t* v420 = (const int8_t*) v419;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v421 = *(const int8_t *)(v420);
        const uint8_t* v422 = v40 + 88;
        const int8_t* v423 = (const int8_t*) v422;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v424 = *(const int8_t *)(v423);
        vint32m2_t v425 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v426 = __riscv_vwmul_vx_i16m1(v406, v421, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v427 = __riscv_vwadd_wv_i32m2(v425, v426, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v428 = __riscv_vwmul_vx_i16m1(v409, v424, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v429 = __riscv_vwadd_wv_i32m2(v427, v428, 8);
        v57 = v429;
        vint32m2_t v430 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v431 = __riscv_vwmul_vx_i16m1(v415, v421, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v432 = __riscv_vwadd_wv_i32m2(v430, v431, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v433 = __riscv_vwmul_vx_i16m1(v418, v424, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v434 = __riscv_vwadd_wv_i32m2(v432, v433, 8);
        v59 = v434;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v435 = v40 + 25;
        const int8_t* v436 = (const int8_t*) v435;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v437 = *(const int8_t *)(v436);
        const uint8_t* v438 = v40 + 89;
        const int8_t* v439 = (const int8_t*) v438;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v440 = *(const int8_t *)(v439);
        vint32m2_t v441 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v442 = __riscv_vwmul_vx_i16m1(v406, v437, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v443 = __riscv_vwadd_wv_i32m2(v441, v442, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v444 = __riscv_vwmul_vx_i16m1(v409, v440, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v445 = __riscv_vwadd_wv_i32m2(v443, v444, 8);
        v61 = v445;
        vint32m2_t v446 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v447 = __riscv_vwmul_vx_i16m1(v415, v437, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v448 = __riscv_vwadd_wv_i32m2(v446, v447, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v449 = __riscv_vwmul_vx_i16m1(v418, v440, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v450 = __riscv_vwadd_wv_i32m2(v448, v449, 8);
        v63 = v450;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v451 = v40 + 26;
        const int8_t* v452 = (const int8_t*) v451;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v453 = *(const int8_t *)(v452);
        const uint8_t* v454 = v40 + 90;
        const int8_t* v455 = (const int8_t*) v454;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v456 = *(const int8_t *)(v455);
        vint32m2_t v457 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v458 = __riscv_vwmul_vx_i16m1(v406, v453, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v459 = __riscv_vwadd_wv_i32m2(v457, v458, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v460 = __riscv_vwmul_vx_i16m1(v409, v456, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v461 = __riscv_vwadd_wv_i32m2(v459, v460, 8);
        v65 = v461;
        vint32m2_t v462 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v463 = __riscv_vwmul_vx_i16m1(v415, v453, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v464 = __riscv_vwadd_wv_i32m2(v462, v463, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v465 = __riscv_vwmul_vx_i16m1(v418, v456, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v466 = __riscv_vwadd_wv_i32m2(v464, v465, 8);
        v67 = v466;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v467 = v40 + 27;
        const int8_t* v468 = (const int8_t*) v467;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v469 = *(const int8_t *)(v468);
        const uint8_t* v470 = v40 + 91;
        const int8_t* v471 = (const int8_t*) v470;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v472 = *(const int8_t *)(v471);
        vint32m2_t v473 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v474 = __riscv_vwmul_vx_i16m1(v406, v469, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v475 = __riscv_vwadd_wv_i32m2(v473, v474, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v476 = __riscv_vwmul_vx_i16m1(v409, v472, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v477 = __riscv_vwadd_wv_i32m2(v475, v476, 8);
        v69 = v477;
        vint32m2_t v478 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v479 = __riscv_vwmul_vx_i16m1(v415, v469, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v480 = __riscv_vwadd_wv_i32m2(v478, v479, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v481 = __riscv_vwmul_vx_i16m1(v418, v472, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v482 = __riscv_vwadd_wv_i32m2(v480, v481, 8);
        v71 = v482;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v483 = v38 + 112;
        const uint8_t* v484 = (const uint8_t*) v483;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v485 = __riscv_vle8_v_u8mf2(v484, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v486 = __riscv_vand_vx_u8mf2(v485, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v487 = __riscv_vzext_vf2_u16m1(v486, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v488 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v487, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v489 = __riscv_vsrl_vx_u8mf2(v485, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v490 = __riscv_vzext_vf2_u16m1(v489, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v491 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v490, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v492 = v38 + 120;
        const uint8_t* v493 = (const uint8_t*) v492;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v494 = __riscv_vle8_v_u8mf2(v493, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v495 = __riscv_vand_vx_u8mf2(v494, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v496 = __riscv_vzext_vf2_u16m1(v495, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v497 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v496, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v498 = __riscv_vsrl_vx_u8mf2(v494, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v499 = __riscv_vzext_vf2_u16m1(v498, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v500 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v499, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v501 = v40 + 28;
        const int8_t* v502 = (const int8_t*) v501;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v503 = *(const int8_t *)(v502);
        const uint8_t* v504 = v40 + 92;
        const int8_t* v505 = (const int8_t*) v504;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v506 = *(const int8_t *)(v505);
        vint32m2_t v507 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v508 = __riscv_vwmul_vx_i16m1(v488, v503, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v509 = __riscv_vwadd_wv_i32m2(v507, v508, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v510 = __riscv_vwmul_vx_i16m1(v491, v506, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v511 = __riscv_vwadd_wv_i32m2(v509, v510, 8);
        v57 = v511;
        vint32m2_t v512 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v513 = __riscv_vwmul_vx_i16m1(v497, v503, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v514 = __riscv_vwadd_wv_i32m2(v512, v513, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v515 = __riscv_vwmul_vx_i16m1(v500, v506, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v516 = __riscv_vwadd_wv_i32m2(v514, v515, 8);
        v59 = v516;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v517 = v40 + 29;
        const int8_t* v518 = (const int8_t*) v517;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v519 = *(const int8_t *)(v518);
        const uint8_t* v520 = v40 + 93;
        const int8_t* v521 = (const int8_t*) v520;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v522 = *(const int8_t *)(v521);
        vint32m2_t v523 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v524 = __riscv_vwmul_vx_i16m1(v488, v519, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v525 = __riscv_vwadd_wv_i32m2(v523, v524, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v526 = __riscv_vwmul_vx_i16m1(v491, v522, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v527 = __riscv_vwadd_wv_i32m2(v525, v526, 8);
        v61 = v527;
        vint32m2_t v528 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v529 = __riscv_vwmul_vx_i16m1(v497, v519, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v530 = __riscv_vwadd_wv_i32m2(v528, v529, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v531 = __riscv_vwmul_vx_i16m1(v500, v522, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v532 = __riscv_vwadd_wv_i32m2(v530, v531, 8);
        v63 = v532;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v533 = v40 + 30;
        const int8_t* v534 = (const int8_t*) v533;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v535 = *(const int8_t *)(v534);
        const uint8_t* v536 = v40 + 94;
        const int8_t* v537 = (const int8_t*) v536;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v538 = *(const int8_t *)(v537);
        vint32m2_t v539 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v540 = __riscv_vwmul_vx_i16m1(v488, v535, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v541 = __riscv_vwadd_wv_i32m2(v539, v540, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v542 = __riscv_vwmul_vx_i16m1(v491, v538, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v543 = __riscv_vwadd_wv_i32m2(v541, v542, 8);
        v65 = v543;
        vint32m2_t v544 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v545 = __riscv_vwmul_vx_i16m1(v497, v535, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v546 = __riscv_vwadd_wv_i32m2(v544, v545, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v547 = __riscv_vwmul_vx_i16m1(v500, v538, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v548 = __riscv_vwadd_wv_i32m2(v546, v547, 8);
        v67 = v548;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v549 = v40 + 31;
        const int8_t* v550 = (const int8_t*) v549;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v551 = *(const int8_t *)(v550);
        const uint8_t* v552 = v40 + 95;
        const int8_t* v553 = (const int8_t*) v552;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v554 = *(const int8_t *)(v553);
        vint32m2_t v555 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v556 = __riscv_vwmul_vx_i16m1(v488, v551, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v557 = __riscv_vwadd_wv_i32m2(v555, v556, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v558 = __riscv_vwmul_vx_i16m1(v491, v554, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v559 = __riscv_vwadd_wv_i32m2(v557, v558, 8);
        v69 = v559;
        vint32m2_t v560 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v561 = __riscv_vwmul_vx_i16m1(v497, v551, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v562 = __riscv_vwadd_wv_i32m2(v560, v561, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v563 = __riscv_vwmul_vx_i16m1(v500, v554, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v564 = __riscv_vwadd_wv_i32m2(v562, v563, 8);
        v71 = v564;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v565 = v38 + 128;
        const uint8_t* v566 = (const uint8_t*) v565;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v567 = __riscv_vle8_v_u8mf2(v566, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v568 = __riscv_vand_vx_u8mf2(v567, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v569 = __riscv_vzext_vf2_u16m1(v568, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v570 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v569, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v571 = __riscv_vsrl_vx_u8mf2(v567, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v572 = __riscv_vzext_vf2_u16m1(v571, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v573 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v572, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v574 = v38 + 136;
        const uint8_t* v575 = (const uint8_t*) v574;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v576 = __riscv_vle8_v_u8mf2(v575, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v577 = __riscv_vand_vx_u8mf2(v576, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v578 = __riscv_vzext_vf2_u16m1(v577, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v579 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v578, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v580 = __riscv_vsrl_vx_u8mf2(v576, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v581 = __riscv_vzext_vf2_u16m1(v580, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v582 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v581, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v583 = v40 + 32;
        const int8_t* v584 = (const int8_t*) v583;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v585 = *(const int8_t *)(v584);
        const uint8_t* v586 = v40 + 96;
        const int8_t* v587 = (const int8_t*) v586;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v588 = *(const int8_t *)(v587);
        vint32m2_t v589 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v590 = __riscv_vwmul_vx_i16m1(v570, v585, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v591 = __riscv_vwadd_wv_i32m2(v589, v590, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v592 = __riscv_vwmul_vx_i16m1(v573, v588, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v593 = __riscv_vwadd_wv_i32m2(v591, v592, 8);
        v57 = v593;
        vint32m2_t v594 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v595 = __riscv_vwmul_vx_i16m1(v579, v585, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v596 = __riscv_vwadd_wv_i32m2(v594, v595, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v597 = __riscv_vwmul_vx_i16m1(v582, v588, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v598 = __riscv_vwadd_wv_i32m2(v596, v597, 8);
        v59 = v598;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v599 = v40 + 33;
        const int8_t* v600 = (const int8_t*) v599;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v601 = *(const int8_t *)(v600);
        const uint8_t* v602 = v40 + 97;
        const int8_t* v603 = (const int8_t*) v602;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v604 = *(const int8_t *)(v603);
        vint32m2_t v605 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v606 = __riscv_vwmul_vx_i16m1(v570, v601, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v607 = __riscv_vwadd_wv_i32m2(v605, v606, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v608 = __riscv_vwmul_vx_i16m1(v573, v604, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v609 = __riscv_vwadd_wv_i32m2(v607, v608, 8);
        v61 = v609;
        vint32m2_t v610 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v611 = __riscv_vwmul_vx_i16m1(v579, v601, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v612 = __riscv_vwadd_wv_i32m2(v610, v611, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v613 = __riscv_vwmul_vx_i16m1(v582, v604, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v614 = __riscv_vwadd_wv_i32m2(v612, v613, 8);
        v63 = v614;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v615 = v40 + 34;
        const int8_t* v616 = (const int8_t*) v615;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v617 = *(const int8_t *)(v616);
        const uint8_t* v618 = v40 + 98;
        const int8_t* v619 = (const int8_t*) v618;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v620 = *(const int8_t *)(v619);
        vint32m2_t v621 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v622 = __riscv_vwmul_vx_i16m1(v570, v617, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v623 = __riscv_vwadd_wv_i32m2(v621, v622, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v624 = __riscv_vwmul_vx_i16m1(v573, v620, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v625 = __riscv_vwadd_wv_i32m2(v623, v624, 8);
        v65 = v625;
        vint32m2_t v626 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v627 = __riscv_vwmul_vx_i16m1(v579, v617, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v628 = __riscv_vwadd_wv_i32m2(v626, v627, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v629 = __riscv_vwmul_vx_i16m1(v582, v620, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v630 = __riscv_vwadd_wv_i32m2(v628, v629, 8);
        v67 = v630;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v631 = v40 + 35;
        const int8_t* v632 = (const int8_t*) v631;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v633 = *(const int8_t *)(v632);
        const uint8_t* v634 = v40 + 99;
        const int8_t* v635 = (const int8_t*) v634;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v636 = *(const int8_t *)(v635);
        vint32m2_t v637 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v638 = __riscv_vwmul_vx_i16m1(v570, v633, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v639 = __riscv_vwadd_wv_i32m2(v637, v638, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v640 = __riscv_vwmul_vx_i16m1(v573, v636, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v641 = __riscv_vwadd_wv_i32m2(v639, v640, 8);
        v69 = v641;
        vint32m2_t v642 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v643 = __riscv_vwmul_vx_i16m1(v579, v633, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v644 = __riscv_vwadd_wv_i32m2(v642, v643, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v645 = __riscv_vwmul_vx_i16m1(v582, v636, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v646 = __riscv_vwadd_wv_i32m2(v644, v645, 8);
        v71 = v646;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v647 = v38 + 144;
        const uint8_t* v648 = (const uint8_t*) v647;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v649 = __riscv_vle8_v_u8mf2(v648, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v650 = __riscv_vand_vx_u8mf2(v649, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v651 = __riscv_vzext_vf2_u16m1(v650, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v652 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v651, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v653 = __riscv_vsrl_vx_u8mf2(v649, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v654 = __riscv_vzext_vf2_u16m1(v653, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v655 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v654, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v656 = v38 + 152;
        const uint8_t* v657 = (const uint8_t*) v656;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v658 = __riscv_vle8_v_u8mf2(v657, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v659 = __riscv_vand_vx_u8mf2(v658, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v660 = __riscv_vzext_vf2_u16m1(v659, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v661 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v660, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v662 = __riscv_vsrl_vx_u8mf2(v658, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v663 = __riscv_vzext_vf2_u16m1(v662, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v664 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v663, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v665 = v40 + 36;
        const int8_t* v666 = (const int8_t*) v665;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v667 = *(const int8_t *)(v666);
        const uint8_t* v668 = v40 + 100;
        const int8_t* v669 = (const int8_t*) v668;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v670 = *(const int8_t *)(v669);
        vint32m2_t v671 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v672 = __riscv_vwmul_vx_i16m1(v652, v667, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v673 = __riscv_vwadd_wv_i32m2(v671, v672, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v674 = __riscv_vwmul_vx_i16m1(v655, v670, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v675 = __riscv_vwadd_wv_i32m2(v673, v674, 8);
        v57 = v675;
        vint32m2_t v676 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v677 = __riscv_vwmul_vx_i16m1(v661, v667, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v678 = __riscv_vwadd_wv_i32m2(v676, v677, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v679 = __riscv_vwmul_vx_i16m1(v664, v670, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v680 = __riscv_vwadd_wv_i32m2(v678, v679, 8);
        v59 = v680;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v681 = v40 + 37;
        const int8_t* v682 = (const int8_t*) v681;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v683 = *(const int8_t *)(v682);
        const uint8_t* v684 = v40 + 101;
        const int8_t* v685 = (const int8_t*) v684;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v686 = *(const int8_t *)(v685);
        vint32m2_t v687 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v688 = __riscv_vwmul_vx_i16m1(v652, v683, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v689 = __riscv_vwadd_wv_i32m2(v687, v688, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v690 = __riscv_vwmul_vx_i16m1(v655, v686, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v691 = __riscv_vwadd_wv_i32m2(v689, v690, 8);
        v61 = v691;
        vint32m2_t v692 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v693 = __riscv_vwmul_vx_i16m1(v661, v683, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v694 = __riscv_vwadd_wv_i32m2(v692, v693, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v695 = __riscv_vwmul_vx_i16m1(v664, v686, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v696 = __riscv_vwadd_wv_i32m2(v694, v695, 8);
        v63 = v696;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v697 = v40 + 38;
        const int8_t* v698 = (const int8_t*) v697;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v699 = *(const int8_t *)(v698);
        const uint8_t* v700 = v40 + 102;
        const int8_t* v701 = (const int8_t*) v700;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v702 = *(const int8_t *)(v701);
        vint32m2_t v703 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v704 = __riscv_vwmul_vx_i16m1(v652, v699, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v705 = __riscv_vwadd_wv_i32m2(v703, v704, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v706 = __riscv_vwmul_vx_i16m1(v655, v702, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v707 = __riscv_vwadd_wv_i32m2(v705, v706, 8);
        v65 = v707;
        vint32m2_t v708 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v709 = __riscv_vwmul_vx_i16m1(v661, v699, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v710 = __riscv_vwadd_wv_i32m2(v708, v709, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v711 = __riscv_vwmul_vx_i16m1(v664, v702, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v712 = __riscv_vwadd_wv_i32m2(v710, v711, 8);
        v67 = v712;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v713 = v40 + 39;
        const int8_t* v714 = (const int8_t*) v713;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v715 = *(const int8_t *)(v714);
        const uint8_t* v716 = v40 + 103;
        const int8_t* v717 = (const int8_t*) v716;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v718 = *(const int8_t *)(v717);
        vint32m2_t v719 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v720 = __riscv_vwmul_vx_i16m1(v652, v715, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v721 = __riscv_vwadd_wv_i32m2(v719, v720, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v722 = __riscv_vwmul_vx_i16m1(v655, v718, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v723 = __riscv_vwadd_wv_i32m2(v721, v722, 8);
        v69 = v723;
        vint32m2_t v724 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v725 = __riscv_vwmul_vx_i16m1(v661, v715, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v726 = __riscv_vwadd_wv_i32m2(v724, v725, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v727 = __riscv_vwmul_vx_i16m1(v664, v718, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v728 = __riscv_vwadd_wv_i32m2(v726, v727, 8);
        v71 = v728;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v729 = v38 + 160;
        const uint8_t* v730 = (const uint8_t*) v729;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v731 = __riscv_vle8_v_u8mf2(v730, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v732 = __riscv_vand_vx_u8mf2(v731, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v733 = __riscv_vzext_vf2_u16m1(v732, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v734 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v733, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v735 = __riscv_vsrl_vx_u8mf2(v731, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v736 = __riscv_vzext_vf2_u16m1(v735, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v737 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v736, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v738 = v38 + 168;
        const uint8_t* v739 = (const uint8_t*) v738;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v740 = __riscv_vle8_v_u8mf2(v739, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v741 = __riscv_vand_vx_u8mf2(v740, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v742 = __riscv_vzext_vf2_u16m1(v741, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v743 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v742, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v744 = __riscv_vsrl_vx_u8mf2(v740, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v745 = __riscv_vzext_vf2_u16m1(v744, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v746 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v745, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v747 = v40 + 40;
        const int8_t* v748 = (const int8_t*) v747;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v749 = *(const int8_t *)(v748);
        const uint8_t* v750 = v40 + 104;
        const int8_t* v751 = (const int8_t*) v750;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v752 = *(const int8_t *)(v751);
        vint32m2_t v753 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v754 = __riscv_vwmul_vx_i16m1(v734, v749, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v755 = __riscv_vwadd_wv_i32m2(v753, v754, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v756 = __riscv_vwmul_vx_i16m1(v737, v752, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v757 = __riscv_vwadd_wv_i32m2(v755, v756, 8);
        v57 = v757;
        vint32m2_t v758 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v759 = __riscv_vwmul_vx_i16m1(v743, v749, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v760 = __riscv_vwadd_wv_i32m2(v758, v759, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v761 = __riscv_vwmul_vx_i16m1(v746, v752, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v762 = __riscv_vwadd_wv_i32m2(v760, v761, 8);
        v59 = v762;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v763 = v40 + 41;
        const int8_t* v764 = (const int8_t*) v763;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v765 = *(const int8_t *)(v764);
        const uint8_t* v766 = v40 + 105;
        const int8_t* v767 = (const int8_t*) v766;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v768 = *(const int8_t *)(v767);
        vint32m2_t v769 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v770 = __riscv_vwmul_vx_i16m1(v734, v765, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v771 = __riscv_vwadd_wv_i32m2(v769, v770, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v772 = __riscv_vwmul_vx_i16m1(v737, v768, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v773 = __riscv_vwadd_wv_i32m2(v771, v772, 8);
        v61 = v773;
        vint32m2_t v774 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v775 = __riscv_vwmul_vx_i16m1(v743, v765, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v776 = __riscv_vwadd_wv_i32m2(v774, v775, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v777 = __riscv_vwmul_vx_i16m1(v746, v768, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v778 = __riscv_vwadd_wv_i32m2(v776, v777, 8);
        v63 = v778;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v779 = v40 + 42;
        const int8_t* v780 = (const int8_t*) v779;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v781 = *(const int8_t *)(v780);
        const uint8_t* v782 = v40 + 106;
        const int8_t* v783 = (const int8_t*) v782;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v784 = *(const int8_t *)(v783);
        vint32m2_t v785 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v786 = __riscv_vwmul_vx_i16m1(v734, v781, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v787 = __riscv_vwadd_wv_i32m2(v785, v786, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v788 = __riscv_vwmul_vx_i16m1(v737, v784, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v789 = __riscv_vwadd_wv_i32m2(v787, v788, 8);
        v65 = v789;
        vint32m2_t v790 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v791 = __riscv_vwmul_vx_i16m1(v743, v781, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v792 = __riscv_vwadd_wv_i32m2(v790, v791, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v793 = __riscv_vwmul_vx_i16m1(v746, v784, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v794 = __riscv_vwadd_wv_i32m2(v792, v793, 8);
        v67 = v794;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v795 = v40 + 43;
        const int8_t* v796 = (const int8_t*) v795;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v797 = *(const int8_t *)(v796);
        const uint8_t* v798 = v40 + 107;
        const int8_t* v799 = (const int8_t*) v798;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v800 = *(const int8_t *)(v799);
        vint32m2_t v801 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v802 = __riscv_vwmul_vx_i16m1(v734, v797, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v803 = __riscv_vwadd_wv_i32m2(v801, v802, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v804 = __riscv_vwmul_vx_i16m1(v737, v800, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v805 = __riscv_vwadd_wv_i32m2(v803, v804, 8);
        v69 = v805;
        vint32m2_t v806 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v807 = __riscv_vwmul_vx_i16m1(v743, v797, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v808 = __riscv_vwadd_wv_i32m2(v806, v807, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v809 = __riscv_vwmul_vx_i16m1(v746, v800, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v810 = __riscv_vwadd_wv_i32m2(v808, v809, 8);
        v71 = v810;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v811 = v38 + 176;
        const uint8_t* v812 = (const uint8_t*) v811;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v813 = __riscv_vle8_v_u8mf2(v812, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v814 = __riscv_vand_vx_u8mf2(v813, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v815 = __riscv_vzext_vf2_u16m1(v814, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v816 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v815, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v817 = __riscv_vsrl_vx_u8mf2(v813, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v818 = __riscv_vzext_vf2_u16m1(v817, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v819 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v818, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v820 = v38 + 184;
        const uint8_t* v821 = (const uint8_t*) v820;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v822 = __riscv_vle8_v_u8mf2(v821, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v823 = __riscv_vand_vx_u8mf2(v822, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v824 = __riscv_vzext_vf2_u16m1(v823, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v825 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v824, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v826 = __riscv_vsrl_vx_u8mf2(v822, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v827 = __riscv_vzext_vf2_u16m1(v826, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v828 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v827, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v829 = v40 + 44;
        const int8_t* v830 = (const int8_t*) v829;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v831 = *(const int8_t *)(v830);
        const uint8_t* v832 = v40 + 108;
        const int8_t* v833 = (const int8_t*) v832;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v834 = *(const int8_t *)(v833);
        vint32m2_t v835 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v836 = __riscv_vwmul_vx_i16m1(v816, v831, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v837 = __riscv_vwadd_wv_i32m2(v835, v836, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v838 = __riscv_vwmul_vx_i16m1(v819, v834, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v839 = __riscv_vwadd_wv_i32m2(v837, v838, 8);
        v57 = v839;
        vint32m2_t v840 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v841 = __riscv_vwmul_vx_i16m1(v825, v831, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v842 = __riscv_vwadd_wv_i32m2(v840, v841, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v843 = __riscv_vwmul_vx_i16m1(v828, v834, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v844 = __riscv_vwadd_wv_i32m2(v842, v843, 8);
        v59 = v844;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v845 = v40 + 45;
        const int8_t* v846 = (const int8_t*) v845;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v847 = *(const int8_t *)(v846);
        const uint8_t* v848 = v40 + 109;
        const int8_t* v849 = (const int8_t*) v848;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v850 = *(const int8_t *)(v849);
        vint32m2_t v851 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v852 = __riscv_vwmul_vx_i16m1(v816, v847, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v853 = __riscv_vwadd_wv_i32m2(v851, v852, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v854 = __riscv_vwmul_vx_i16m1(v819, v850, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v855 = __riscv_vwadd_wv_i32m2(v853, v854, 8);
        v61 = v855;
        vint32m2_t v856 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v857 = __riscv_vwmul_vx_i16m1(v825, v847, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v858 = __riscv_vwadd_wv_i32m2(v856, v857, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v859 = __riscv_vwmul_vx_i16m1(v828, v850, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v860 = __riscv_vwadd_wv_i32m2(v858, v859, 8);
        v63 = v860;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v861 = v40 + 46;
        const int8_t* v862 = (const int8_t*) v861;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v863 = *(const int8_t *)(v862);
        const uint8_t* v864 = v40 + 110;
        const int8_t* v865 = (const int8_t*) v864;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v866 = *(const int8_t *)(v865);
        vint32m2_t v867 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v868 = __riscv_vwmul_vx_i16m1(v816, v863, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v869 = __riscv_vwadd_wv_i32m2(v867, v868, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v870 = __riscv_vwmul_vx_i16m1(v819, v866, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v871 = __riscv_vwadd_wv_i32m2(v869, v870, 8);
        v65 = v871;
        vint32m2_t v872 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v873 = __riscv_vwmul_vx_i16m1(v825, v863, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v874 = __riscv_vwadd_wv_i32m2(v872, v873, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v875 = __riscv_vwmul_vx_i16m1(v828, v866, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v876 = __riscv_vwadd_wv_i32m2(v874, v875, 8);
        v67 = v876;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v877 = v40 + 47;
        const int8_t* v878 = (const int8_t*) v877;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v879 = *(const int8_t *)(v878);
        const uint8_t* v880 = v40 + 111;
        const int8_t* v881 = (const int8_t*) v880;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v882 = *(const int8_t *)(v881);
        vint32m2_t v883 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v884 = __riscv_vwmul_vx_i16m1(v816, v879, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v885 = __riscv_vwadd_wv_i32m2(v883, v884, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v886 = __riscv_vwmul_vx_i16m1(v819, v882, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v887 = __riscv_vwadd_wv_i32m2(v885, v886, 8);
        v69 = v887;
        vint32m2_t v888 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v889 = __riscv_vwmul_vx_i16m1(v825, v879, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v890 = __riscv_vwadd_wv_i32m2(v888, v889, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v891 = __riscv_vwmul_vx_i16m1(v828, v882, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v892 = __riscv_vwadd_wv_i32m2(v890, v891, 8);
        v71 = v892;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v893 = v38 + 192;
        const uint8_t* v894 = (const uint8_t*) v893;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v895 = __riscv_vle8_v_u8mf2(v894, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v896 = __riscv_vand_vx_u8mf2(v895, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v897 = __riscv_vzext_vf2_u16m1(v896, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v898 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v897, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v899 = __riscv_vsrl_vx_u8mf2(v895, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v900 = __riscv_vzext_vf2_u16m1(v899, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v901 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v900, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v902 = v38 + 200;
        const uint8_t* v903 = (const uint8_t*) v902;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v904 = __riscv_vle8_v_u8mf2(v903, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v905 = __riscv_vand_vx_u8mf2(v904, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v906 = __riscv_vzext_vf2_u16m1(v905, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v907 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v906, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v908 = __riscv_vsrl_vx_u8mf2(v904, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v909 = __riscv_vzext_vf2_u16m1(v908, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v910 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v909, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v911 = v40 + 48;
        const int8_t* v912 = (const int8_t*) v911;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v913 = *(const int8_t *)(v912);
        const uint8_t* v914 = v40 + 112;
        const int8_t* v915 = (const int8_t*) v914;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v916 = *(const int8_t *)(v915);
        vint32m2_t v917 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v918 = __riscv_vwmul_vx_i16m1(v898, v913, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v919 = __riscv_vwadd_wv_i32m2(v917, v918, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v920 = __riscv_vwmul_vx_i16m1(v901, v916, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v921 = __riscv_vwadd_wv_i32m2(v919, v920, 8);
        v57 = v921;
        vint32m2_t v922 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v923 = __riscv_vwmul_vx_i16m1(v907, v913, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v924 = __riscv_vwadd_wv_i32m2(v922, v923, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v925 = __riscv_vwmul_vx_i16m1(v910, v916, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v926 = __riscv_vwadd_wv_i32m2(v924, v925, 8);
        v59 = v926;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v927 = v40 + 49;
        const int8_t* v928 = (const int8_t*) v927;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v929 = *(const int8_t *)(v928);
        const uint8_t* v930 = v40 + 113;
        const int8_t* v931 = (const int8_t*) v930;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v932 = *(const int8_t *)(v931);
        vint32m2_t v933 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v934 = __riscv_vwmul_vx_i16m1(v898, v929, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v935 = __riscv_vwadd_wv_i32m2(v933, v934, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v936 = __riscv_vwmul_vx_i16m1(v901, v932, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v937 = __riscv_vwadd_wv_i32m2(v935, v936, 8);
        v61 = v937;
        vint32m2_t v938 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v939 = __riscv_vwmul_vx_i16m1(v907, v929, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v940 = __riscv_vwadd_wv_i32m2(v938, v939, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v941 = __riscv_vwmul_vx_i16m1(v910, v932, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v942 = __riscv_vwadd_wv_i32m2(v940, v941, 8);
        v63 = v942;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v943 = v40 + 50;
        const int8_t* v944 = (const int8_t*) v943;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v945 = *(const int8_t *)(v944);
        const uint8_t* v946 = v40 + 114;
        const int8_t* v947 = (const int8_t*) v946;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v948 = *(const int8_t *)(v947);
        vint32m2_t v949 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v950 = __riscv_vwmul_vx_i16m1(v898, v945, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v951 = __riscv_vwadd_wv_i32m2(v949, v950, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v952 = __riscv_vwmul_vx_i16m1(v901, v948, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v953 = __riscv_vwadd_wv_i32m2(v951, v952, 8);
        v65 = v953;
        vint32m2_t v954 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v955 = __riscv_vwmul_vx_i16m1(v907, v945, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v956 = __riscv_vwadd_wv_i32m2(v954, v955, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v957 = __riscv_vwmul_vx_i16m1(v910, v948, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v958 = __riscv_vwadd_wv_i32m2(v956, v957, 8);
        v67 = v958;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v959 = v40 + 51;
        const int8_t* v960 = (const int8_t*) v959;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v961 = *(const int8_t *)(v960);
        const uint8_t* v962 = v40 + 115;
        const int8_t* v963 = (const int8_t*) v962;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v964 = *(const int8_t *)(v963);
        vint32m2_t v965 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v966 = __riscv_vwmul_vx_i16m1(v898, v961, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v967 = __riscv_vwadd_wv_i32m2(v965, v966, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v968 = __riscv_vwmul_vx_i16m1(v901, v964, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v969 = __riscv_vwadd_wv_i32m2(v967, v968, 8);
        v69 = v969;
        vint32m2_t v970 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v971 = __riscv_vwmul_vx_i16m1(v907, v961, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v972 = __riscv_vwadd_wv_i32m2(v970, v971, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v973 = __riscv_vwmul_vx_i16m1(v910, v964, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v974 = __riscv_vwadd_wv_i32m2(v972, v973, 8);
        v71 = v974;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v975 = v38 + 208;
        const uint8_t* v976 = (const uint8_t*) v975;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v977 = __riscv_vle8_v_u8mf2(v976, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v978 = __riscv_vand_vx_u8mf2(v977, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v979 = __riscv_vzext_vf2_u16m1(v978, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v980 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v979, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v981 = __riscv_vsrl_vx_u8mf2(v977, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v982 = __riscv_vzext_vf2_u16m1(v981, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v983 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v982, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v984 = v38 + 216;
        const uint8_t* v985 = (const uint8_t*) v984;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v986 = __riscv_vle8_v_u8mf2(v985, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v987 = __riscv_vand_vx_u8mf2(v986, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v988 = __riscv_vzext_vf2_u16m1(v987, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v989 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v988, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v990 = __riscv_vsrl_vx_u8mf2(v986, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v991 = __riscv_vzext_vf2_u16m1(v990, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v992 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v991, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v993 = v40 + 52;
        const int8_t* v994 = (const int8_t*) v993;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v995 = *(const int8_t *)(v994);
        const uint8_t* v996 = v40 + 116;
        const int8_t* v997 = (const int8_t*) v996;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v998 = *(const int8_t *)(v997);
        vint32m2_t v999 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1000 = __riscv_vwmul_vx_i16m1(v980, v995, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1001 = __riscv_vwadd_wv_i32m2(v999, v1000, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1002 = __riscv_vwmul_vx_i16m1(v983, v998, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1003 = __riscv_vwadd_wv_i32m2(v1001, v1002, 8);
        v57 = v1003;
        vint32m2_t v1004 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1005 = __riscv_vwmul_vx_i16m1(v989, v995, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1006 = __riscv_vwadd_wv_i32m2(v1004, v1005, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1007 = __riscv_vwmul_vx_i16m1(v992, v998, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1008 = __riscv_vwadd_wv_i32m2(v1006, v1007, 8);
        v59 = v1008;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1009 = v40 + 53;
        const int8_t* v1010 = (const int8_t*) v1009;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1011 = *(const int8_t *)(v1010);
        const uint8_t* v1012 = v40 + 117;
        const int8_t* v1013 = (const int8_t*) v1012;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1014 = *(const int8_t *)(v1013);
        vint32m2_t v1015 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1016 = __riscv_vwmul_vx_i16m1(v980, v1011, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1017 = __riscv_vwadd_wv_i32m2(v1015, v1016, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1018 = __riscv_vwmul_vx_i16m1(v983, v1014, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1019 = __riscv_vwadd_wv_i32m2(v1017, v1018, 8);
        v61 = v1019;
        vint32m2_t v1020 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1021 = __riscv_vwmul_vx_i16m1(v989, v1011, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1022 = __riscv_vwadd_wv_i32m2(v1020, v1021, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1023 = __riscv_vwmul_vx_i16m1(v992, v1014, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1024 = __riscv_vwadd_wv_i32m2(v1022, v1023, 8);
        v63 = v1024;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1025 = v40 + 54;
        const int8_t* v1026 = (const int8_t*) v1025;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1027 = *(const int8_t *)(v1026);
        const uint8_t* v1028 = v40 + 118;
        const int8_t* v1029 = (const int8_t*) v1028;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1030 = *(const int8_t *)(v1029);
        vint32m2_t v1031 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1032 = __riscv_vwmul_vx_i16m1(v980, v1027, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1033 = __riscv_vwadd_wv_i32m2(v1031, v1032, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1034 = __riscv_vwmul_vx_i16m1(v983, v1030, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1035 = __riscv_vwadd_wv_i32m2(v1033, v1034, 8);
        v65 = v1035;
        vint32m2_t v1036 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1037 = __riscv_vwmul_vx_i16m1(v989, v1027, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1038 = __riscv_vwadd_wv_i32m2(v1036, v1037, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1039 = __riscv_vwmul_vx_i16m1(v992, v1030, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1040 = __riscv_vwadd_wv_i32m2(v1038, v1039, 8);
        v67 = v1040;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1041 = v40 + 55;
        const int8_t* v1042 = (const int8_t*) v1041;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1043 = *(const int8_t *)(v1042);
        const uint8_t* v1044 = v40 + 119;
        const int8_t* v1045 = (const int8_t*) v1044;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1046 = *(const int8_t *)(v1045);
        vint32m2_t v1047 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1048 = __riscv_vwmul_vx_i16m1(v980, v1043, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1049 = __riscv_vwadd_wv_i32m2(v1047, v1048, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1050 = __riscv_vwmul_vx_i16m1(v983, v1046, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1051 = __riscv_vwadd_wv_i32m2(v1049, v1050, 8);
        v69 = v1051;
        vint32m2_t v1052 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1053 = __riscv_vwmul_vx_i16m1(v989, v1043, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1054 = __riscv_vwadd_wv_i32m2(v1052, v1053, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1055 = __riscv_vwmul_vx_i16m1(v992, v1046, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1056 = __riscv_vwadd_wv_i32m2(v1054, v1055, 8);
        v71 = v1056;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1057 = v38 + 224;
        const uint8_t* v1058 = (const uint8_t*) v1057;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1059 = __riscv_vle8_v_u8mf2(v1058, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1060 = __riscv_vand_vx_u8mf2(v1059, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1061 = __riscv_vzext_vf2_u16m1(v1060, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1062 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1061, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1063 = __riscv_vsrl_vx_u8mf2(v1059, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1064 = __riscv_vzext_vf2_u16m1(v1063, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1065 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1064, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1066 = v38 + 232;
        const uint8_t* v1067 = (const uint8_t*) v1066;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1068 = __riscv_vle8_v_u8mf2(v1067, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1069 = __riscv_vand_vx_u8mf2(v1068, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1070 = __riscv_vzext_vf2_u16m1(v1069, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1071 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1070, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1072 = __riscv_vsrl_vx_u8mf2(v1068, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1073 = __riscv_vzext_vf2_u16m1(v1072, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1074 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1073, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1075 = v40 + 56;
        const int8_t* v1076 = (const int8_t*) v1075;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1077 = *(const int8_t *)(v1076);
        const uint8_t* v1078 = v40 + 120;
        const int8_t* v1079 = (const int8_t*) v1078;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1080 = *(const int8_t *)(v1079);
        vint32m2_t v1081 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1082 = __riscv_vwmul_vx_i16m1(v1062, v1077, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1083 = __riscv_vwadd_wv_i32m2(v1081, v1082, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1084 = __riscv_vwmul_vx_i16m1(v1065, v1080, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1085 = __riscv_vwadd_wv_i32m2(v1083, v1084, 8);
        v57 = v1085;
        vint32m2_t v1086 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1087 = __riscv_vwmul_vx_i16m1(v1071, v1077, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1088 = __riscv_vwadd_wv_i32m2(v1086, v1087, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1089 = __riscv_vwmul_vx_i16m1(v1074, v1080, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1090 = __riscv_vwadd_wv_i32m2(v1088, v1089, 8);
        v59 = v1090;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1091 = v40 + 57;
        const int8_t* v1092 = (const int8_t*) v1091;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1093 = *(const int8_t *)(v1092);
        const uint8_t* v1094 = v40 + 121;
        const int8_t* v1095 = (const int8_t*) v1094;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1096 = *(const int8_t *)(v1095);
        vint32m2_t v1097 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1098 = __riscv_vwmul_vx_i16m1(v1062, v1093, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1099 = __riscv_vwadd_wv_i32m2(v1097, v1098, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1100 = __riscv_vwmul_vx_i16m1(v1065, v1096, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1101 = __riscv_vwadd_wv_i32m2(v1099, v1100, 8);
        v61 = v1101;
        vint32m2_t v1102 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1103 = __riscv_vwmul_vx_i16m1(v1071, v1093, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1104 = __riscv_vwadd_wv_i32m2(v1102, v1103, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1105 = __riscv_vwmul_vx_i16m1(v1074, v1096, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1106 = __riscv_vwadd_wv_i32m2(v1104, v1105, 8);
        v63 = v1106;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1107 = v40 + 58;
        const int8_t* v1108 = (const int8_t*) v1107;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1109 = *(const int8_t *)(v1108);
        const uint8_t* v1110 = v40 + 122;
        const int8_t* v1111 = (const int8_t*) v1110;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1112 = *(const int8_t *)(v1111);
        vint32m2_t v1113 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1114 = __riscv_vwmul_vx_i16m1(v1062, v1109, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1115 = __riscv_vwadd_wv_i32m2(v1113, v1114, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1116 = __riscv_vwmul_vx_i16m1(v1065, v1112, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1117 = __riscv_vwadd_wv_i32m2(v1115, v1116, 8);
        v65 = v1117;
        vint32m2_t v1118 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1119 = __riscv_vwmul_vx_i16m1(v1071, v1109, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1120 = __riscv_vwadd_wv_i32m2(v1118, v1119, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1121 = __riscv_vwmul_vx_i16m1(v1074, v1112, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1122 = __riscv_vwadd_wv_i32m2(v1120, v1121, 8);
        v67 = v1122;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1123 = v40 + 59;
        const int8_t* v1124 = (const int8_t*) v1123;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1125 = *(const int8_t *)(v1124);
        const uint8_t* v1126 = v40 + 123;
        const int8_t* v1127 = (const int8_t*) v1126;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1128 = *(const int8_t *)(v1127);
        vint32m2_t v1129 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1130 = __riscv_vwmul_vx_i16m1(v1062, v1125, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1131 = __riscv_vwadd_wv_i32m2(v1129, v1130, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1132 = __riscv_vwmul_vx_i16m1(v1065, v1128, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1133 = __riscv_vwadd_wv_i32m2(v1131, v1132, 8);
        v69 = v1133;
        vint32m2_t v1134 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1135 = __riscv_vwmul_vx_i16m1(v1071, v1125, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1136 = __riscv_vwadd_wv_i32m2(v1134, v1135, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1137 = __riscv_vwmul_vx_i16m1(v1074, v1128, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1138 = __riscv_vwadd_wv_i32m2(v1136, v1137, 8);
        v71 = v1138;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1139 = v38 + 240;
        const uint8_t* v1140 = (const uint8_t*) v1139;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1141 = __riscv_vle8_v_u8mf2(v1140, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1142 = __riscv_vand_vx_u8mf2(v1141, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1143 = __riscv_vzext_vf2_u16m1(v1142, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1144 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1143, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1145 = __riscv_vsrl_vx_u8mf2(v1141, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1146 = __riscv_vzext_vf2_u16m1(v1145, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1147 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1146, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1148 = v38 + 248;
        const uint8_t* v1149 = (const uint8_t*) v1148;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1150 = __riscv_vle8_v_u8mf2(v1149, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1151 = __riscv_vand_vx_u8mf2(v1150, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1152 = __riscv_vzext_vf2_u16m1(v1151, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1153 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1152, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1154 = __riscv_vsrl_vx_u8mf2(v1150, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1155 = __riscv_vzext_vf2_u16m1(v1154, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1156 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1155, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1157 = v40 + 60;
        const int8_t* v1158 = (const int8_t*) v1157;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1159 = *(const int8_t *)(v1158);
        const uint8_t* v1160 = v40 + 124;
        const int8_t* v1161 = (const int8_t*) v1160;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1162 = *(const int8_t *)(v1161);
        vint32m2_t v1163 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1164 = __riscv_vwmul_vx_i16m1(v1144, v1159, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1165 = __riscv_vwadd_wv_i32m2(v1163, v1164, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1166 = __riscv_vwmul_vx_i16m1(v1147, v1162, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1167 = __riscv_vwadd_wv_i32m2(v1165, v1166, 8);
        v57 = v1167;
        vint32m2_t v1168 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1169 = __riscv_vwmul_vx_i16m1(v1153, v1159, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1170 = __riscv_vwadd_wv_i32m2(v1168, v1169, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1171 = __riscv_vwmul_vx_i16m1(v1156, v1162, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1172 = __riscv_vwadd_wv_i32m2(v1170, v1171, 8);
        v59 = v1172;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1173 = v40 + 61;
        const int8_t* v1174 = (const int8_t*) v1173;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1175 = *(const int8_t *)(v1174);
        const uint8_t* v1176 = v40 + 125;
        const int8_t* v1177 = (const int8_t*) v1176;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1178 = *(const int8_t *)(v1177);
        vint32m2_t v1179 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1180 = __riscv_vwmul_vx_i16m1(v1144, v1175, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1181 = __riscv_vwadd_wv_i32m2(v1179, v1180, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1182 = __riscv_vwmul_vx_i16m1(v1147, v1178, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1183 = __riscv_vwadd_wv_i32m2(v1181, v1182, 8);
        v61 = v1183;
        vint32m2_t v1184 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1185 = __riscv_vwmul_vx_i16m1(v1153, v1175, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1186 = __riscv_vwadd_wv_i32m2(v1184, v1185, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1187 = __riscv_vwmul_vx_i16m1(v1156, v1178, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1188 = __riscv_vwadd_wv_i32m2(v1186, v1187, 8);
        v63 = v1188;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1189 = v40 + 62;
        const int8_t* v1190 = (const int8_t*) v1189;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1191 = *(const int8_t *)(v1190);
        const uint8_t* v1192 = v40 + 126;
        const int8_t* v1193 = (const int8_t*) v1192;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1194 = *(const int8_t *)(v1193);
        vint32m2_t v1195 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1196 = __riscv_vwmul_vx_i16m1(v1144, v1191, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1197 = __riscv_vwadd_wv_i32m2(v1195, v1196, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1198 = __riscv_vwmul_vx_i16m1(v1147, v1194, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1199 = __riscv_vwadd_wv_i32m2(v1197, v1198, 8);
        v65 = v1199;
        vint32m2_t v1200 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1201 = __riscv_vwmul_vx_i16m1(v1153, v1191, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1202 = __riscv_vwadd_wv_i32m2(v1200, v1201, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1203 = __riscv_vwmul_vx_i16m1(v1156, v1194, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1204 = __riscv_vwadd_wv_i32m2(v1202, v1203, 8);
        v67 = v1204;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1205 = v40 + 63;
        const int8_t* v1206 = (const int8_t*) v1205;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1207 = *(const int8_t *)(v1206);
        const uint8_t* v1208 = v40 + 127;
        const int8_t* v1209 = (const int8_t*) v1208;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1210 = *(const int8_t *)(v1209);
        vint32m2_t v1211 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1212 = __riscv_vwmul_vx_i16m1(v1144, v1207, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1213 = __riscv_vwadd_wv_i32m2(v1211, v1212, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1214 = __riscv_vwmul_vx_i16m1(v1147, v1210, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1215 = __riscv_vwadd_wv_i32m2(v1213, v1214, 8);
        v69 = v1215;
        vint32m2_t v1216 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1217 = __riscv_vwmul_vx_i16m1(v1153, v1207, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1218 = __riscv_vwadd_wv_i32m2(v1216, v1217, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1219 = __riscv_vwmul_vx_i16m1(v1156, v1210, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1220 = __riscv_vwadd_wv_i32m2(v1218, v1219, 8);
        v71 = v1220;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1221 = v38 + 256;
        const uint8_t* v1222 = (const uint8_t*) v1221;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1223 = __riscv_vle8_v_u8mf2(v1222, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1224 = __riscv_vand_vx_u8mf2(v1223, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1225 = __riscv_vzext_vf2_u16m1(v1224, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1226 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1225, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1227 = __riscv_vsrl_vx_u8mf2(v1223, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1228 = __riscv_vzext_vf2_u16m1(v1227, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1229 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1228, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1230 = v38 + 264;
        const uint8_t* v1231 = (const uint8_t*) v1230;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1232 = __riscv_vle8_v_u8mf2(v1231, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1233 = __riscv_vand_vx_u8mf2(v1232, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1234 = __riscv_vzext_vf2_u16m1(v1233, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1235 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1234, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1236 = __riscv_vsrl_vx_u8mf2(v1232, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1237 = __riscv_vzext_vf2_u16m1(v1236, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1238 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1237, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1239 = v40 + 64;
        const int8_t* v1240 = (const int8_t*) v1239;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1241 = *(const int8_t *)(v1240);
        const uint8_t* v1242 = v40 + 128;
        const int8_t* v1243 = (const int8_t*) v1242;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1244 = *(const int8_t *)(v1243);
        vint32m2_t v1245 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1246 = __riscv_vwmul_vx_i16m1(v1226, v1241, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1247 = __riscv_vwadd_wv_i32m2(v1245, v1246, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1248 = __riscv_vwmul_vx_i16m1(v1229, v1244, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1249 = __riscv_vwadd_wv_i32m2(v1247, v1248, 8);
        v57 = v1249;
        vint32m2_t v1250 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1251 = __riscv_vwmul_vx_i16m1(v1235, v1241, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1252 = __riscv_vwadd_wv_i32m2(v1250, v1251, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1253 = __riscv_vwmul_vx_i16m1(v1238, v1244, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1254 = __riscv_vwadd_wv_i32m2(v1252, v1253, 8);
        v59 = v1254;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1255 = v40 + 65;
        const int8_t* v1256 = (const int8_t*) v1255;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1257 = *(const int8_t *)(v1256);
        const uint8_t* v1258 = v40 + 129;
        const int8_t* v1259 = (const int8_t*) v1258;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1260 = *(const int8_t *)(v1259);
        vint32m2_t v1261 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1262 = __riscv_vwmul_vx_i16m1(v1226, v1257, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1263 = __riscv_vwadd_wv_i32m2(v1261, v1262, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1264 = __riscv_vwmul_vx_i16m1(v1229, v1260, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1265 = __riscv_vwadd_wv_i32m2(v1263, v1264, 8);
        v61 = v1265;
        vint32m2_t v1266 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1267 = __riscv_vwmul_vx_i16m1(v1235, v1257, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1268 = __riscv_vwadd_wv_i32m2(v1266, v1267, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1269 = __riscv_vwmul_vx_i16m1(v1238, v1260, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1270 = __riscv_vwadd_wv_i32m2(v1268, v1269, 8);
        v63 = v1270;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1271 = v40 + 66;
        const int8_t* v1272 = (const int8_t*) v1271;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1273 = *(const int8_t *)(v1272);
        const uint8_t* v1274 = v40 + 130;
        const int8_t* v1275 = (const int8_t*) v1274;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1276 = *(const int8_t *)(v1275);
        vint32m2_t v1277 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1278 = __riscv_vwmul_vx_i16m1(v1226, v1273, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1279 = __riscv_vwadd_wv_i32m2(v1277, v1278, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1280 = __riscv_vwmul_vx_i16m1(v1229, v1276, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1281 = __riscv_vwadd_wv_i32m2(v1279, v1280, 8);
        v65 = v1281;
        vint32m2_t v1282 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1283 = __riscv_vwmul_vx_i16m1(v1235, v1273, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1284 = __riscv_vwadd_wv_i32m2(v1282, v1283, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1285 = __riscv_vwmul_vx_i16m1(v1238, v1276, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1286 = __riscv_vwadd_wv_i32m2(v1284, v1285, 8);
        v67 = v1286;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1287 = v40 + 67;
        const int8_t* v1288 = (const int8_t*) v1287;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1289 = *(const int8_t *)(v1288);
        const uint8_t* v1290 = v40 + 131;
        const int8_t* v1291 = (const int8_t*) v1290;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1292 = *(const int8_t *)(v1291);
        vint32m2_t v1293 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1294 = __riscv_vwmul_vx_i16m1(v1226, v1289, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1295 = __riscv_vwadd_wv_i32m2(v1293, v1294, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1296 = __riscv_vwmul_vx_i16m1(v1229, v1292, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1297 = __riscv_vwadd_wv_i32m2(v1295, v1296, 8);
        v69 = v1297;
        vint32m2_t v1298 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1299 = __riscv_vwmul_vx_i16m1(v1235, v1289, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1300 = __riscv_vwadd_wv_i32m2(v1298, v1299, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1301 = __riscv_vwmul_vx_i16m1(v1238, v1292, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1302 = __riscv_vwadd_wv_i32m2(v1300, v1301, 8);
        v71 = v1302;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1303 = v38 + 272;
        const uint8_t* v1304 = (const uint8_t*) v1303;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1305 = __riscv_vle8_v_u8mf2(v1304, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1306 = __riscv_vand_vx_u8mf2(v1305, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1307 = __riscv_vzext_vf2_u16m1(v1306, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1308 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1307, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1309 = __riscv_vsrl_vx_u8mf2(v1305, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1310 = __riscv_vzext_vf2_u16m1(v1309, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1311 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1310, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1312 = v38 + 280;
        const uint8_t* v1313 = (const uint8_t*) v1312;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1314 = __riscv_vle8_v_u8mf2(v1313, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1315 = __riscv_vand_vx_u8mf2(v1314, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1316 = __riscv_vzext_vf2_u16m1(v1315, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1317 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1316, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1318 = __riscv_vsrl_vx_u8mf2(v1314, 0x04, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v1319 = __riscv_vzext_vf2_u16m1(v1318, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v1320 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v1319, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1321 = v40 + 68;
        const int8_t* v1322 = (const int8_t*) v1321;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1323 = *(const int8_t *)(v1322);
        const uint8_t* v1324 = v40 + 132;
        const int8_t* v1325 = (const int8_t*) v1324;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1326 = *(const int8_t *)(v1325);
        vint32m2_t v1327 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1328 = __riscv_vwmul_vx_i16m1(v1308, v1323, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1329 = __riscv_vwadd_wv_i32m2(v1327, v1328, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1330 = __riscv_vwmul_vx_i16m1(v1311, v1326, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1331 = __riscv_vwadd_wv_i32m2(v1329, v1330, 8);
        v57 = v1331;
        vint32m2_t v1332 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1333 = __riscv_vwmul_vx_i16m1(v1317, v1323, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1334 = __riscv_vwadd_wv_i32m2(v1332, v1333, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1335 = __riscv_vwmul_vx_i16m1(v1320, v1326, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1336 = __riscv_vwadd_wv_i32m2(v1334, v1335, 8);
        v59 = v1336;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1337 = v40 + 69;
        const int8_t* v1338 = (const int8_t*) v1337;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1339 = *(const int8_t *)(v1338);
        const uint8_t* v1340 = v40 + 133;
        const int8_t* v1341 = (const int8_t*) v1340;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1342 = *(const int8_t *)(v1341);
        vint32m2_t v1343 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1344 = __riscv_vwmul_vx_i16m1(v1308, v1339, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1345 = __riscv_vwadd_wv_i32m2(v1343, v1344, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1346 = __riscv_vwmul_vx_i16m1(v1311, v1342, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1347 = __riscv_vwadd_wv_i32m2(v1345, v1346, 8);
        v61 = v1347;
        vint32m2_t v1348 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1349 = __riscv_vwmul_vx_i16m1(v1317, v1339, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1350 = __riscv_vwadd_wv_i32m2(v1348, v1349, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1351 = __riscv_vwmul_vx_i16m1(v1320, v1342, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1352 = __riscv_vwadd_wv_i32m2(v1350, v1351, 8);
        v63 = v1352;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1353 = v40 + 70;
        const int8_t* v1354 = (const int8_t*) v1353;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1355 = *(const int8_t *)(v1354);
        const uint8_t* v1356 = v40 + 134;
        const int8_t* v1357 = (const int8_t*) v1356;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1358 = *(const int8_t *)(v1357);
        vint32m2_t v1359 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1360 = __riscv_vwmul_vx_i16m1(v1308, v1355, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1361 = __riscv_vwadd_wv_i32m2(v1359, v1360, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1362 = __riscv_vwmul_vx_i16m1(v1311, v1358, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1363 = __riscv_vwadd_wv_i32m2(v1361, v1362, 8);
        v65 = v1363;
        vint32m2_t v1364 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1365 = __riscv_vwmul_vx_i16m1(v1317, v1355, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1366 = __riscv_vwadd_wv_i32m2(v1364, v1365, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1367 = __riscv_vwmul_vx_i16m1(v1320, v1358, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1368 = __riscv_vwadd_wv_i32m2(v1366, v1367, 8);
        v67 = v1368;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1369 = v40 + 71;
        const int8_t* v1370 = (const int8_t*) v1369;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1371 = *(const int8_t *)(v1370);
        const uint8_t* v1372 = v40 + 135;
        const int8_t* v1373 = (const int8_t*) v1372;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1374 = *(const int8_t *)(v1373);
        vint32m2_t v1375 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1376 = __riscv_vwmul_vx_i16m1(v1308, v1371, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1377 = __riscv_vwadd_wv_i32m2(v1375, v1376, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1378 = __riscv_vwmul_vx_i16m1(v1311, v1374, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1379 = __riscv_vwadd_wv_i32m2(v1377, v1378, 8);
        v69 = v1379;
        vint32m2_t v1380 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1381 = __riscv_vwmul_vx_i16m1(v1317, v1371, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1382 = __riscv_vwadd_wv_i32m2(v1380, v1381, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v1383 = __riscv_vwmul_vx_i16m1(v1320, v1374, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v1384 = __riscv_vwadd_wv_i32m2(v1382, v1383, 8);
        v71 = v1384;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
        vfloat32m2_t v1385 = __riscv_vfwmul_vf_f32m2(v53, v42, 8);
        vint32m2_t v1386 = v57;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1387 = __riscv_vfcvt_f_x_v_f32m2(v1386, 8);
        vfloat32m2_t v1388 = v20;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1389 = __riscv_vfmacc_vv_f32m2(v1388, v1387, v1385, 8);
        v20 = v1389;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
        vfloat32m2_t v1390 = __riscv_vfwmul_vf_f32m2(v56, v42, 8);
        vint32m2_t v1391 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1392 = __riscv_vfcvt_f_x_v_f32m2(v1391, 8);
        vfloat32m2_t v1393 = v22;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1394 = __riscv_vfmacc_vv_f32m2(v1393, v1392, v1390, 8);
        v22 = v1394;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
        vfloat32m2_t v1395 = __riscv_vfwmul_vf_f32m2(v53, v45, 8);
        vint32m2_t v1396 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1397 = __riscv_vfcvt_f_x_v_f32m2(v1396, 8);
        vfloat32m2_t v1398 = v24;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1399 = __riscv_vfmacc_vv_f32m2(v1398, v1397, v1395, 8);
        v24 = v1399;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
        vfloat32m2_t v1400 = __riscv_vfwmul_vf_f32m2(v56, v45, 8);
        vint32m2_t v1401 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1402 = __riscv_vfcvt_f_x_v_f32m2(v1401, 8);
        vfloat32m2_t v1403 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1404 = __riscv_vfmacc_vv_f32m2(v1403, v1402, v1400, 8);
        v26 = v1404;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
        vfloat32m2_t v1405 = __riscv_vfwmul_vf_f32m2(v53, v48, 8);
        vint32m2_t v1406 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1407 = __riscv_vfcvt_f_x_v_f32m2(v1406, 8);
        vfloat32m2_t v1408 = v28;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1409 = __riscv_vfmacc_vv_f32m2(v1408, v1407, v1405, 8);
        v28 = v1409;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
        vfloat32m2_t v1410 = __riscv_vfwmul_vf_f32m2(v56, v48, 8);
        vint32m2_t v1411 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1412 = __riscv_vfcvt_f_x_v_f32m2(v1411, 8);
        vfloat32m2_t v1413 = v30;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1414 = __riscv_vfmacc_vv_f32m2(v1413, v1412, v1410, 8);
        v30 = v1414;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
        vfloat32m2_t v1415 = __riscv_vfwmul_vf_f32m2(v53, v51, 8);
        vint32m2_t v1416 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1417 = __riscv_vfcvt_f_x_v_f32m2(v1416, 8);
        vfloat32m2_t v1418 = v32;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1419 = __riscv_vfmacc_vv_f32m2(v1418, v1417, v1415, 8);
        v32 = v1419;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
        vfloat32m2_t v1420 = __riscv_vfwmul_vf_f32m2(v56, v51, 8);
        vint32m2_t v1421 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1422 = __riscv_vfcvt_f_x_v_f32m2(v1421, 8);
        vfloat32m2_t v1423 = v34;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v1424 = __riscv_vfmacc_vv_f32m2(v1423, v1422, v1420, 8);
        v34 = v1424;
      }
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1425 = v12 * 4;
      size_t v1426 = v1425 + 0;
      size_t v1427 = v1426 * v7;
      size_t v1428 = v16 * 16;
      size_t v1429 = v1427 + v1428;
      float* v1430 = v2 + v1429;
      vfloat32m2_t v1431 = v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1430, v1431, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1432 = v12 * 4;
      size_t v1433 = v1432 + 0;
      size_t v1434 = v1433 * v7;
      size_t v1435 = v16 * 16;
      size_t v1436 = v1434 + v1435;
      size_t v1437 = v1436 + 8;
      float* v1438 = v2 + v1437;
      vfloat32m2_t v1439 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1438, v1439, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1440 = v12 * 4;
      size_t v1441 = v1440 + 1;
      size_t v1442 = v1441 * v7;
      size_t v1443 = v16 * 16;
      size_t v1444 = v1442 + v1443;
      float* v1445 = v2 + v1444;
      vfloat32m2_t v1446 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1445, v1446, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1447 = v12 * 4;
      size_t v1448 = v1447 + 1;
      size_t v1449 = v1448 * v7;
      size_t v1450 = v16 * 16;
      size_t v1451 = v1449 + v1450;
      size_t v1452 = v1451 + 8;
      float* v1453 = v2 + v1452;
      vfloat32m2_t v1454 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1453, v1454, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1455 = v12 * 4;
      size_t v1456 = v1455 + 2;
      size_t v1457 = v1456 * v7;
      size_t v1458 = v16 * 16;
      size_t v1459 = v1457 + v1458;
      float* v1460 = v2 + v1459;
      vfloat32m2_t v1461 = v28;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1460, v1461, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1462 = v12 * 4;
      size_t v1463 = v1462 + 2;
      size_t v1464 = v1463 * v7;
      size_t v1465 = v16 * 16;
      size_t v1466 = v1464 + v1465;
      size_t v1467 = v1466 + 8;
      float* v1468 = v2 + v1467;
      vfloat32m2_t v1469 = v30;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1468, v1469, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1470 = v12 * 4;
      size_t v1471 = v1470 + 3;
      size_t v1472 = v1471 * v7;
      size_t v1473 = v16 * 16;
      size_t v1474 = v1472 + v1473;
      float* v1475 = v2 + v1474;
      vfloat32m2_t v1476 = v32;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1475, v1476, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1477 = v12 * 4;
      size_t v1478 = v1477 + 3;
      size_t v1479 = v1478 * v7;
      size_t v1480 = v16 * 16;
      size_t v1481 = v1479 + v1480;
      size_t v1482 = v1481 + 8;
      float* v1483 = v2 + v1482;
      vfloat32m2_t v1484 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1483, v1484, 8);
    }
  }
  return;
}


