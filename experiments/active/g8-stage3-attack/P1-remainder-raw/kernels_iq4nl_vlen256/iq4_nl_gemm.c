#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0(size_t v1, size_t v2, size_t v3, float* v4, const uint8_t* v5, const uint8_t* v6, size_t v7) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v8 = __riscv_vsetvl_e32m1(v3);
  // weft_emitc.loop_order_override selector=col_outer/prior realized=row_outer gate=measured-gate-blocks-unmeasured
  // weft_emitc.route_source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_iq4_nl_repack_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v9 = v3 / 32;
  // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=row_group_count
  size_t v10 = v1 / 4;
  // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=col_group_count
  size_t v11 = v7 / 16;
  for (size_t v12 = 0; v12 < v10; v12 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_group_base
    size_t v13 = v12 * v9;
    size_t v14 = v13 * 136;
    const uint8_t* v15 = v6 + v14;
    for (size_t v16 = 0; v16 < v11; v16 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_group_base
      size_t v17 = v16 * v9;
      size_t v18 = v17 * 288;
      const uint8_t* v19 = v5 + v18;
      vfloat32m2_t v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v21 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      v20 = v21;
      vfloat32m2_t v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v23 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      v22 = v23;
      vfloat32m2_t v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v25 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      v24 = v25;
      vfloat32m2_t v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v27 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      v26 = v27;
      for (size_t v28 = 0; v28 < v9; v28 += 1) {
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
        size_t v29 = v28 * 288;
        const uint8_t* v30 = v19 + v29;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
        size_t v31 = v28 * 136;
        const uint8_t* v32 = v15 + v31;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const _Float16* v33 = (const _Float16*) v32;
        _Float16 v34 = *(const _Float16 *)(v33);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v35 = v32 + 2;
        const _Float16* v36 = (const _Float16*) v35;
        _Float16 v37 = *(const _Float16 *)(v36);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v38 = v32 + 4;
        const _Float16* v39 = (const _Float16*) v38;
        _Float16 v40 = *(const _Float16 *)(v39);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v41 = v32 + 6;
        const _Float16* v42 = (const _Float16*) v41;
        _Float16 v43 = *(const _Float16 *)(v42);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const _Float16* v44 = (const _Float16*) v30;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v45 = __riscv_vle16_v_f16m1(v44, 16);
        vint32m2_t v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v47 = __riscv_vmv_v_x_i32m2(0, 16);
        v46 = v47;
        vint32m2_t v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v49 = __riscv_vmv_v_x_i32m2(0, 16);
        v48 = v49;
        vint32m2_t v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v51 = __riscv_vmv_v_x_i32m2(0, 16);
        v50 = v51;
        vint32m2_t v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v53 = __riscv_vmv_v_x_i32m2(0, 16);
        v52 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v54 = v30 + 32;
        const uint8_t* v55 = (const uint8_t*) v54;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v56 = __riscv_vle8_v_u8mf2(v55, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v57 = __riscv_vand_vx_u8mf2(v56, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v58 = __riscv_vzext_vf2_u16m1(v57, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v59 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v58, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v60 = __riscv_vsrl_vx_u8mf2(v56, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v61 = __riscv_vzext_vf2_u16m1(v60, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v62 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v61, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v63 = v32 + 8;
        const int8_t* v64 = (const int8_t*) v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v65 = *(const int8_t *)(v64);
        const uint8_t* v66 = v32 + 72;
        const int8_t* v67 = (const int8_t*) v66;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v68 = *(const int8_t *)(v67);
        vint32m2_t v69 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v70 = __riscv_vwmul_vx_i16m1(v59, v65, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v71 = __riscv_vwadd_wv_i32m2(v69, v70, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v72 = __riscv_vwmul_vx_i16m1(v62, v68, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v73 = __riscv_vwadd_wv_i32m2(v71, v72, 16);
        v46 = v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v74 = v32 + 9;
        const int8_t* v75 = (const int8_t*) v74;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v76 = *(const int8_t *)(v75);
        const uint8_t* v77 = v32 + 73;
        const int8_t* v78 = (const int8_t*) v77;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v79 = *(const int8_t *)(v78);
        vint32m2_t v80 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v81 = __riscv_vwmul_vx_i16m1(v59, v76, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v82 = __riscv_vwadd_wv_i32m2(v80, v81, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v83 = __riscv_vwmul_vx_i16m1(v62, v79, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v84 = __riscv_vwadd_wv_i32m2(v82, v83, 16);
        v48 = v84;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v85 = v32 + 10;
        const int8_t* v86 = (const int8_t*) v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v87 = *(const int8_t *)(v86);
        const uint8_t* v88 = v32 + 74;
        const int8_t* v89 = (const int8_t*) v88;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v90 = *(const int8_t *)(v89);
        vint32m2_t v91 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v92 = __riscv_vwmul_vx_i16m1(v59, v87, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v93 = __riscv_vwadd_wv_i32m2(v91, v92, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v94 = __riscv_vwmul_vx_i16m1(v62, v90, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v95 = __riscv_vwadd_wv_i32m2(v93, v94, 16);
        v50 = v95;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v96 = v32 + 11;
        const int8_t* v97 = (const int8_t*) v96;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v98 = *(const int8_t *)(v97);
        const uint8_t* v99 = v32 + 75;
        const int8_t* v100 = (const int8_t*) v99;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v101 = *(const int8_t *)(v100);
        vint32m2_t v102 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v103 = __riscv_vwmul_vx_i16m1(v59, v98, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v104 = __riscv_vwadd_wv_i32m2(v102, v103, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v105 = __riscv_vwmul_vx_i16m1(v62, v101, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v106 = __riscv_vwadd_wv_i32m2(v104, v105, 16);
        v52 = v106;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v107 = v30 + 48;
        const uint8_t* v108 = (const uint8_t*) v107;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v109 = __riscv_vle8_v_u8mf2(v108, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v110 = __riscv_vand_vx_u8mf2(v109, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v111 = __riscv_vzext_vf2_u16m1(v110, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v112 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v111, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v113 = __riscv_vsrl_vx_u8mf2(v109, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v114 = __riscv_vzext_vf2_u16m1(v113, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v115 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v114, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v116 = v32 + 12;
        const int8_t* v117 = (const int8_t*) v116;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v118 = *(const int8_t *)(v117);
        const uint8_t* v119 = v32 + 76;
        const int8_t* v120 = (const int8_t*) v119;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v121 = *(const int8_t *)(v120);
        vint32m2_t v122 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v123 = __riscv_vwmul_vx_i16m1(v112, v118, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v124 = __riscv_vwadd_wv_i32m2(v122, v123, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v125 = __riscv_vwmul_vx_i16m1(v115, v121, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v126 = __riscv_vwadd_wv_i32m2(v124, v125, 16);
        v46 = v126;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v127 = v32 + 13;
        const int8_t* v128 = (const int8_t*) v127;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v129 = *(const int8_t *)(v128);
        const uint8_t* v130 = v32 + 77;
        const int8_t* v131 = (const int8_t*) v130;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v132 = *(const int8_t *)(v131);
        vint32m2_t v133 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v134 = __riscv_vwmul_vx_i16m1(v112, v129, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v135 = __riscv_vwadd_wv_i32m2(v133, v134, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v136 = __riscv_vwmul_vx_i16m1(v115, v132, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v137 = __riscv_vwadd_wv_i32m2(v135, v136, 16);
        v48 = v137;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v138 = v32 + 14;
        const int8_t* v139 = (const int8_t*) v138;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v140 = *(const int8_t *)(v139);
        const uint8_t* v141 = v32 + 78;
        const int8_t* v142 = (const int8_t*) v141;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v143 = *(const int8_t *)(v142);
        vint32m2_t v144 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v145 = __riscv_vwmul_vx_i16m1(v112, v140, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v146 = __riscv_vwadd_wv_i32m2(v144, v145, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v147 = __riscv_vwmul_vx_i16m1(v115, v143, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v148 = __riscv_vwadd_wv_i32m2(v146, v147, 16);
        v50 = v148;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v149 = v32 + 15;
        const int8_t* v150 = (const int8_t*) v149;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v151 = *(const int8_t *)(v150);
        const uint8_t* v152 = v32 + 79;
        const int8_t* v153 = (const int8_t*) v152;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v154 = *(const int8_t *)(v153);
        vint32m2_t v155 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v156 = __riscv_vwmul_vx_i16m1(v112, v151, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v157 = __riscv_vwadd_wv_i32m2(v155, v156, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v158 = __riscv_vwmul_vx_i16m1(v115, v154, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v159 = __riscv_vwadd_wv_i32m2(v157, v158, 16);
        v52 = v159;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v160 = v30 + 64;
        const uint8_t* v161 = (const uint8_t*) v160;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v162 = __riscv_vle8_v_u8mf2(v161, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v163 = __riscv_vand_vx_u8mf2(v162, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v164 = __riscv_vzext_vf2_u16m1(v163, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v165 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v164, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v166 = __riscv_vsrl_vx_u8mf2(v162, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v167 = __riscv_vzext_vf2_u16m1(v166, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v168 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v167, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v169 = v32 + 16;
        const int8_t* v170 = (const int8_t*) v169;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v171 = *(const int8_t *)(v170);
        const uint8_t* v172 = v32 + 80;
        const int8_t* v173 = (const int8_t*) v172;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v174 = *(const int8_t *)(v173);
        vint32m2_t v175 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v176 = __riscv_vwmul_vx_i16m1(v165, v171, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v177 = __riscv_vwadd_wv_i32m2(v175, v176, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v178 = __riscv_vwmul_vx_i16m1(v168, v174, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v179 = __riscv_vwadd_wv_i32m2(v177, v178, 16);
        v46 = v179;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v180 = v32 + 17;
        const int8_t* v181 = (const int8_t*) v180;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v182 = *(const int8_t *)(v181);
        const uint8_t* v183 = v32 + 81;
        const int8_t* v184 = (const int8_t*) v183;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v185 = *(const int8_t *)(v184);
        vint32m2_t v186 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v187 = __riscv_vwmul_vx_i16m1(v165, v182, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v188 = __riscv_vwadd_wv_i32m2(v186, v187, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v189 = __riscv_vwmul_vx_i16m1(v168, v185, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v190 = __riscv_vwadd_wv_i32m2(v188, v189, 16);
        v48 = v190;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v191 = v32 + 18;
        const int8_t* v192 = (const int8_t*) v191;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v193 = *(const int8_t *)(v192);
        const uint8_t* v194 = v32 + 82;
        const int8_t* v195 = (const int8_t*) v194;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v196 = *(const int8_t *)(v195);
        vint32m2_t v197 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v198 = __riscv_vwmul_vx_i16m1(v165, v193, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v199 = __riscv_vwadd_wv_i32m2(v197, v198, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v200 = __riscv_vwmul_vx_i16m1(v168, v196, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v201 = __riscv_vwadd_wv_i32m2(v199, v200, 16);
        v50 = v201;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v202 = v32 + 19;
        const int8_t* v203 = (const int8_t*) v202;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v204 = *(const int8_t *)(v203);
        const uint8_t* v205 = v32 + 83;
        const int8_t* v206 = (const int8_t*) v205;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v207 = *(const int8_t *)(v206);
        vint32m2_t v208 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v209 = __riscv_vwmul_vx_i16m1(v165, v204, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v210 = __riscv_vwadd_wv_i32m2(v208, v209, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v211 = __riscv_vwmul_vx_i16m1(v168, v207, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v212 = __riscv_vwadd_wv_i32m2(v210, v211, 16);
        v52 = v212;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v213 = v30 + 80;
        const uint8_t* v214 = (const uint8_t*) v213;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v215 = __riscv_vle8_v_u8mf2(v214, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v216 = __riscv_vand_vx_u8mf2(v215, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v217 = __riscv_vzext_vf2_u16m1(v216, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v218 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v217, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v219 = __riscv_vsrl_vx_u8mf2(v215, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v220 = __riscv_vzext_vf2_u16m1(v219, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v221 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v220, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v222 = v32 + 20;
        const int8_t* v223 = (const int8_t*) v222;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v224 = *(const int8_t *)(v223);
        const uint8_t* v225 = v32 + 84;
        const int8_t* v226 = (const int8_t*) v225;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v227 = *(const int8_t *)(v226);
        vint32m2_t v228 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v229 = __riscv_vwmul_vx_i16m1(v218, v224, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v230 = __riscv_vwadd_wv_i32m2(v228, v229, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v231 = __riscv_vwmul_vx_i16m1(v221, v227, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v232 = __riscv_vwadd_wv_i32m2(v230, v231, 16);
        v46 = v232;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v233 = v32 + 21;
        const int8_t* v234 = (const int8_t*) v233;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v235 = *(const int8_t *)(v234);
        const uint8_t* v236 = v32 + 85;
        const int8_t* v237 = (const int8_t*) v236;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v238 = *(const int8_t *)(v237);
        vint32m2_t v239 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v240 = __riscv_vwmul_vx_i16m1(v218, v235, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v241 = __riscv_vwadd_wv_i32m2(v239, v240, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v242 = __riscv_vwmul_vx_i16m1(v221, v238, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v243 = __riscv_vwadd_wv_i32m2(v241, v242, 16);
        v48 = v243;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v244 = v32 + 22;
        const int8_t* v245 = (const int8_t*) v244;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v246 = *(const int8_t *)(v245);
        const uint8_t* v247 = v32 + 86;
        const int8_t* v248 = (const int8_t*) v247;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v249 = *(const int8_t *)(v248);
        vint32m2_t v250 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v251 = __riscv_vwmul_vx_i16m1(v218, v246, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v252 = __riscv_vwadd_wv_i32m2(v250, v251, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v253 = __riscv_vwmul_vx_i16m1(v221, v249, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v254 = __riscv_vwadd_wv_i32m2(v252, v253, 16);
        v50 = v254;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v255 = v32 + 23;
        const int8_t* v256 = (const int8_t*) v255;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v257 = *(const int8_t *)(v256);
        const uint8_t* v258 = v32 + 87;
        const int8_t* v259 = (const int8_t*) v258;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v260 = *(const int8_t *)(v259);
        vint32m2_t v261 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v262 = __riscv_vwmul_vx_i16m1(v218, v257, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v263 = __riscv_vwadd_wv_i32m2(v261, v262, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v264 = __riscv_vwmul_vx_i16m1(v221, v260, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v265 = __riscv_vwadd_wv_i32m2(v263, v264, 16);
        v52 = v265;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v266 = v30 + 96;
        const uint8_t* v267 = (const uint8_t*) v266;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v268 = __riscv_vle8_v_u8mf2(v267, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v269 = __riscv_vand_vx_u8mf2(v268, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v270 = __riscv_vzext_vf2_u16m1(v269, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v271 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v270, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v272 = __riscv_vsrl_vx_u8mf2(v268, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v273 = __riscv_vzext_vf2_u16m1(v272, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v274 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v273, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v275 = v32 + 24;
        const int8_t* v276 = (const int8_t*) v275;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v277 = *(const int8_t *)(v276);
        const uint8_t* v278 = v32 + 88;
        const int8_t* v279 = (const int8_t*) v278;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v280 = *(const int8_t *)(v279);
        vint32m2_t v281 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v282 = __riscv_vwmul_vx_i16m1(v271, v277, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v283 = __riscv_vwadd_wv_i32m2(v281, v282, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v284 = __riscv_vwmul_vx_i16m1(v274, v280, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v285 = __riscv_vwadd_wv_i32m2(v283, v284, 16);
        v46 = v285;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v286 = v32 + 25;
        const int8_t* v287 = (const int8_t*) v286;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v288 = *(const int8_t *)(v287);
        const uint8_t* v289 = v32 + 89;
        const int8_t* v290 = (const int8_t*) v289;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v291 = *(const int8_t *)(v290);
        vint32m2_t v292 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v293 = __riscv_vwmul_vx_i16m1(v271, v288, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v294 = __riscv_vwadd_wv_i32m2(v292, v293, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v295 = __riscv_vwmul_vx_i16m1(v274, v291, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v296 = __riscv_vwadd_wv_i32m2(v294, v295, 16);
        v48 = v296;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v297 = v32 + 26;
        const int8_t* v298 = (const int8_t*) v297;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v299 = *(const int8_t *)(v298);
        const uint8_t* v300 = v32 + 90;
        const int8_t* v301 = (const int8_t*) v300;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v302 = *(const int8_t *)(v301);
        vint32m2_t v303 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v304 = __riscv_vwmul_vx_i16m1(v271, v299, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v305 = __riscv_vwadd_wv_i32m2(v303, v304, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v306 = __riscv_vwmul_vx_i16m1(v274, v302, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v307 = __riscv_vwadd_wv_i32m2(v305, v306, 16);
        v50 = v307;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v308 = v32 + 27;
        const int8_t* v309 = (const int8_t*) v308;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v310 = *(const int8_t *)(v309);
        const uint8_t* v311 = v32 + 91;
        const int8_t* v312 = (const int8_t*) v311;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v313 = *(const int8_t *)(v312);
        vint32m2_t v314 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v315 = __riscv_vwmul_vx_i16m1(v271, v310, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v316 = __riscv_vwadd_wv_i32m2(v314, v315, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v317 = __riscv_vwmul_vx_i16m1(v274, v313, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v318 = __riscv_vwadd_wv_i32m2(v316, v317, 16);
        v52 = v318;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v319 = v30 + 112;
        const uint8_t* v320 = (const uint8_t*) v319;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v321 = __riscv_vle8_v_u8mf2(v320, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v322 = __riscv_vand_vx_u8mf2(v321, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v323 = __riscv_vzext_vf2_u16m1(v322, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v324 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v323, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v325 = __riscv_vsrl_vx_u8mf2(v321, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v326 = __riscv_vzext_vf2_u16m1(v325, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v327 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v326, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v328 = v32 + 28;
        const int8_t* v329 = (const int8_t*) v328;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v330 = *(const int8_t *)(v329);
        const uint8_t* v331 = v32 + 92;
        const int8_t* v332 = (const int8_t*) v331;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v333 = *(const int8_t *)(v332);
        vint32m2_t v334 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v335 = __riscv_vwmul_vx_i16m1(v324, v330, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v336 = __riscv_vwadd_wv_i32m2(v334, v335, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v337 = __riscv_vwmul_vx_i16m1(v327, v333, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v338 = __riscv_vwadd_wv_i32m2(v336, v337, 16);
        v46 = v338;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v339 = v32 + 29;
        const int8_t* v340 = (const int8_t*) v339;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v341 = *(const int8_t *)(v340);
        const uint8_t* v342 = v32 + 93;
        const int8_t* v343 = (const int8_t*) v342;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v344 = *(const int8_t *)(v343);
        vint32m2_t v345 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v346 = __riscv_vwmul_vx_i16m1(v324, v341, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v347 = __riscv_vwadd_wv_i32m2(v345, v346, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v348 = __riscv_vwmul_vx_i16m1(v327, v344, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v349 = __riscv_vwadd_wv_i32m2(v347, v348, 16);
        v48 = v349;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v350 = v32 + 30;
        const int8_t* v351 = (const int8_t*) v350;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v352 = *(const int8_t *)(v351);
        const uint8_t* v353 = v32 + 94;
        const int8_t* v354 = (const int8_t*) v353;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v355 = *(const int8_t *)(v354);
        vint32m2_t v356 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v357 = __riscv_vwmul_vx_i16m1(v324, v352, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v358 = __riscv_vwadd_wv_i32m2(v356, v357, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v359 = __riscv_vwmul_vx_i16m1(v327, v355, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v360 = __riscv_vwadd_wv_i32m2(v358, v359, 16);
        v50 = v360;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v361 = v32 + 31;
        const int8_t* v362 = (const int8_t*) v361;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v363 = *(const int8_t *)(v362);
        const uint8_t* v364 = v32 + 95;
        const int8_t* v365 = (const int8_t*) v364;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v366 = *(const int8_t *)(v365);
        vint32m2_t v367 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v368 = __riscv_vwmul_vx_i16m1(v324, v363, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v369 = __riscv_vwadd_wv_i32m2(v367, v368, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v370 = __riscv_vwmul_vx_i16m1(v327, v366, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v371 = __riscv_vwadd_wv_i32m2(v369, v370, 16);
        v52 = v371;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v372 = v30 + 128;
        const uint8_t* v373 = (const uint8_t*) v372;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v374 = __riscv_vle8_v_u8mf2(v373, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v375 = __riscv_vand_vx_u8mf2(v374, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v376 = __riscv_vzext_vf2_u16m1(v375, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v377 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v376, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v378 = __riscv_vsrl_vx_u8mf2(v374, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v379 = __riscv_vzext_vf2_u16m1(v378, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v380 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v379, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v381 = v32 + 32;
        const int8_t* v382 = (const int8_t*) v381;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v383 = *(const int8_t *)(v382);
        const uint8_t* v384 = v32 + 96;
        const int8_t* v385 = (const int8_t*) v384;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v386 = *(const int8_t *)(v385);
        vint32m2_t v387 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v388 = __riscv_vwmul_vx_i16m1(v377, v383, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v389 = __riscv_vwadd_wv_i32m2(v387, v388, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v390 = __riscv_vwmul_vx_i16m1(v380, v386, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v391 = __riscv_vwadd_wv_i32m2(v389, v390, 16);
        v46 = v391;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v392 = v32 + 33;
        const int8_t* v393 = (const int8_t*) v392;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v394 = *(const int8_t *)(v393);
        const uint8_t* v395 = v32 + 97;
        const int8_t* v396 = (const int8_t*) v395;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v397 = *(const int8_t *)(v396);
        vint32m2_t v398 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v399 = __riscv_vwmul_vx_i16m1(v377, v394, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v400 = __riscv_vwadd_wv_i32m2(v398, v399, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v401 = __riscv_vwmul_vx_i16m1(v380, v397, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v402 = __riscv_vwadd_wv_i32m2(v400, v401, 16);
        v48 = v402;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v403 = v32 + 34;
        const int8_t* v404 = (const int8_t*) v403;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v405 = *(const int8_t *)(v404);
        const uint8_t* v406 = v32 + 98;
        const int8_t* v407 = (const int8_t*) v406;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v408 = *(const int8_t *)(v407);
        vint32m2_t v409 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v410 = __riscv_vwmul_vx_i16m1(v377, v405, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v411 = __riscv_vwadd_wv_i32m2(v409, v410, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v412 = __riscv_vwmul_vx_i16m1(v380, v408, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v413 = __riscv_vwadd_wv_i32m2(v411, v412, 16);
        v50 = v413;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v414 = v32 + 35;
        const int8_t* v415 = (const int8_t*) v414;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v416 = *(const int8_t *)(v415);
        const uint8_t* v417 = v32 + 99;
        const int8_t* v418 = (const int8_t*) v417;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v419 = *(const int8_t *)(v418);
        vint32m2_t v420 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v421 = __riscv_vwmul_vx_i16m1(v377, v416, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v422 = __riscv_vwadd_wv_i32m2(v420, v421, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v423 = __riscv_vwmul_vx_i16m1(v380, v419, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v424 = __riscv_vwadd_wv_i32m2(v422, v423, 16);
        v52 = v424;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v425 = v30 + 144;
        const uint8_t* v426 = (const uint8_t*) v425;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v427 = __riscv_vle8_v_u8mf2(v426, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v428 = __riscv_vand_vx_u8mf2(v427, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v429 = __riscv_vzext_vf2_u16m1(v428, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v430 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v429, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v431 = __riscv_vsrl_vx_u8mf2(v427, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v432 = __riscv_vzext_vf2_u16m1(v431, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v433 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v432, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v434 = v32 + 36;
        const int8_t* v435 = (const int8_t*) v434;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v436 = *(const int8_t *)(v435);
        const uint8_t* v437 = v32 + 100;
        const int8_t* v438 = (const int8_t*) v437;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v439 = *(const int8_t *)(v438);
        vint32m2_t v440 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v441 = __riscv_vwmul_vx_i16m1(v430, v436, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v442 = __riscv_vwadd_wv_i32m2(v440, v441, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v443 = __riscv_vwmul_vx_i16m1(v433, v439, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v444 = __riscv_vwadd_wv_i32m2(v442, v443, 16);
        v46 = v444;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v445 = v32 + 37;
        const int8_t* v446 = (const int8_t*) v445;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v447 = *(const int8_t *)(v446);
        const uint8_t* v448 = v32 + 101;
        const int8_t* v449 = (const int8_t*) v448;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v450 = *(const int8_t *)(v449);
        vint32m2_t v451 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v452 = __riscv_vwmul_vx_i16m1(v430, v447, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v453 = __riscv_vwadd_wv_i32m2(v451, v452, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v454 = __riscv_vwmul_vx_i16m1(v433, v450, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v455 = __riscv_vwadd_wv_i32m2(v453, v454, 16);
        v48 = v455;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v456 = v32 + 38;
        const int8_t* v457 = (const int8_t*) v456;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v458 = *(const int8_t *)(v457);
        const uint8_t* v459 = v32 + 102;
        const int8_t* v460 = (const int8_t*) v459;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v461 = *(const int8_t *)(v460);
        vint32m2_t v462 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v463 = __riscv_vwmul_vx_i16m1(v430, v458, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v464 = __riscv_vwadd_wv_i32m2(v462, v463, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v465 = __riscv_vwmul_vx_i16m1(v433, v461, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v466 = __riscv_vwadd_wv_i32m2(v464, v465, 16);
        v50 = v466;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v467 = v32 + 39;
        const int8_t* v468 = (const int8_t*) v467;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v469 = *(const int8_t *)(v468);
        const uint8_t* v470 = v32 + 103;
        const int8_t* v471 = (const int8_t*) v470;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v472 = *(const int8_t *)(v471);
        vint32m2_t v473 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v474 = __riscv_vwmul_vx_i16m1(v430, v469, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v475 = __riscv_vwadd_wv_i32m2(v473, v474, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v476 = __riscv_vwmul_vx_i16m1(v433, v472, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v477 = __riscv_vwadd_wv_i32m2(v475, v476, 16);
        v52 = v477;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v478 = v30 + 160;
        const uint8_t* v479 = (const uint8_t*) v478;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v480 = __riscv_vle8_v_u8mf2(v479, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v481 = __riscv_vand_vx_u8mf2(v480, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v482 = __riscv_vzext_vf2_u16m1(v481, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v483 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v482, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v484 = __riscv_vsrl_vx_u8mf2(v480, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v485 = __riscv_vzext_vf2_u16m1(v484, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v486 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v485, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v487 = v32 + 40;
        const int8_t* v488 = (const int8_t*) v487;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v489 = *(const int8_t *)(v488);
        const uint8_t* v490 = v32 + 104;
        const int8_t* v491 = (const int8_t*) v490;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v492 = *(const int8_t *)(v491);
        vint32m2_t v493 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v494 = __riscv_vwmul_vx_i16m1(v483, v489, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v495 = __riscv_vwadd_wv_i32m2(v493, v494, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v496 = __riscv_vwmul_vx_i16m1(v486, v492, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v497 = __riscv_vwadd_wv_i32m2(v495, v496, 16);
        v46 = v497;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v498 = v32 + 41;
        const int8_t* v499 = (const int8_t*) v498;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v500 = *(const int8_t *)(v499);
        const uint8_t* v501 = v32 + 105;
        const int8_t* v502 = (const int8_t*) v501;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v503 = *(const int8_t *)(v502);
        vint32m2_t v504 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v505 = __riscv_vwmul_vx_i16m1(v483, v500, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v506 = __riscv_vwadd_wv_i32m2(v504, v505, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v507 = __riscv_vwmul_vx_i16m1(v486, v503, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v508 = __riscv_vwadd_wv_i32m2(v506, v507, 16);
        v48 = v508;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v509 = v32 + 42;
        const int8_t* v510 = (const int8_t*) v509;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v511 = *(const int8_t *)(v510);
        const uint8_t* v512 = v32 + 106;
        const int8_t* v513 = (const int8_t*) v512;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v514 = *(const int8_t *)(v513);
        vint32m2_t v515 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v516 = __riscv_vwmul_vx_i16m1(v483, v511, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v517 = __riscv_vwadd_wv_i32m2(v515, v516, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v518 = __riscv_vwmul_vx_i16m1(v486, v514, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v519 = __riscv_vwadd_wv_i32m2(v517, v518, 16);
        v50 = v519;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v520 = v32 + 43;
        const int8_t* v521 = (const int8_t*) v520;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v522 = *(const int8_t *)(v521);
        const uint8_t* v523 = v32 + 107;
        const int8_t* v524 = (const int8_t*) v523;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v525 = *(const int8_t *)(v524);
        vint32m2_t v526 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v527 = __riscv_vwmul_vx_i16m1(v483, v522, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v528 = __riscv_vwadd_wv_i32m2(v526, v527, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v529 = __riscv_vwmul_vx_i16m1(v486, v525, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v530 = __riscv_vwadd_wv_i32m2(v528, v529, 16);
        v52 = v530;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v531 = v30 + 176;
        const uint8_t* v532 = (const uint8_t*) v531;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v533 = __riscv_vle8_v_u8mf2(v532, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v534 = __riscv_vand_vx_u8mf2(v533, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v535 = __riscv_vzext_vf2_u16m1(v534, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v536 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v535, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v537 = __riscv_vsrl_vx_u8mf2(v533, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v538 = __riscv_vzext_vf2_u16m1(v537, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v539 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v538, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v540 = v32 + 44;
        const int8_t* v541 = (const int8_t*) v540;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v542 = *(const int8_t *)(v541);
        const uint8_t* v543 = v32 + 108;
        const int8_t* v544 = (const int8_t*) v543;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v545 = *(const int8_t *)(v544);
        vint32m2_t v546 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v547 = __riscv_vwmul_vx_i16m1(v536, v542, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v548 = __riscv_vwadd_wv_i32m2(v546, v547, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v549 = __riscv_vwmul_vx_i16m1(v539, v545, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v550 = __riscv_vwadd_wv_i32m2(v548, v549, 16);
        v46 = v550;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v551 = v32 + 45;
        const int8_t* v552 = (const int8_t*) v551;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v553 = *(const int8_t *)(v552);
        const uint8_t* v554 = v32 + 109;
        const int8_t* v555 = (const int8_t*) v554;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v556 = *(const int8_t *)(v555);
        vint32m2_t v557 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v558 = __riscv_vwmul_vx_i16m1(v536, v553, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v559 = __riscv_vwadd_wv_i32m2(v557, v558, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v560 = __riscv_vwmul_vx_i16m1(v539, v556, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v561 = __riscv_vwadd_wv_i32m2(v559, v560, 16);
        v48 = v561;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v562 = v32 + 46;
        const int8_t* v563 = (const int8_t*) v562;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v564 = *(const int8_t *)(v563);
        const uint8_t* v565 = v32 + 110;
        const int8_t* v566 = (const int8_t*) v565;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v567 = *(const int8_t *)(v566);
        vint32m2_t v568 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v569 = __riscv_vwmul_vx_i16m1(v536, v564, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v570 = __riscv_vwadd_wv_i32m2(v568, v569, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v571 = __riscv_vwmul_vx_i16m1(v539, v567, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v572 = __riscv_vwadd_wv_i32m2(v570, v571, 16);
        v50 = v572;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v573 = v32 + 47;
        const int8_t* v574 = (const int8_t*) v573;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v575 = *(const int8_t *)(v574);
        const uint8_t* v576 = v32 + 111;
        const int8_t* v577 = (const int8_t*) v576;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v578 = *(const int8_t *)(v577);
        vint32m2_t v579 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v580 = __riscv_vwmul_vx_i16m1(v536, v575, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v581 = __riscv_vwadd_wv_i32m2(v579, v580, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v582 = __riscv_vwmul_vx_i16m1(v539, v578, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v583 = __riscv_vwadd_wv_i32m2(v581, v582, 16);
        v52 = v583;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v584 = v30 + 192;
        const uint8_t* v585 = (const uint8_t*) v584;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v586 = __riscv_vle8_v_u8mf2(v585, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v587 = __riscv_vand_vx_u8mf2(v586, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v588 = __riscv_vzext_vf2_u16m1(v587, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v589 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v588, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v590 = __riscv_vsrl_vx_u8mf2(v586, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v591 = __riscv_vzext_vf2_u16m1(v590, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v592 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v591, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v593 = v32 + 48;
        const int8_t* v594 = (const int8_t*) v593;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v595 = *(const int8_t *)(v594);
        const uint8_t* v596 = v32 + 112;
        const int8_t* v597 = (const int8_t*) v596;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v598 = *(const int8_t *)(v597);
        vint32m2_t v599 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v600 = __riscv_vwmul_vx_i16m1(v589, v595, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v601 = __riscv_vwadd_wv_i32m2(v599, v600, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v602 = __riscv_vwmul_vx_i16m1(v592, v598, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v603 = __riscv_vwadd_wv_i32m2(v601, v602, 16);
        v46 = v603;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v604 = v32 + 49;
        const int8_t* v605 = (const int8_t*) v604;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v606 = *(const int8_t *)(v605);
        const uint8_t* v607 = v32 + 113;
        const int8_t* v608 = (const int8_t*) v607;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v609 = *(const int8_t *)(v608);
        vint32m2_t v610 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v611 = __riscv_vwmul_vx_i16m1(v589, v606, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v612 = __riscv_vwadd_wv_i32m2(v610, v611, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v613 = __riscv_vwmul_vx_i16m1(v592, v609, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v614 = __riscv_vwadd_wv_i32m2(v612, v613, 16);
        v48 = v614;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v615 = v32 + 50;
        const int8_t* v616 = (const int8_t*) v615;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v617 = *(const int8_t *)(v616);
        const uint8_t* v618 = v32 + 114;
        const int8_t* v619 = (const int8_t*) v618;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v620 = *(const int8_t *)(v619);
        vint32m2_t v621 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v622 = __riscv_vwmul_vx_i16m1(v589, v617, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v623 = __riscv_vwadd_wv_i32m2(v621, v622, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v624 = __riscv_vwmul_vx_i16m1(v592, v620, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v625 = __riscv_vwadd_wv_i32m2(v623, v624, 16);
        v50 = v625;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v626 = v32 + 51;
        const int8_t* v627 = (const int8_t*) v626;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v628 = *(const int8_t *)(v627);
        const uint8_t* v629 = v32 + 115;
        const int8_t* v630 = (const int8_t*) v629;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v631 = *(const int8_t *)(v630);
        vint32m2_t v632 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v633 = __riscv_vwmul_vx_i16m1(v589, v628, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v634 = __riscv_vwadd_wv_i32m2(v632, v633, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v635 = __riscv_vwmul_vx_i16m1(v592, v631, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v636 = __riscv_vwadd_wv_i32m2(v634, v635, 16);
        v52 = v636;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v637 = v30 + 208;
        const uint8_t* v638 = (const uint8_t*) v637;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v639 = __riscv_vle8_v_u8mf2(v638, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v640 = __riscv_vand_vx_u8mf2(v639, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v641 = __riscv_vzext_vf2_u16m1(v640, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v642 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v641, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v643 = __riscv_vsrl_vx_u8mf2(v639, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v644 = __riscv_vzext_vf2_u16m1(v643, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v645 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v644, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v646 = v32 + 52;
        const int8_t* v647 = (const int8_t*) v646;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v648 = *(const int8_t *)(v647);
        const uint8_t* v649 = v32 + 116;
        const int8_t* v650 = (const int8_t*) v649;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v651 = *(const int8_t *)(v650);
        vint32m2_t v652 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v653 = __riscv_vwmul_vx_i16m1(v642, v648, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v654 = __riscv_vwadd_wv_i32m2(v652, v653, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v655 = __riscv_vwmul_vx_i16m1(v645, v651, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v656 = __riscv_vwadd_wv_i32m2(v654, v655, 16);
        v46 = v656;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v657 = v32 + 53;
        const int8_t* v658 = (const int8_t*) v657;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v659 = *(const int8_t *)(v658);
        const uint8_t* v660 = v32 + 117;
        const int8_t* v661 = (const int8_t*) v660;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v662 = *(const int8_t *)(v661);
        vint32m2_t v663 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v664 = __riscv_vwmul_vx_i16m1(v642, v659, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v665 = __riscv_vwadd_wv_i32m2(v663, v664, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v666 = __riscv_vwmul_vx_i16m1(v645, v662, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v667 = __riscv_vwadd_wv_i32m2(v665, v666, 16);
        v48 = v667;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v668 = v32 + 54;
        const int8_t* v669 = (const int8_t*) v668;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v670 = *(const int8_t *)(v669);
        const uint8_t* v671 = v32 + 118;
        const int8_t* v672 = (const int8_t*) v671;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v673 = *(const int8_t *)(v672);
        vint32m2_t v674 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v675 = __riscv_vwmul_vx_i16m1(v642, v670, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v676 = __riscv_vwadd_wv_i32m2(v674, v675, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v677 = __riscv_vwmul_vx_i16m1(v645, v673, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v678 = __riscv_vwadd_wv_i32m2(v676, v677, 16);
        v50 = v678;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v679 = v32 + 55;
        const int8_t* v680 = (const int8_t*) v679;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v681 = *(const int8_t *)(v680);
        const uint8_t* v682 = v32 + 119;
        const int8_t* v683 = (const int8_t*) v682;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v684 = *(const int8_t *)(v683);
        vint32m2_t v685 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v686 = __riscv_vwmul_vx_i16m1(v642, v681, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v687 = __riscv_vwadd_wv_i32m2(v685, v686, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v688 = __riscv_vwmul_vx_i16m1(v645, v684, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v689 = __riscv_vwadd_wv_i32m2(v687, v688, 16);
        v52 = v689;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v690 = v30 + 224;
        const uint8_t* v691 = (const uint8_t*) v690;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v692 = __riscv_vle8_v_u8mf2(v691, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v693 = __riscv_vand_vx_u8mf2(v692, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v694 = __riscv_vzext_vf2_u16m1(v693, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v695 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v694, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v696 = __riscv_vsrl_vx_u8mf2(v692, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v697 = __riscv_vzext_vf2_u16m1(v696, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v698 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v697, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v699 = v32 + 56;
        const int8_t* v700 = (const int8_t*) v699;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v701 = *(const int8_t *)(v700);
        const uint8_t* v702 = v32 + 120;
        const int8_t* v703 = (const int8_t*) v702;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v704 = *(const int8_t *)(v703);
        vint32m2_t v705 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v706 = __riscv_vwmul_vx_i16m1(v695, v701, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v707 = __riscv_vwadd_wv_i32m2(v705, v706, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v708 = __riscv_vwmul_vx_i16m1(v698, v704, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v709 = __riscv_vwadd_wv_i32m2(v707, v708, 16);
        v46 = v709;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v710 = v32 + 57;
        const int8_t* v711 = (const int8_t*) v710;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v712 = *(const int8_t *)(v711);
        const uint8_t* v713 = v32 + 121;
        const int8_t* v714 = (const int8_t*) v713;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v715 = *(const int8_t *)(v714);
        vint32m2_t v716 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v717 = __riscv_vwmul_vx_i16m1(v695, v712, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v718 = __riscv_vwadd_wv_i32m2(v716, v717, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v719 = __riscv_vwmul_vx_i16m1(v698, v715, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v720 = __riscv_vwadd_wv_i32m2(v718, v719, 16);
        v48 = v720;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v721 = v32 + 58;
        const int8_t* v722 = (const int8_t*) v721;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v723 = *(const int8_t *)(v722);
        const uint8_t* v724 = v32 + 122;
        const int8_t* v725 = (const int8_t*) v724;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v726 = *(const int8_t *)(v725);
        vint32m2_t v727 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v728 = __riscv_vwmul_vx_i16m1(v695, v723, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v729 = __riscv_vwadd_wv_i32m2(v727, v728, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v730 = __riscv_vwmul_vx_i16m1(v698, v726, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v731 = __riscv_vwadd_wv_i32m2(v729, v730, 16);
        v50 = v731;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v732 = v32 + 59;
        const int8_t* v733 = (const int8_t*) v732;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v734 = *(const int8_t *)(v733);
        const uint8_t* v735 = v32 + 123;
        const int8_t* v736 = (const int8_t*) v735;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v737 = *(const int8_t *)(v736);
        vint32m2_t v738 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v739 = __riscv_vwmul_vx_i16m1(v695, v734, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v740 = __riscv_vwadd_wv_i32m2(v738, v739, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v741 = __riscv_vwmul_vx_i16m1(v698, v737, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v742 = __riscv_vwadd_wv_i32m2(v740, v741, 16);
        v52 = v742;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v743 = v30 + 240;
        const uint8_t* v744 = (const uint8_t*) v743;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v745 = __riscv_vle8_v_u8mf2(v744, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v746 = __riscv_vand_vx_u8mf2(v745, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v747 = __riscv_vzext_vf2_u16m1(v746, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v748 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v747, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v749 = __riscv_vsrl_vx_u8mf2(v745, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v750 = __riscv_vzext_vf2_u16m1(v749, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v751 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v750, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v752 = v32 + 60;
        const int8_t* v753 = (const int8_t*) v752;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v754 = *(const int8_t *)(v753);
        const uint8_t* v755 = v32 + 124;
        const int8_t* v756 = (const int8_t*) v755;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v757 = *(const int8_t *)(v756);
        vint32m2_t v758 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v759 = __riscv_vwmul_vx_i16m1(v748, v754, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v760 = __riscv_vwadd_wv_i32m2(v758, v759, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v761 = __riscv_vwmul_vx_i16m1(v751, v757, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v762 = __riscv_vwadd_wv_i32m2(v760, v761, 16);
        v46 = v762;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v763 = v32 + 61;
        const int8_t* v764 = (const int8_t*) v763;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v765 = *(const int8_t *)(v764);
        const uint8_t* v766 = v32 + 125;
        const int8_t* v767 = (const int8_t*) v766;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v768 = *(const int8_t *)(v767);
        vint32m2_t v769 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v770 = __riscv_vwmul_vx_i16m1(v748, v765, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v771 = __riscv_vwadd_wv_i32m2(v769, v770, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v772 = __riscv_vwmul_vx_i16m1(v751, v768, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v773 = __riscv_vwadd_wv_i32m2(v771, v772, 16);
        v48 = v773;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v774 = v32 + 62;
        const int8_t* v775 = (const int8_t*) v774;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v776 = *(const int8_t *)(v775);
        const uint8_t* v777 = v32 + 126;
        const int8_t* v778 = (const int8_t*) v777;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v779 = *(const int8_t *)(v778);
        vint32m2_t v780 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v781 = __riscv_vwmul_vx_i16m1(v748, v776, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v782 = __riscv_vwadd_wv_i32m2(v780, v781, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v783 = __riscv_vwmul_vx_i16m1(v751, v779, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v784 = __riscv_vwadd_wv_i32m2(v782, v783, 16);
        v50 = v784;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v785 = v32 + 63;
        const int8_t* v786 = (const int8_t*) v785;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v787 = *(const int8_t *)(v786);
        const uint8_t* v788 = v32 + 127;
        const int8_t* v789 = (const int8_t*) v788;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v790 = *(const int8_t *)(v789);
        vint32m2_t v791 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v792 = __riscv_vwmul_vx_i16m1(v748, v787, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v793 = __riscv_vwadd_wv_i32m2(v791, v792, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v794 = __riscv_vwmul_vx_i16m1(v751, v790, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v795 = __riscv_vwadd_wv_i32m2(v793, v794, 16);
        v52 = v795;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v796 = v30 + 256;
        const uint8_t* v797 = (const uint8_t*) v796;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v798 = __riscv_vle8_v_u8mf2(v797, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v799 = __riscv_vand_vx_u8mf2(v798, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v800 = __riscv_vzext_vf2_u16m1(v799, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v801 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v800, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v802 = __riscv_vsrl_vx_u8mf2(v798, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v803 = __riscv_vzext_vf2_u16m1(v802, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v804 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v803, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v805 = v32 + 64;
        const int8_t* v806 = (const int8_t*) v805;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v807 = *(const int8_t *)(v806);
        const uint8_t* v808 = v32 + 128;
        const int8_t* v809 = (const int8_t*) v808;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v810 = *(const int8_t *)(v809);
        vint32m2_t v811 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v812 = __riscv_vwmul_vx_i16m1(v801, v807, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v813 = __riscv_vwadd_wv_i32m2(v811, v812, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v814 = __riscv_vwmul_vx_i16m1(v804, v810, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v815 = __riscv_vwadd_wv_i32m2(v813, v814, 16);
        v46 = v815;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v816 = v32 + 65;
        const int8_t* v817 = (const int8_t*) v816;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v818 = *(const int8_t *)(v817);
        const uint8_t* v819 = v32 + 129;
        const int8_t* v820 = (const int8_t*) v819;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v821 = *(const int8_t *)(v820);
        vint32m2_t v822 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v823 = __riscv_vwmul_vx_i16m1(v801, v818, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v824 = __riscv_vwadd_wv_i32m2(v822, v823, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v825 = __riscv_vwmul_vx_i16m1(v804, v821, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v826 = __riscv_vwadd_wv_i32m2(v824, v825, 16);
        v48 = v826;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v827 = v32 + 66;
        const int8_t* v828 = (const int8_t*) v827;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v829 = *(const int8_t *)(v828);
        const uint8_t* v830 = v32 + 130;
        const int8_t* v831 = (const int8_t*) v830;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v832 = *(const int8_t *)(v831);
        vint32m2_t v833 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v834 = __riscv_vwmul_vx_i16m1(v801, v829, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v835 = __riscv_vwadd_wv_i32m2(v833, v834, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v836 = __riscv_vwmul_vx_i16m1(v804, v832, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v837 = __riscv_vwadd_wv_i32m2(v835, v836, 16);
        v50 = v837;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v838 = v32 + 67;
        const int8_t* v839 = (const int8_t*) v838;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v840 = *(const int8_t *)(v839);
        const uint8_t* v841 = v32 + 131;
        const int8_t* v842 = (const int8_t*) v841;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v843 = *(const int8_t *)(v842);
        vint32m2_t v844 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v845 = __riscv_vwmul_vx_i16m1(v801, v840, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v846 = __riscv_vwadd_wv_i32m2(v844, v845, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v847 = __riscv_vwmul_vx_i16m1(v804, v843, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v848 = __riscv_vwadd_wv_i32m2(v846, v847, 16);
        v52 = v848;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v849 = v30 + 272;
        const uint8_t* v850 = (const uint8_t*) v849;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v851 = __riscv_vle8_v_u8mf2(v850, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v852 = __riscv_vand_vx_u8mf2(v851, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v853 = __riscv_vzext_vf2_u16m1(v852, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v854 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v853, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v855 = __riscv_vsrl_vx_u8mf2(v851, 0x04, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
        vuint16m1_t v856 = __riscv_vzext_vf2_u16m1(v855, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
        vint8mf2_t v857 = __riscv_vluxei16_v_i8mf2(weft_iq4_nl_repack_kvalues, v856, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v858 = v32 + 68;
        const int8_t* v859 = (const int8_t*) v858;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v860 = *(const int8_t *)(v859);
        const uint8_t* v861 = v32 + 132;
        const int8_t* v862 = (const int8_t*) v861;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v863 = *(const int8_t *)(v862);
        vint32m2_t v864 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v865 = __riscv_vwmul_vx_i16m1(v854, v860, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v866 = __riscv_vwadd_wv_i32m2(v864, v865, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v867 = __riscv_vwmul_vx_i16m1(v857, v863, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v868 = __riscv_vwadd_wv_i32m2(v866, v867, 16);
        v46 = v868;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v869 = v32 + 69;
        const int8_t* v870 = (const int8_t*) v869;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v871 = *(const int8_t *)(v870);
        const uint8_t* v872 = v32 + 133;
        const int8_t* v873 = (const int8_t*) v872;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v874 = *(const int8_t *)(v873);
        vint32m2_t v875 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v876 = __riscv_vwmul_vx_i16m1(v854, v871, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v877 = __riscv_vwadd_wv_i32m2(v875, v876, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v878 = __riscv_vwmul_vx_i16m1(v857, v874, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v879 = __riscv_vwadd_wv_i32m2(v877, v878, 16);
        v48 = v879;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v880 = v32 + 70;
        const int8_t* v881 = (const int8_t*) v880;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v882 = *(const int8_t *)(v881);
        const uint8_t* v883 = v32 + 134;
        const int8_t* v884 = (const int8_t*) v883;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v885 = *(const int8_t *)(v884);
        vint32m2_t v886 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v887 = __riscv_vwmul_vx_i16m1(v854, v882, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v888 = __riscv_vwadd_wv_i32m2(v886, v887, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v889 = __riscv_vwmul_vx_i16m1(v857, v885, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v890 = __riscv_vwadd_wv_i32m2(v888, v889, 16);
        v50 = v890;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v891 = v32 + 71;
        const int8_t* v892 = (const int8_t*) v891;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v893 = *(const int8_t *)(v892);
        const uint8_t* v894 = v32 + 135;
        const int8_t* v895 = (const int8_t*) v894;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v896 = *(const int8_t *)(v895);
        vint32m2_t v897 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v898 = __riscv_vwmul_vx_i16m1(v854, v893, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v899 = __riscv_vwadd_wv_i32m2(v897, v898, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v900 = __riscv_vwmul_vx_i16m1(v857, v896, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v901 = __riscv_vwadd_wv_i32m2(v899, v900, 16);
        v52 = v901;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
        vfloat32m2_t v902 = __riscv_vfwmul_vf_f32m2(v45, v34, 16);
        vint32m2_t v903 = v46;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v904 = __riscv_vfcvt_f_x_v_f32m2(v903, 16);
        vfloat32m2_t v905 = v20;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v906 = __riscv_vfmacc_vv_f32m2(v905, v904, v902, 16);
        v20 = v906;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
        vfloat32m2_t v907 = __riscv_vfwmul_vf_f32m2(v45, v37, 16);
        vint32m2_t v908 = v48;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v909 = __riscv_vfcvt_f_x_v_f32m2(v908, 16);
        vfloat32m2_t v910 = v22;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v911 = __riscv_vfmacc_vv_f32m2(v910, v909, v907, 16);
        v22 = v911;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
        vfloat32m2_t v912 = __riscv_vfwmul_vf_f32m2(v45, v40, 16);
        vint32m2_t v913 = v50;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v914 = __riscv_vfcvt_f_x_v_f32m2(v913, 16);
        vfloat32m2_t v915 = v24;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v916 = __riscv_vfmacc_vv_f32m2(v915, v914, v912, 16);
        v24 = v916;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
        vfloat32m2_t v917 = __riscv_vfwmul_vf_f32m2(v45, v43, 16);
        vint32m2_t v918 = v52;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v919 = __riscv_vfcvt_f_x_v_f32m2(v918, 16);
        vfloat32m2_t v920 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        vfloat32m2_t v921 = __riscv_vfmacc_vv_f32m2(v920, v919, v917, 16);
        v26 = v921;
      }
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v922 = v12 * 4;
      size_t v923 = v922 + 0;
      size_t v924 = v923 * v2;
      size_t v925 = v16 * 16;
      size_t v926 = v924 + v925;
      float* v927 = v4 + v926;
      vfloat32m2_t v928 = v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v927, v928, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v929 = v12 * 4;
      size_t v930 = v929 + 1;
      size_t v931 = v930 * v2;
      size_t v932 = v16 * 16;
      size_t v933 = v931 + v932;
      float* v934 = v4 + v933;
      vfloat32m2_t v935 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v934, v935, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v936 = v12 * 4;
      size_t v937 = v936 + 2;
      size_t v938 = v937 * v2;
      size_t v939 = v16 * 16;
      size_t v940 = v938 + v939;
      float* v941 = v4 + v940;
      vfloat32m2_t v942 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v941, v942, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v943 = v12 * 4;
      size_t v944 = v943 + 3;
      size_t v945 = v944 * v2;
      size_t v946 = v16 * 16;
      size_t v947 = v945 + v946;
      float* v948 = v4 + v947;
      vfloat32m2_t v949 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v948, v949, 16);
    }
  }
  return;
}


