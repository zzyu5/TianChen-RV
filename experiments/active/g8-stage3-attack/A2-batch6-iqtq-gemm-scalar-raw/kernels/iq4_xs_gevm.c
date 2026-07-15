#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemv_iq4_xs_q8_K_kernel_ggml_repack_gemv_iq4_xs_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_iq4_xs_repack_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=col_group_count
  size_t v8 = v5 / 16;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_group_base
    size_t v10 = v9 * v7;
    size_t v11 = v10 * 2176;
    const uint8_t* v12 = v3 + v11;
    vfloat32m2_t v13;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v14 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v13 = v14;
    vfloat32m2_t v15;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v16 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v15 = v16;
    for (size_t v17 = 0; v17 < v7; v17 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
      size_t v18 = v17 * 2176;
      const uint8_t* v19 = v12 + v18;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
      size_t v20 = v17 * 292;
      const uint8_t* v21 = v4 + v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
      const float* v22 = (const float*) v21;
      float v23 = *(const float *)(v22);
      vint32m2_t v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v25 = __riscv_vmv_v_x_i32m2(0, 8);
      v24 = v25;
      vint32m2_t v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v27 = __riscv_vmv_v_x_i32m2(0, 8);
      v26 = v27;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_signed_scale
      const uint8_t* v28 = v19 + 64;
      const uint8_t* v29 = (const uint8_t*) v28;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v30 = __riscv_vle8_v_u8mf2(v29, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v31 = __riscv_vand_vx_u8mf2(v30, 0x0F, 8);
      const uint8_t* v32 = v19 + 32;
      const uint8_t* v33 = (const uint8_t*) v32;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v34 = __riscv_vle8_v_u8mf2(v33, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v35 = __riscv_vsrl_vx_u8mf2(v34, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v36 = __riscv_vand_vx_u8mf2(v35, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v37 = __riscv_vsll_vx_u8mf2(v36, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v38 = __riscv_vor_vv_u8mf2(v37, v31, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v39 = __riscv_vreinterpret_v_u8mf2_i8mf2(v38);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v40 = __riscv_vsub_vx_i8mf2(v39, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v41 = __riscv_vsext_vf4_i32m2(v40, 8);
      const uint8_t* v42 = v19 + 72;
      const uint8_t* v43 = (const uint8_t*) v42;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v44 = __riscv_vle8_v_u8mf2(v43, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v45 = __riscv_vand_vx_u8mf2(v44, 0x0F, 8);
      const uint8_t* v46 = v19 + 40;
      const uint8_t* v47 = (const uint8_t*) v46;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v48 = __riscv_vle8_v_u8mf2(v47, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v49 = __riscv_vsrl_vx_u8mf2(v48, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v50 = __riscv_vand_vx_u8mf2(v49, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v51 = __riscv_vsll_vx_u8mf2(v50, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v52 = __riscv_vor_vv_u8mf2(v51, v45, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v53 = __riscv_vreinterpret_v_u8mf2_i8mf2(v52);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v54 = __riscv_vsub_vx_i8mf2(v53, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v55 = __riscv_vsext_vf4_i32m2(v54, 8);
      vint32m2_t v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v57 = __riscv_vmv_v_x_i32m2(0, 8);
      v56 = v57;
      vint32m2_t v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v59 = __riscv_vmv_v_x_i32m2(0, 8);
      v58 = v59;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v60 = v19 + 128;
      const uint8_t* v61 = (const uint8_t*) v60;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v62 = __riscv_vle8_v_u8mf2(v61, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v63 = __riscv_vand_vx_u8mf2(v62, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v64 = __riscv_vzext_vf2_u16m1(v63, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v65 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v66 = __riscv_vsrl_vx_u8mf2(v62, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v67 = __riscv_vzext_vf2_u16m1(v66, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v68 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v67, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v69 = v21 + 4;
      const int8_t* v70 = (const int8_t*) v69;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v71 = *(const int8_t *)(v70);
      const uint8_t* v72 = v21 + 20;
      const int8_t* v73 = (const int8_t*) v72;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v74 = *(const int8_t *)(v73);
      vint32m2_t v75 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v76 = __riscv_vwmul_vx_i16m1(v65, v71, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v77 = __riscv_vwadd_wv_i32m2(v75, v76, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v78 = __riscv_vwmul_vx_i16m1(v68, v74, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v79 = __riscv_vwadd_wv_i32m2(v77, v78, 8);
      v56 = v79;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v80 = v19 + 136;
      const uint8_t* v81 = (const uint8_t*) v80;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v82 = __riscv_vle8_v_u8mf2(v81, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v83 = __riscv_vand_vx_u8mf2(v82, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v84 = __riscv_vzext_vf2_u16m1(v83, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v85 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v84, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v86 = __riscv_vsrl_vx_u8mf2(v82, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v87 = __riscv_vzext_vf2_u16m1(v86, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v88 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v87, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v89 = v21 + 4;
      const int8_t* v90 = (const int8_t*) v89;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v91 = *(const int8_t *)(v90);
      const uint8_t* v92 = v21 + 20;
      const int8_t* v93 = (const int8_t*) v92;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v94 = *(const int8_t *)(v93);
      vint32m2_t v95 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v96 = __riscv_vwmul_vx_i16m1(v85, v91, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v97 = __riscv_vwadd_wv_i32m2(v95, v96, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v98 = __riscv_vwmul_vx_i16m1(v88, v94, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v99 = __riscv_vwadd_wv_i32m2(v97, v98, 8);
      v58 = v99;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v100 = v19 + 144;
      const uint8_t* v101 = (const uint8_t*) v100;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v102 = __riscv_vle8_v_u8mf2(v101, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v103 = __riscv_vand_vx_u8mf2(v102, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v104 = __riscv_vzext_vf2_u16m1(v103, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v105 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v104, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v106 = __riscv_vsrl_vx_u8mf2(v102, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v107 = __riscv_vzext_vf2_u16m1(v106, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v108 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v107, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v109 = v21 + 5;
      const int8_t* v110 = (const int8_t*) v109;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v111 = *(const int8_t *)(v110);
      const uint8_t* v112 = v21 + 21;
      const int8_t* v113 = (const int8_t*) v112;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v114 = *(const int8_t *)(v113);
      vint32m2_t v115 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v116 = __riscv_vwmul_vx_i16m1(v105, v111, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v117 = __riscv_vwadd_wv_i32m2(v115, v116, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v118 = __riscv_vwmul_vx_i16m1(v108, v114, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v119 = __riscv_vwadd_wv_i32m2(v117, v118, 8);
      v56 = v119;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v120 = v19 + 152;
      const uint8_t* v121 = (const uint8_t*) v120;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v122 = __riscv_vle8_v_u8mf2(v121, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v123 = __riscv_vand_vx_u8mf2(v122, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v124 = __riscv_vzext_vf2_u16m1(v123, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v125 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v124, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v126 = __riscv_vsrl_vx_u8mf2(v122, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v127 = __riscv_vzext_vf2_u16m1(v126, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v128 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v127, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v129 = v21 + 5;
      const int8_t* v130 = (const int8_t*) v129;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v131 = *(const int8_t *)(v130);
      const uint8_t* v132 = v21 + 21;
      const int8_t* v133 = (const int8_t*) v132;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v134 = *(const int8_t *)(v133);
      vint32m2_t v135 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v136 = __riscv_vwmul_vx_i16m1(v125, v131, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v137 = __riscv_vwadd_wv_i32m2(v135, v136, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v138 = __riscv_vwmul_vx_i16m1(v128, v134, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v139 = __riscv_vwadd_wv_i32m2(v137, v138, 8);
      v58 = v139;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v140 = v19 + 160;
      const uint8_t* v141 = (const uint8_t*) v140;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v142 = __riscv_vle8_v_u8mf2(v141, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v143 = __riscv_vand_vx_u8mf2(v142, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v144 = __riscv_vzext_vf2_u16m1(v143, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v145 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v144, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v146 = __riscv_vsrl_vx_u8mf2(v142, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v147 = __riscv_vzext_vf2_u16m1(v146, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v148 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v147, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v149 = v21 + 6;
      const int8_t* v150 = (const int8_t*) v149;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v151 = *(const int8_t *)(v150);
      const uint8_t* v152 = v21 + 22;
      const int8_t* v153 = (const int8_t*) v152;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v154 = *(const int8_t *)(v153);
      vint32m2_t v155 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v156 = __riscv_vwmul_vx_i16m1(v145, v151, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v157 = __riscv_vwadd_wv_i32m2(v155, v156, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v158 = __riscv_vwmul_vx_i16m1(v148, v154, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v159 = __riscv_vwadd_wv_i32m2(v157, v158, 8);
      v56 = v159;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v160 = v19 + 168;
      const uint8_t* v161 = (const uint8_t*) v160;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v162 = __riscv_vle8_v_u8mf2(v161, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v163 = __riscv_vand_vx_u8mf2(v162, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v164 = __riscv_vzext_vf2_u16m1(v163, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v165 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v164, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v166 = __riscv_vsrl_vx_u8mf2(v162, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v167 = __riscv_vzext_vf2_u16m1(v166, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v168 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v167, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v169 = v21 + 6;
      const int8_t* v170 = (const int8_t*) v169;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v171 = *(const int8_t *)(v170);
      const uint8_t* v172 = v21 + 22;
      const int8_t* v173 = (const int8_t*) v172;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v174 = *(const int8_t *)(v173);
      vint32m2_t v175 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v176 = __riscv_vwmul_vx_i16m1(v165, v171, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v177 = __riscv_vwadd_wv_i32m2(v175, v176, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v178 = __riscv_vwmul_vx_i16m1(v168, v174, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v179 = __riscv_vwadd_wv_i32m2(v177, v178, 8);
      v58 = v179;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v180 = v19 + 176;
      const uint8_t* v181 = (const uint8_t*) v180;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v182 = __riscv_vle8_v_u8mf2(v181, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v183 = __riscv_vand_vx_u8mf2(v182, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v184 = __riscv_vzext_vf2_u16m1(v183, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v185 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v184, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v186 = __riscv_vsrl_vx_u8mf2(v182, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v187 = __riscv_vzext_vf2_u16m1(v186, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v188 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v187, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v189 = v21 + 7;
      const int8_t* v190 = (const int8_t*) v189;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v191 = *(const int8_t *)(v190);
      const uint8_t* v192 = v21 + 23;
      const int8_t* v193 = (const int8_t*) v192;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v194 = *(const int8_t *)(v193);
      vint32m2_t v195 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v196 = __riscv_vwmul_vx_i16m1(v185, v191, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v197 = __riscv_vwadd_wv_i32m2(v195, v196, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v198 = __riscv_vwmul_vx_i16m1(v188, v194, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v199 = __riscv_vwadd_wv_i32m2(v197, v198, 8);
      v56 = v199;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v200 = v19 + 184;
      const uint8_t* v201 = (const uint8_t*) v200;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v202 = __riscv_vle8_v_u8mf2(v201, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v203 = __riscv_vand_vx_u8mf2(v202, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v204 = __riscv_vzext_vf2_u16m1(v203, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v205 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v204, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v206 = __riscv_vsrl_vx_u8mf2(v202, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v207 = __riscv_vzext_vf2_u16m1(v206, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v208 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v207, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v209 = v21 + 7;
      const int8_t* v210 = (const int8_t*) v209;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v211 = *(const int8_t *)(v210);
      const uint8_t* v212 = v21 + 23;
      const int8_t* v213 = (const int8_t*) v212;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v214 = *(const int8_t *)(v213);
      vint32m2_t v215 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v216 = __riscv_vwmul_vx_i16m1(v205, v211, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v217 = __riscv_vwadd_wv_i32m2(v215, v216, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v218 = __riscv_vwmul_vx_i16m1(v208, v214, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v219 = __riscv_vwadd_wv_i32m2(v217, v218, 8);
      v58 = v219;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v220 = v19 + 192;
      const uint8_t* v221 = (const uint8_t*) v220;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v222 = __riscv_vle8_v_u8mf2(v221, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v223 = __riscv_vand_vx_u8mf2(v222, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v224 = __riscv_vzext_vf2_u16m1(v223, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v225 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v224, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v226 = __riscv_vsrl_vx_u8mf2(v222, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v227 = __riscv_vzext_vf2_u16m1(v226, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v228 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v227, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v229 = v21 + 8;
      const int8_t* v230 = (const int8_t*) v229;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v231 = *(const int8_t *)(v230);
      const uint8_t* v232 = v21 + 24;
      const int8_t* v233 = (const int8_t*) v232;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v234 = *(const int8_t *)(v233);
      vint32m2_t v235 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v236 = __riscv_vwmul_vx_i16m1(v225, v231, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v237 = __riscv_vwadd_wv_i32m2(v235, v236, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v238 = __riscv_vwmul_vx_i16m1(v228, v234, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v239 = __riscv_vwadd_wv_i32m2(v237, v238, 8);
      v56 = v239;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v240 = v19 + 200;
      const uint8_t* v241 = (const uint8_t*) v240;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v242 = __riscv_vle8_v_u8mf2(v241, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v243 = __riscv_vand_vx_u8mf2(v242, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v244 = __riscv_vzext_vf2_u16m1(v243, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v245 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v244, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v246 = __riscv_vsrl_vx_u8mf2(v242, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v247 = __riscv_vzext_vf2_u16m1(v246, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v248 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v247, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v249 = v21 + 8;
      const int8_t* v250 = (const int8_t*) v249;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v251 = *(const int8_t *)(v250);
      const uint8_t* v252 = v21 + 24;
      const int8_t* v253 = (const int8_t*) v252;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v254 = *(const int8_t *)(v253);
      vint32m2_t v255 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v256 = __riscv_vwmul_vx_i16m1(v245, v251, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v257 = __riscv_vwadd_wv_i32m2(v255, v256, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v258 = __riscv_vwmul_vx_i16m1(v248, v254, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v259 = __riscv_vwadd_wv_i32m2(v257, v258, 8);
      v58 = v259;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v260 = v19 + 208;
      const uint8_t* v261 = (const uint8_t*) v260;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v262 = __riscv_vle8_v_u8mf2(v261, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v263 = __riscv_vand_vx_u8mf2(v262, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v264 = __riscv_vzext_vf2_u16m1(v263, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v265 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v264, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v266 = __riscv_vsrl_vx_u8mf2(v262, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v267 = __riscv_vzext_vf2_u16m1(v266, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v268 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v267, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v269 = v21 + 9;
      const int8_t* v270 = (const int8_t*) v269;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v271 = *(const int8_t *)(v270);
      const uint8_t* v272 = v21 + 25;
      const int8_t* v273 = (const int8_t*) v272;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v274 = *(const int8_t *)(v273);
      vint32m2_t v275 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v276 = __riscv_vwmul_vx_i16m1(v265, v271, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v277 = __riscv_vwadd_wv_i32m2(v275, v276, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v278 = __riscv_vwmul_vx_i16m1(v268, v274, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v279 = __riscv_vwadd_wv_i32m2(v277, v278, 8);
      v56 = v279;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v280 = v19 + 216;
      const uint8_t* v281 = (const uint8_t*) v280;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v282 = __riscv_vle8_v_u8mf2(v281, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v283 = __riscv_vand_vx_u8mf2(v282, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v284 = __riscv_vzext_vf2_u16m1(v283, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v285 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v284, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v286 = __riscv_vsrl_vx_u8mf2(v282, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v287 = __riscv_vzext_vf2_u16m1(v286, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v288 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v287, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v289 = v21 + 9;
      const int8_t* v290 = (const int8_t*) v289;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v291 = *(const int8_t *)(v290);
      const uint8_t* v292 = v21 + 25;
      const int8_t* v293 = (const int8_t*) v292;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v294 = *(const int8_t *)(v293);
      vint32m2_t v295 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v296 = __riscv_vwmul_vx_i16m1(v285, v291, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v297 = __riscv_vwadd_wv_i32m2(v295, v296, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v298 = __riscv_vwmul_vx_i16m1(v288, v294, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v299 = __riscv_vwadd_wv_i32m2(v297, v298, 8);
      v58 = v299;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v300 = v19 + 224;
      const uint8_t* v301 = (const uint8_t*) v300;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v302 = __riscv_vle8_v_u8mf2(v301, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v303 = __riscv_vand_vx_u8mf2(v302, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v304 = __riscv_vzext_vf2_u16m1(v303, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v305 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v304, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v306 = __riscv_vsrl_vx_u8mf2(v302, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v307 = __riscv_vzext_vf2_u16m1(v306, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v308 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v307, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v309 = v21 + 10;
      const int8_t* v310 = (const int8_t*) v309;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v311 = *(const int8_t *)(v310);
      const uint8_t* v312 = v21 + 26;
      const int8_t* v313 = (const int8_t*) v312;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v314 = *(const int8_t *)(v313);
      vint32m2_t v315 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v316 = __riscv_vwmul_vx_i16m1(v305, v311, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v317 = __riscv_vwadd_wv_i32m2(v315, v316, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v318 = __riscv_vwmul_vx_i16m1(v308, v314, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v319 = __riscv_vwadd_wv_i32m2(v317, v318, 8);
      v56 = v319;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v320 = v19 + 232;
      const uint8_t* v321 = (const uint8_t*) v320;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v322 = __riscv_vle8_v_u8mf2(v321, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v323 = __riscv_vand_vx_u8mf2(v322, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v324 = __riscv_vzext_vf2_u16m1(v323, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v325 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v324, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v326 = __riscv_vsrl_vx_u8mf2(v322, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v327 = __riscv_vzext_vf2_u16m1(v326, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v328 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v327, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v329 = v21 + 10;
      const int8_t* v330 = (const int8_t*) v329;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v331 = *(const int8_t *)(v330);
      const uint8_t* v332 = v21 + 26;
      const int8_t* v333 = (const int8_t*) v332;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v334 = *(const int8_t *)(v333);
      vint32m2_t v335 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v336 = __riscv_vwmul_vx_i16m1(v325, v331, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v337 = __riscv_vwadd_wv_i32m2(v335, v336, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v338 = __riscv_vwmul_vx_i16m1(v328, v334, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v339 = __riscv_vwadd_wv_i32m2(v337, v338, 8);
      v58 = v339;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v340 = v19 + 240;
      const uint8_t* v341 = (const uint8_t*) v340;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v342 = __riscv_vle8_v_u8mf2(v341, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v343 = __riscv_vand_vx_u8mf2(v342, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v344 = __riscv_vzext_vf2_u16m1(v343, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v345 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v344, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v346 = __riscv_vsrl_vx_u8mf2(v342, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v347 = __riscv_vzext_vf2_u16m1(v346, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v348 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v347, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v349 = v21 + 11;
      const int8_t* v350 = (const int8_t*) v349;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v351 = *(const int8_t *)(v350);
      const uint8_t* v352 = v21 + 27;
      const int8_t* v353 = (const int8_t*) v352;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v354 = *(const int8_t *)(v353);
      vint32m2_t v355 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v356 = __riscv_vwmul_vx_i16m1(v345, v351, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v357 = __riscv_vwadd_wv_i32m2(v355, v356, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v358 = __riscv_vwmul_vx_i16m1(v348, v354, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v359 = __riscv_vwadd_wv_i32m2(v357, v358, 8);
      v56 = v359;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v360 = v19 + 248;
      const uint8_t* v361 = (const uint8_t*) v360;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v362 = __riscv_vle8_v_u8mf2(v361, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v363 = __riscv_vand_vx_u8mf2(v362, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v364 = __riscv_vzext_vf2_u16m1(v363, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v365 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v364, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v366 = __riscv_vsrl_vx_u8mf2(v362, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v367 = __riscv_vzext_vf2_u16m1(v366, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v368 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v367, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v369 = v21 + 11;
      const int8_t* v370 = (const int8_t*) v369;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v371 = *(const int8_t *)(v370);
      const uint8_t* v372 = v21 + 27;
      const int8_t* v373 = (const int8_t*) v372;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v374 = *(const int8_t *)(v373);
      vint32m2_t v375 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v376 = __riscv_vwmul_vx_i16m1(v365, v371, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v377 = __riscv_vwadd_wv_i32m2(v375, v376, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v378 = __riscv_vwmul_vx_i16m1(v368, v374, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v379 = __riscv_vwadd_wv_i32m2(v377, v378, 8);
      v58 = v379;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v380 = v19 + 256;
      const uint8_t* v381 = (const uint8_t*) v380;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v382 = __riscv_vle8_v_u8mf2(v381, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v383 = __riscv_vand_vx_u8mf2(v382, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v384 = __riscv_vzext_vf2_u16m1(v383, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v385 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v384, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v386 = __riscv_vsrl_vx_u8mf2(v382, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v387 = __riscv_vzext_vf2_u16m1(v386, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v388 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v387, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v389 = v21 + 12;
      const int8_t* v390 = (const int8_t*) v389;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v391 = *(const int8_t *)(v390);
      const uint8_t* v392 = v21 + 28;
      const int8_t* v393 = (const int8_t*) v392;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v394 = *(const int8_t *)(v393);
      vint32m2_t v395 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v396 = __riscv_vwmul_vx_i16m1(v385, v391, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v397 = __riscv_vwadd_wv_i32m2(v395, v396, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v398 = __riscv_vwmul_vx_i16m1(v388, v394, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v399 = __riscv_vwadd_wv_i32m2(v397, v398, 8);
      v56 = v399;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v400 = v19 + 264;
      const uint8_t* v401 = (const uint8_t*) v400;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v402 = __riscv_vle8_v_u8mf2(v401, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v403 = __riscv_vand_vx_u8mf2(v402, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v404 = __riscv_vzext_vf2_u16m1(v403, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v405 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v404, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v406 = __riscv_vsrl_vx_u8mf2(v402, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v407 = __riscv_vzext_vf2_u16m1(v406, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v408 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v407, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v409 = v21 + 12;
      const int8_t* v410 = (const int8_t*) v409;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v411 = *(const int8_t *)(v410);
      const uint8_t* v412 = v21 + 28;
      const int8_t* v413 = (const int8_t*) v412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v414 = *(const int8_t *)(v413);
      vint32m2_t v415 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v416 = __riscv_vwmul_vx_i16m1(v405, v411, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v417 = __riscv_vwadd_wv_i32m2(v415, v416, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v418 = __riscv_vwmul_vx_i16m1(v408, v414, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v419 = __riscv_vwadd_wv_i32m2(v417, v418, 8);
      v58 = v419;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v420 = v19 + 272;
      const uint8_t* v421 = (const uint8_t*) v420;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v422 = __riscv_vle8_v_u8mf2(v421, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v423 = __riscv_vand_vx_u8mf2(v422, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v424 = __riscv_vzext_vf2_u16m1(v423, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v425 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v424, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v426 = __riscv_vsrl_vx_u8mf2(v422, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v427 = __riscv_vzext_vf2_u16m1(v426, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v428 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v427, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v429 = v21 + 13;
      const int8_t* v430 = (const int8_t*) v429;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v431 = *(const int8_t *)(v430);
      const uint8_t* v432 = v21 + 29;
      const int8_t* v433 = (const int8_t*) v432;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v434 = *(const int8_t *)(v433);
      vint32m2_t v435 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v436 = __riscv_vwmul_vx_i16m1(v425, v431, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v437 = __riscv_vwadd_wv_i32m2(v435, v436, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v438 = __riscv_vwmul_vx_i16m1(v428, v434, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v439 = __riscv_vwadd_wv_i32m2(v437, v438, 8);
      v56 = v439;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v440 = v19 + 280;
      const uint8_t* v441 = (const uint8_t*) v440;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v442 = __riscv_vle8_v_u8mf2(v441, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v443 = __riscv_vand_vx_u8mf2(v442, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v444 = __riscv_vzext_vf2_u16m1(v443, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v445 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v444, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v446 = __riscv_vsrl_vx_u8mf2(v442, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v447 = __riscv_vzext_vf2_u16m1(v446, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v448 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v447, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v449 = v21 + 13;
      const int8_t* v450 = (const int8_t*) v449;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v451 = *(const int8_t *)(v450);
      const uint8_t* v452 = v21 + 29;
      const int8_t* v453 = (const int8_t*) v452;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v454 = *(const int8_t *)(v453);
      vint32m2_t v455 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v456 = __riscv_vwmul_vx_i16m1(v445, v451, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v457 = __riscv_vwadd_wv_i32m2(v455, v456, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v458 = __riscv_vwmul_vx_i16m1(v448, v454, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v459 = __riscv_vwadd_wv_i32m2(v457, v458, 8);
      v58 = v459;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v460 = v19 + 288;
      const uint8_t* v461 = (const uint8_t*) v460;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v462 = __riscv_vle8_v_u8mf2(v461, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v463 = __riscv_vand_vx_u8mf2(v462, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v464 = __riscv_vzext_vf2_u16m1(v463, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v465 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v464, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v466 = __riscv_vsrl_vx_u8mf2(v462, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v467 = __riscv_vzext_vf2_u16m1(v466, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v468 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v467, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v469 = v21 + 14;
      const int8_t* v470 = (const int8_t*) v469;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v471 = *(const int8_t *)(v470);
      const uint8_t* v472 = v21 + 30;
      const int8_t* v473 = (const int8_t*) v472;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v474 = *(const int8_t *)(v473);
      vint32m2_t v475 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v476 = __riscv_vwmul_vx_i16m1(v465, v471, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v477 = __riscv_vwadd_wv_i32m2(v475, v476, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v478 = __riscv_vwmul_vx_i16m1(v468, v474, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v479 = __riscv_vwadd_wv_i32m2(v477, v478, 8);
      v56 = v479;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v480 = v19 + 296;
      const uint8_t* v481 = (const uint8_t*) v480;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v482 = __riscv_vle8_v_u8mf2(v481, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v483 = __riscv_vand_vx_u8mf2(v482, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v484 = __riscv_vzext_vf2_u16m1(v483, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v485 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v484, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v486 = __riscv_vsrl_vx_u8mf2(v482, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v487 = __riscv_vzext_vf2_u16m1(v486, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v488 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v487, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v489 = v21 + 14;
      const int8_t* v490 = (const int8_t*) v489;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v491 = *(const int8_t *)(v490);
      const uint8_t* v492 = v21 + 30;
      const int8_t* v493 = (const int8_t*) v492;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v494 = *(const int8_t *)(v493);
      vint32m2_t v495 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v496 = __riscv_vwmul_vx_i16m1(v485, v491, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v497 = __riscv_vwadd_wv_i32m2(v495, v496, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v498 = __riscv_vwmul_vx_i16m1(v488, v494, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v499 = __riscv_vwadd_wv_i32m2(v497, v498, 8);
      v58 = v499;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v500 = v19 + 304;
      const uint8_t* v501 = (const uint8_t*) v500;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v502 = __riscv_vle8_v_u8mf2(v501, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v503 = __riscv_vand_vx_u8mf2(v502, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v504 = __riscv_vzext_vf2_u16m1(v503, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v505 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v504, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v506 = __riscv_vsrl_vx_u8mf2(v502, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v507 = __riscv_vzext_vf2_u16m1(v506, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v508 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v507, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v509 = v21 + 15;
      const int8_t* v510 = (const int8_t*) v509;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v511 = *(const int8_t *)(v510);
      const uint8_t* v512 = v21 + 31;
      const int8_t* v513 = (const int8_t*) v512;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v514 = *(const int8_t *)(v513);
      vint32m2_t v515 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v516 = __riscv_vwmul_vx_i16m1(v505, v511, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v517 = __riscv_vwadd_wv_i32m2(v515, v516, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v518 = __riscv_vwmul_vx_i16m1(v508, v514, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v519 = __riscv_vwadd_wv_i32m2(v517, v518, 8);
      v56 = v519;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v520 = v19 + 312;
      const uint8_t* v521 = (const uint8_t*) v520;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v522 = __riscv_vle8_v_u8mf2(v521, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v523 = __riscv_vand_vx_u8mf2(v522, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v524 = __riscv_vzext_vf2_u16m1(v523, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v525 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v524, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v526 = __riscv_vsrl_vx_u8mf2(v522, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v527 = __riscv_vzext_vf2_u16m1(v526, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v528 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v527, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v529 = v21 + 15;
      const int8_t* v530 = (const int8_t*) v529;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v531 = *(const int8_t *)(v530);
      const uint8_t* v532 = v21 + 31;
      const int8_t* v533 = (const int8_t*) v532;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v534 = *(const int8_t *)(v533);
      vint32m2_t v535 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v536 = __riscv_vwmul_vx_i16m1(v525, v531, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v537 = __riscv_vwadd_wv_i32m2(v535, v536, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v538 = __riscv_vwmul_vx_i16m1(v528, v534, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v539 = __riscv_vwadd_wv_i32m2(v537, v538, 8);
      v58 = v539;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v540 = v19 + 320;
      const uint8_t* v541 = (const uint8_t*) v540;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v542 = __riscv_vle8_v_u8mf2(v541, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v543 = __riscv_vand_vx_u8mf2(v542, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v544 = __riscv_vzext_vf2_u16m1(v543, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v545 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v544, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v546 = __riscv_vsrl_vx_u8mf2(v542, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v547 = __riscv_vzext_vf2_u16m1(v546, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v548 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v547, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v549 = v21 + 16;
      const int8_t* v550 = (const int8_t*) v549;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v551 = *(const int8_t *)(v550);
      const uint8_t* v552 = v21 + 32;
      const int8_t* v553 = (const int8_t*) v552;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v554 = *(const int8_t *)(v553);
      vint32m2_t v555 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v556 = __riscv_vwmul_vx_i16m1(v545, v551, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v557 = __riscv_vwadd_wv_i32m2(v555, v556, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v558 = __riscv_vwmul_vx_i16m1(v548, v554, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v559 = __riscv_vwadd_wv_i32m2(v557, v558, 8);
      v56 = v559;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v560 = v19 + 328;
      const uint8_t* v561 = (const uint8_t*) v560;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v562 = __riscv_vle8_v_u8mf2(v561, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v563 = __riscv_vand_vx_u8mf2(v562, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v564 = __riscv_vzext_vf2_u16m1(v563, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v565 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v564, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v566 = __riscv_vsrl_vx_u8mf2(v562, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v567 = __riscv_vzext_vf2_u16m1(v566, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v568 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v567, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v569 = v21 + 16;
      const int8_t* v570 = (const int8_t*) v569;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v571 = *(const int8_t *)(v570);
      const uint8_t* v572 = v21 + 32;
      const int8_t* v573 = (const int8_t*) v572;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v574 = *(const int8_t *)(v573);
      vint32m2_t v575 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v576 = __riscv_vwmul_vx_i16m1(v565, v571, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v577 = __riscv_vwadd_wv_i32m2(v575, v576, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v578 = __riscv_vwmul_vx_i16m1(v568, v574, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v579 = __riscv_vwadd_wv_i32m2(v577, v578, 8);
      v58 = v579;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v580 = v19 + 336;
      const uint8_t* v581 = (const uint8_t*) v580;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v582 = __riscv_vle8_v_u8mf2(v581, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v583 = __riscv_vand_vx_u8mf2(v582, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v584 = __riscv_vzext_vf2_u16m1(v583, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v585 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v584, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v586 = __riscv_vsrl_vx_u8mf2(v582, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v587 = __riscv_vzext_vf2_u16m1(v586, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v588 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v587, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v589 = v21 + 17;
      const int8_t* v590 = (const int8_t*) v589;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v591 = *(const int8_t *)(v590);
      const uint8_t* v592 = v21 + 33;
      const int8_t* v593 = (const int8_t*) v592;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v594 = *(const int8_t *)(v593);
      vint32m2_t v595 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v596 = __riscv_vwmul_vx_i16m1(v585, v591, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v597 = __riscv_vwadd_wv_i32m2(v595, v596, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v598 = __riscv_vwmul_vx_i16m1(v588, v594, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v599 = __riscv_vwadd_wv_i32m2(v597, v598, 8);
      v56 = v599;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v600 = v19 + 344;
      const uint8_t* v601 = (const uint8_t*) v600;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v602 = __riscv_vle8_v_u8mf2(v601, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v603 = __riscv_vand_vx_u8mf2(v602, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v604 = __riscv_vzext_vf2_u16m1(v603, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v605 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v604, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v606 = __riscv_vsrl_vx_u8mf2(v602, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v607 = __riscv_vzext_vf2_u16m1(v606, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v608 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v607, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v609 = v21 + 17;
      const int8_t* v610 = (const int8_t*) v609;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v611 = *(const int8_t *)(v610);
      const uint8_t* v612 = v21 + 33;
      const int8_t* v613 = (const int8_t*) v612;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v614 = *(const int8_t *)(v613);
      vint32m2_t v615 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v616 = __riscv_vwmul_vx_i16m1(v605, v611, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v617 = __riscv_vwadd_wv_i32m2(v615, v616, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v618 = __riscv_vwmul_vx_i16m1(v608, v614, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v619 = __riscv_vwadd_wv_i32m2(v617, v618, 8);
      v58 = v619;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v620 = v19 + 352;
      const uint8_t* v621 = (const uint8_t*) v620;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v622 = __riscv_vle8_v_u8mf2(v621, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v623 = __riscv_vand_vx_u8mf2(v622, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v624 = __riscv_vzext_vf2_u16m1(v623, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v625 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v624, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v626 = __riscv_vsrl_vx_u8mf2(v622, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v627 = __riscv_vzext_vf2_u16m1(v626, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v628 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v627, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v629 = v21 + 18;
      const int8_t* v630 = (const int8_t*) v629;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v631 = *(const int8_t *)(v630);
      const uint8_t* v632 = v21 + 34;
      const int8_t* v633 = (const int8_t*) v632;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v634 = *(const int8_t *)(v633);
      vint32m2_t v635 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v636 = __riscv_vwmul_vx_i16m1(v625, v631, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v637 = __riscv_vwadd_wv_i32m2(v635, v636, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v638 = __riscv_vwmul_vx_i16m1(v628, v634, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v639 = __riscv_vwadd_wv_i32m2(v637, v638, 8);
      v56 = v639;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v640 = v19 + 360;
      const uint8_t* v641 = (const uint8_t*) v640;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v642 = __riscv_vle8_v_u8mf2(v641, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v643 = __riscv_vand_vx_u8mf2(v642, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v644 = __riscv_vzext_vf2_u16m1(v643, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v645 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v644, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v646 = __riscv_vsrl_vx_u8mf2(v642, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v647 = __riscv_vzext_vf2_u16m1(v646, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v648 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v647, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v649 = v21 + 18;
      const int8_t* v650 = (const int8_t*) v649;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v651 = *(const int8_t *)(v650);
      const uint8_t* v652 = v21 + 34;
      const int8_t* v653 = (const int8_t*) v652;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v654 = *(const int8_t *)(v653);
      vint32m2_t v655 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v656 = __riscv_vwmul_vx_i16m1(v645, v651, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v657 = __riscv_vwadd_wv_i32m2(v655, v656, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v658 = __riscv_vwmul_vx_i16m1(v648, v654, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v659 = __riscv_vwadd_wv_i32m2(v657, v658, 8);
      v58 = v659;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v660 = v19 + 368;
      const uint8_t* v661 = (const uint8_t*) v660;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v662 = __riscv_vle8_v_u8mf2(v661, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v663 = __riscv_vand_vx_u8mf2(v662, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v664 = __riscv_vzext_vf2_u16m1(v663, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v665 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v664, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v666 = __riscv_vsrl_vx_u8mf2(v662, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v667 = __riscv_vzext_vf2_u16m1(v666, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v668 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v667, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v669 = v21 + 19;
      const int8_t* v670 = (const int8_t*) v669;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v671 = *(const int8_t *)(v670);
      const uint8_t* v672 = v21 + 35;
      const int8_t* v673 = (const int8_t*) v672;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v674 = *(const int8_t *)(v673);
      vint32m2_t v675 = v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v676 = __riscv_vwmul_vx_i16m1(v665, v671, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v677 = __riscv_vwadd_wv_i32m2(v675, v676, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v678 = __riscv_vwmul_vx_i16m1(v668, v674, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v679 = __riscv_vwadd_wv_i32m2(v677, v678, 8);
      v56 = v679;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v680 = v19 + 376;
      const uint8_t* v681 = (const uint8_t*) v680;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v682 = __riscv_vle8_v_u8mf2(v681, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v683 = __riscv_vand_vx_u8mf2(v682, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v684 = __riscv_vzext_vf2_u16m1(v683, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v685 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v684, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v686 = __riscv_vsrl_vx_u8mf2(v682, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v687 = __riscv_vzext_vf2_u16m1(v686, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v688 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v687, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v689 = v21 + 19;
      const int8_t* v690 = (const int8_t*) v689;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v691 = *(const int8_t *)(v690);
      const uint8_t* v692 = v21 + 35;
      const int8_t* v693 = (const int8_t*) v692;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v694 = *(const int8_t *)(v693);
      vint32m2_t v695 = v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v696 = __riscv_vwmul_vx_i16m1(v685, v691, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v697 = __riscv_vwadd_wv_i32m2(v695, v696, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v698 = __riscv_vwmul_vx_i16m1(v688, v694, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v699 = __riscv_vwadd_wv_i32m2(v697, v698, 8);
      v58 = v699;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_scale_fold
      vint32m2_t v700 = v56;
      vint32m2_t v701 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v702 = __riscv_vmacc_vv_i32m2(v701, v41, v700, 8);
      v24 = v702;
      vint32m2_t v703 = v58;
      vint32m2_t v704 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v705 = __riscv_vmacc_vv_i32m2(v704, v55, v703, 8);
      v26 = v705;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_signed_scale
      const uint8_t* v706 = v19 + 64;
      const uint8_t* v707 = (const uint8_t*) v706;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v708 = __riscv_vle8_v_u8mf2(v707, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v709 = __riscv_vsrl_vx_u8mf2(v708, 4, 8);
      const uint8_t* v710 = v19 + 32;
      const uint8_t* v711 = (const uint8_t*) v710;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v712 = __riscv_vle8_v_u8mf2(v711, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v713 = __riscv_vsrl_vx_u8mf2(v712, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v714 = __riscv_vand_vx_u8mf2(v713, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v715 = __riscv_vsll_vx_u8mf2(v714, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v716 = __riscv_vor_vv_u8mf2(v715, v709, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v717 = __riscv_vreinterpret_v_u8mf2_i8mf2(v716);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v718 = __riscv_vsub_vx_i8mf2(v717, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v719 = __riscv_vsext_vf4_i32m2(v718, 8);
      const uint8_t* v720 = v19 + 72;
      const uint8_t* v721 = (const uint8_t*) v720;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v722 = __riscv_vle8_v_u8mf2(v721, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v723 = __riscv_vsrl_vx_u8mf2(v722, 4, 8);
      const uint8_t* v724 = v19 + 40;
      const uint8_t* v725 = (const uint8_t*) v724;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v726 = __riscv_vle8_v_u8mf2(v725, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v727 = __riscv_vsrl_vx_u8mf2(v726, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v728 = __riscv_vand_vx_u8mf2(v727, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v729 = __riscv_vsll_vx_u8mf2(v728, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v730 = __riscv_vor_vv_u8mf2(v729, v723, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v731 = __riscv_vreinterpret_v_u8mf2_i8mf2(v730);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v732 = __riscv_vsub_vx_i8mf2(v731, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v733 = __riscv_vsext_vf4_i32m2(v732, 8);
      vint32m2_t v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v735 = __riscv_vmv_v_x_i32m2(0, 8);
      v734 = v735;
      vint32m2_t v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v737 = __riscv_vmv_v_x_i32m2(0, 8);
      v736 = v737;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v738 = v19 + 384;
      const uint8_t* v739 = (const uint8_t*) v738;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v740 = __riscv_vle8_v_u8mf2(v739, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v741 = __riscv_vand_vx_u8mf2(v740, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v742 = __riscv_vzext_vf2_u16m1(v741, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v743 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v742, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v744 = __riscv_vsrl_vx_u8mf2(v740, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v745 = __riscv_vzext_vf2_u16m1(v744, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v746 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v745, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v747 = v21 + 36;
      const int8_t* v748 = (const int8_t*) v747;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v749 = *(const int8_t *)(v748);
      const uint8_t* v750 = v21 + 52;
      const int8_t* v751 = (const int8_t*) v750;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v752 = *(const int8_t *)(v751);
      vint32m2_t v753 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v754 = __riscv_vwmul_vx_i16m1(v743, v749, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v755 = __riscv_vwadd_wv_i32m2(v753, v754, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v756 = __riscv_vwmul_vx_i16m1(v746, v752, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v757 = __riscv_vwadd_wv_i32m2(v755, v756, 8);
      v734 = v757;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v758 = v19 + 392;
      const uint8_t* v759 = (const uint8_t*) v758;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v760 = __riscv_vle8_v_u8mf2(v759, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v761 = __riscv_vand_vx_u8mf2(v760, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v762 = __riscv_vzext_vf2_u16m1(v761, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v763 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v762, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v764 = __riscv_vsrl_vx_u8mf2(v760, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v765 = __riscv_vzext_vf2_u16m1(v764, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v766 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v765, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v767 = v21 + 36;
      const int8_t* v768 = (const int8_t*) v767;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v769 = *(const int8_t *)(v768);
      const uint8_t* v770 = v21 + 52;
      const int8_t* v771 = (const int8_t*) v770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v772 = *(const int8_t *)(v771);
      vint32m2_t v773 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v774 = __riscv_vwmul_vx_i16m1(v763, v769, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v775 = __riscv_vwadd_wv_i32m2(v773, v774, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v776 = __riscv_vwmul_vx_i16m1(v766, v772, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v777 = __riscv_vwadd_wv_i32m2(v775, v776, 8);
      v736 = v777;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v778 = v19 + 400;
      const uint8_t* v779 = (const uint8_t*) v778;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v780 = __riscv_vle8_v_u8mf2(v779, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v781 = __riscv_vand_vx_u8mf2(v780, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v782 = __riscv_vzext_vf2_u16m1(v781, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v783 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v782, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v784 = __riscv_vsrl_vx_u8mf2(v780, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v785 = __riscv_vzext_vf2_u16m1(v784, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v786 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v785, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v787 = v21 + 37;
      const int8_t* v788 = (const int8_t*) v787;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v789 = *(const int8_t *)(v788);
      const uint8_t* v790 = v21 + 53;
      const int8_t* v791 = (const int8_t*) v790;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v792 = *(const int8_t *)(v791);
      vint32m2_t v793 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v794 = __riscv_vwmul_vx_i16m1(v783, v789, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v795 = __riscv_vwadd_wv_i32m2(v793, v794, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v796 = __riscv_vwmul_vx_i16m1(v786, v792, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v797 = __riscv_vwadd_wv_i32m2(v795, v796, 8);
      v734 = v797;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v798 = v19 + 408;
      const uint8_t* v799 = (const uint8_t*) v798;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v800 = __riscv_vle8_v_u8mf2(v799, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v801 = __riscv_vand_vx_u8mf2(v800, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v802 = __riscv_vzext_vf2_u16m1(v801, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v803 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v802, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v804 = __riscv_vsrl_vx_u8mf2(v800, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v805 = __riscv_vzext_vf2_u16m1(v804, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v806 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v805, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v807 = v21 + 37;
      const int8_t* v808 = (const int8_t*) v807;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v809 = *(const int8_t *)(v808);
      const uint8_t* v810 = v21 + 53;
      const int8_t* v811 = (const int8_t*) v810;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v812 = *(const int8_t *)(v811);
      vint32m2_t v813 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v814 = __riscv_vwmul_vx_i16m1(v803, v809, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v815 = __riscv_vwadd_wv_i32m2(v813, v814, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v816 = __riscv_vwmul_vx_i16m1(v806, v812, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v817 = __riscv_vwadd_wv_i32m2(v815, v816, 8);
      v736 = v817;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v818 = v19 + 416;
      const uint8_t* v819 = (const uint8_t*) v818;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v820 = __riscv_vle8_v_u8mf2(v819, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v821 = __riscv_vand_vx_u8mf2(v820, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v822 = __riscv_vzext_vf2_u16m1(v821, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v823 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v822, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v824 = __riscv_vsrl_vx_u8mf2(v820, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v825 = __riscv_vzext_vf2_u16m1(v824, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v826 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v825, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v827 = v21 + 38;
      const int8_t* v828 = (const int8_t*) v827;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v829 = *(const int8_t *)(v828);
      const uint8_t* v830 = v21 + 54;
      const int8_t* v831 = (const int8_t*) v830;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v832 = *(const int8_t *)(v831);
      vint32m2_t v833 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v834 = __riscv_vwmul_vx_i16m1(v823, v829, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v835 = __riscv_vwadd_wv_i32m2(v833, v834, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v836 = __riscv_vwmul_vx_i16m1(v826, v832, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v837 = __riscv_vwadd_wv_i32m2(v835, v836, 8);
      v734 = v837;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v838 = v19 + 424;
      const uint8_t* v839 = (const uint8_t*) v838;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v840 = __riscv_vle8_v_u8mf2(v839, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v841 = __riscv_vand_vx_u8mf2(v840, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v842 = __riscv_vzext_vf2_u16m1(v841, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v843 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v842, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v844 = __riscv_vsrl_vx_u8mf2(v840, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v845 = __riscv_vzext_vf2_u16m1(v844, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v846 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v845, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v847 = v21 + 38;
      const int8_t* v848 = (const int8_t*) v847;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v849 = *(const int8_t *)(v848);
      const uint8_t* v850 = v21 + 54;
      const int8_t* v851 = (const int8_t*) v850;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v852 = *(const int8_t *)(v851);
      vint32m2_t v853 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v854 = __riscv_vwmul_vx_i16m1(v843, v849, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v855 = __riscv_vwadd_wv_i32m2(v853, v854, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v856 = __riscv_vwmul_vx_i16m1(v846, v852, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v857 = __riscv_vwadd_wv_i32m2(v855, v856, 8);
      v736 = v857;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v858 = v19 + 432;
      const uint8_t* v859 = (const uint8_t*) v858;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v860 = __riscv_vle8_v_u8mf2(v859, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v861 = __riscv_vand_vx_u8mf2(v860, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v862 = __riscv_vzext_vf2_u16m1(v861, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v863 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v862, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v864 = __riscv_vsrl_vx_u8mf2(v860, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v865 = __riscv_vzext_vf2_u16m1(v864, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v866 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v865, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v867 = v21 + 39;
      const int8_t* v868 = (const int8_t*) v867;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v869 = *(const int8_t *)(v868);
      const uint8_t* v870 = v21 + 55;
      const int8_t* v871 = (const int8_t*) v870;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v872 = *(const int8_t *)(v871);
      vint32m2_t v873 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v874 = __riscv_vwmul_vx_i16m1(v863, v869, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v875 = __riscv_vwadd_wv_i32m2(v873, v874, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v876 = __riscv_vwmul_vx_i16m1(v866, v872, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v877 = __riscv_vwadd_wv_i32m2(v875, v876, 8);
      v734 = v877;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v878 = v19 + 440;
      const uint8_t* v879 = (const uint8_t*) v878;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v880 = __riscv_vle8_v_u8mf2(v879, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v881 = __riscv_vand_vx_u8mf2(v880, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v882 = __riscv_vzext_vf2_u16m1(v881, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v883 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v882, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v884 = __riscv_vsrl_vx_u8mf2(v880, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v885 = __riscv_vzext_vf2_u16m1(v884, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v886 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v885, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v887 = v21 + 39;
      const int8_t* v888 = (const int8_t*) v887;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v889 = *(const int8_t *)(v888);
      const uint8_t* v890 = v21 + 55;
      const int8_t* v891 = (const int8_t*) v890;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v892 = *(const int8_t *)(v891);
      vint32m2_t v893 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v894 = __riscv_vwmul_vx_i16m1(v883, v889, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v895 = __riscv_vwadd_wv_i32m2(v893, v894, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v896 = __riscv_vwmul_vx_i16m1(v886, v892, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v897 = __riscv_vwadd_wv_i32m2(v895, v896, 8);
      v736 = v897;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v898 = v19 + 448;
      const uint8_t* v899 = (const uint8_t*) v898;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v900 = __riscv_vle8_v_u8mf2(v899, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v901 = __riscv_vand_vx_u8mf2(v900, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v902 = __riscv_vzext_vf2_u16m1(v901, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v903 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v902, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v904 = __riscv_vsrl_vx_u8mf2(v900, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v905 = __riscv_vzext_vf2_u16m1(v904, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v906 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v905, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v907 = v21 + 40;
      const int8_t* v908 = (const int8_t*) v907;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v909 = *(const int8_t *)(v908);
      const uint8_t* v910 = v21 + 56;
      const int8_t* v911 = (const int8_t*) v910;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v912 = *(const int8_t *)(v911);
      vint32m2_t v913 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v914 = __riscv_vwmul_vx_i16m1(v903, v909, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v915 = __riscv_vwadd_wv_i32m2(v913, v914, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v916 = __riscv_vwmul_vx_i16m1(v906, v912, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v917 = __riscv_vwadd_wv_i32m2(v915, v916, 8);
      v734 = v917;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v918 = v19 + 456;
      const uint8_t* v919 = (const uint8_t*) v918;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v920 = __riscv_vle8_v_u8mf2(v919, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v921 = __riscv_vand_vx_u8mf2(v920, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v922 = __riscv_vzext_vf2_u16m1(v921, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v923 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v922, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v924 = __riscv_vsrl_vx_u8mf2(v920, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v925 = __riscv_vzext_vf2_u16m1(v924, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v926 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v925, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v927 = v21 + 40;
      const int8_t* v928 = (const int8_t*) v927;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v929 = *(const int8_t *)(v928);
      const uint8_t* v930 = v21 + 56;
      const int8_t* v931 = (const int8_t*) v930;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v932 = *(const int8_t *)(v931);
      vint32m2_t v933 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v934 = __riscv_vwmul_vx_i16m1(v923, v929, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v935 = __riscv_vwadd_wv_i32m2(v933, v934, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v936 = __riscv_vwmul_vx_i16m1(v926, v932, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v937 = __riscv_vwadd_wv_i32m2(v935, v936, 8);
      v736 = v937;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v938 = v19 + 464;
      const uint8_t* v939 = (const uint8_t*) v938;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v940 = __riscv_vle8_v_u8mf2(v939, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v941 = __riscv_vand_vx_u8mf2(v940, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v942 = __riscv_vzext_vf2_u16m1(v941, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v943 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v942, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v944 = __riscv_vsrl_vx_u8mf2(v940, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v945 = __riscv_vzext_vf2_u16m1(v944, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v946 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v945, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v947 = v21 + 41;
      const int8_t* v948 = (const int8_t*) v947;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v949 = *(const int8_t *)(v948);
      const uint8_t* v950 = v21 + 57;
      const int8_t* v951 = (const int8_t*) v950;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v952 = *(const int8_t *)(v951);
      vint32m2_t v953 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v954 = __riscv_vwmul_vx_i16m1(v943, v949, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v955 = __riscv_vwadd_wv_i32m2(v953, v954, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v956 = __riscv_vwmul_vx_i16m1(v946, v952, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v957 = __riscv_vwadd_wv_i32m2(v955, v956, 8);
      v734 = v957;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v958 = v19 + 472;
      const uint8_t* v959 = (const uint8_t*) v958;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v960 = __riscv_vle8_v_u8mf2(v959, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v961 = __riscv_vand_vx_u8mf2(v960, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v962 = __riscv_vzext_vf2_u16m1(v961, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v963 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v962, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v964 = __riscv_vsrl_vx_u8mf2(v960, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v965 = __riscv_vzext_vf2_u16m1(v964, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v966 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v965, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v967 = v21 + 41;
      const int8_t* v968 = (const int8_t*) v967;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v969 = *(const int8_t *)(v968);
      const uint8_t* v970 = v21 + 57;
      const int8_t* v971 = (const int8_t*) v970;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v972 = *(const int8_t *)(v971);
      vint32m2_t v973 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v974 = __riscv_vwmul_vx_i16m1(v963, v969, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v975 = __riscv_vwadd_wv_i32m2(v973, v974, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v976 = __riscv_vwmul_vx_i16m1(v966, v972, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v977 = __riscv_vwadd_wv_i32m2(v975, v976, 8);
      v736 = v977;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v978 = v19 + 480;
      const uint8_t* v979 = (const uint8_t*) v978;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v980 = __riscv_vle8_v_u8mf2(v979, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v981 = __riscv_vand_vx_u8mf2(v980, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v982 = __riscv_vzext_vf2_u16m1(v981, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v983 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v982, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v984 = __riscv_vsrl_vx_u8mf2(v980, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v985 = __riscv_vzext_vf2_u16m1(v984, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v986 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v985, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v987 = v21 + 42;
      const int8_t* v988 = (const int8_t*) v987;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v989 = *(const int8_t *)(v988);
      const uint8_t* v990 = v21 + 58;
      const int8_t* v991 = (const int8_t*) v990;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v992 = *(const int8_t *)(v991);
      vint32m2_t v993 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v994 = __riscv_vwmul_vx_i16m1(v983, v989, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v995 = __riscv_vwadd_wv_i32m2(v993, v994, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v996 = __riscv_vwmul_vx_i16m1(v986, v992, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v997 = __riscv_vwadd_wv_i32m2(v995, v996, 8);
      v734 = v997;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v998 = v19 + 488;
      const uint8_t* v999 = (const uint8_t*) v998;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1000 = __riscv_vle8_v_u8mf2(v999, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1001 = __riscv_vand_vx_u8mf2(v1000, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1002 = __riscv_vzext_vf2_u16m1(v1001, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1003 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1002, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1004 = __riscv_vsrl_vx_u8mf2(v1000, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1005 = __riscv_vzext_vf2_u16m1(v1004, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1006 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1005, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1007 = v21 + 42;
      const int8_t* v1008 = (const int8_t*) v1007;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1009 = *(const int8_t *)(v1008);
      const uint8_t* v1010 = v21 + 58;
      const int8_t* v1011 = (const int8_t*) v1010;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1012 = *(const int8_t *)(v1011);
      vint32m2_t v1013 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1014 = __riscv_vwmul_vx_i16m1(v1003, v1009, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1015 = __riscv_vwadd_wv_i32m2(v1013, v1014, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1016 = __riscv_vwmul_vx_i16m1(v1006, v1012, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1017 = __riscv_vwadd_wv_i32m2(v1015, v1016, 8);
      v736 = v1017;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1018 = v19 + 496;
      const uint8_t* v1019 = (const uint8_t*) v1018;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1020 = __riscv_vle8_v_u8mf2(v1019, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1021 = __riscv_vand_vx_u8mf2(v1020, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1022 = __riscv_vzext_vf2_u16m1(v1021, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1023 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1022, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1024 = __riscv_vsrl_vx_u8mf2(v1020, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1025 = __riscv_vzext_vf2_u16m1(v1024, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1026 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1025, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1027 = v21 + 43;
      const int8_t* v1028 = (const int8_t*) v1027;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1029 = *(const int8_t *)(v1028);
      const uint8_t* v1030 = v21 + 59;
      const int8_t* v1031 = (const int8_t*) v1030;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1032 = *(const int8_t *)(v1031);
      vint32m2_t v1033 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1034 = __riscv_vwmul_vx_i16m1(v1023, v1029, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1035 = __riscv_vwadd_wv_i32m2(v1033, v1034, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1036 = __riscv_vwmul_vx_i16m1(v1026, v1032, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1037 = __riscv_vwadd_wv_i32m2(v1035, v1036, 8);
      v734 = v1037;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1038 = v19 + 504;
      const uint8_t* v1039 = (const uint8_t*) v1038;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1040 = __riscv_vle8_v_u8mf2(v1039, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1041 = __riscv_vand_vx_u8mf2(v1040, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1042 = __riscv_vzext_vf2_u16m1(v1041, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1043 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1042, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1044 = __riscv_vsrl_vx_u8mf2(v1040, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1045 = __riscv_vzext_vf2_u16m1(v1044, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1046 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1045, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1047 = v21 + 43;
      const int8_t* v1048 = (const int8_t*) v1047;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1049 = *(const int8_t *)(v1048);
      const uint8_t* v1050 = v21 + 59;
      const int8_t* v1051 = (const int8_t*) v1050;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1052 = *(const int8_t *)(v1051);
      vint32m2_t v1053 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1054 = __riscv_vwmul_vx_i16m1(v1043, v1049, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1055 = __riscv_vwadd_wv_i32m2(v1053, v1054, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1056 = __riscv_vwmul_vx_i16m1(v1046, v1052, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1057 = __riscv_vwadd_wv_i32m2(v1055, v1056, 8);
      v736 = v1057;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1058 = v19 + 512;
      const uint8_t* v1059 = (const uint8_t*) v1058;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1060 = __riscv_vle8_v_u8mf2(v1059, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1061 = __riscv_vand_vx_u8mf2(v1060, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1062 = __riscv_vzext_vf2_u16m1(v1061, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1063 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1062, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1064 = __riscv_vsrl_vx_u8mf2(v1060, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1065 = __riscv_vzext_vf2_u16m1(v1064, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1066 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1065, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1067 = v21 + 44;
      const int8_t* v1068 = (const int8_t*) v1067;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1069 = *(const int8_t *)(v1068);
      const uint8_t* v1070 = v21 + 60;
      const int8_t* v1071 = (const int8_t*) v1070;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1072 = *(const int8_t *)(v1071);
      vint32m2_t v1073 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1074 = __riscv_vwmul_vx_i16m1(v1063, v1069, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1075 = __riscv_vwadd_wv_i32m2(v1073, v1074, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1076 = __riscv_vwmul_vx_i16m1(v1066, v1072, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1077 = __riscv_vwadd_wv_i32m2(v1075, v1076, 8);
      v734 = v1077;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1078 = v19 + 520;
      const uint8_t* v1079 = (const uint8_t*) v1078;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1080 = __riscv_vle8_v_u8mf2(v1079, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1081 = __riscv_vand_vx_u8mf2(v1080, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1082 = __riscv_vzext_vf2_u16m1(v1081, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1083 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1082, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1084 = __riscv_vsrl_vx_u8mf2(v1080, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1085 = __riscv_vzext_vf2_u16m1(v1084, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1086 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1085, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1087 = v21 + 44;
      const int8_t* v1088 = (const int8_t*) v1087;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1089 = *(const int8_t *)(v1088);
      const uint8_t* v1090 = v21 + 60;
      const int8_t* v1091 = (const int8_t*) v1090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1092 = *(const int8_t *)(v1091);
      vint32m2_t v1093 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1094 = __riscv_vwmul_vx_i16m1(v1083, v1089, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1095 = __riscv_vwadd_wv_i32m2(v1093, v1094, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1096 = __riscv_vwmul_vx_i16m1(v1086, v1092, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1097 = __riscv_vwadd_wv_i32m2(v1095, v1096, 8);
      v736 = v1097;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1098 = v19 + 528;
      const uint8_t* v1099 = (const uint8_t*) v1098;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1100 = __riscv_vle8_v_u8mf2(v1099, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1101 = __riscv_vand_vx_u8mf2(v1100, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1102 = __riscv_vzext_vf2_u16m1(v1101, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1103 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1102, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1104 = __riscv_vsrl_vx_u8mf2(v1100, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1105 = __riscv_vzext_vf2_u16m1(v1104, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1106 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1105, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1107 = v21 + 45;
      const int8_t* v1108 = (const int8_t*) v1107;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1109 = *(const int8_t *)(v1108);
      const uint8_t* v1110 = v21 + 61;
      const int8_t* v1111 = (const int8_t*) v1110;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1112 = *(const int8_t *)(v1111);
      vint32m2_t v1113 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1114 = __riscv_vwmul_vx_i16m1(v1103, v1109, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1115 = __riscv_vwadd_wv_i32m2(v1113, v1114, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1116 = __riscv_vwmul_vx_i16m1(v1106, v1112, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1117 = __riscv_vwadd_wv_i32m2(v1115, v1116, 8);
      v734 = v1117;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1118 = v19 + 536;
      const uint8_t* v1119 = (const uint8_t*) v1118;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1120 = __riscv_vle8_v_u8mf2(v1119, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1121 = __riscv_vand_vx_u8mf2(v1120, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1122 = __riscv_vzext_vf2_u16m1(v1121, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1123 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1122, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1124 = __riscv_vsrl_vx_u8mf2(v1120, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1125 = __riscv_vzext_vf2_u16m1(v1124, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1126 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1125, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1127 = v21 + 45;
      const int8_t* v1128 = (const int8_t*) v1127;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1129 = *(const int8_t *)(v1128);
      const uint8_t* v1130 = v21 + 61;
      const int8_t* v1131 = (const int8_t*) v1130;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1132 = *(const int8_t *)(v1131);
      vint32m2_t v1133 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1134 = __riscv_vwmul_vx_i16m1(v1123, v1129, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1135 = __riscv_vwadd_wv_i32m2(v1133, v1134, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1136 = __riscv_vwmul_vx_i16m1(v1126, v1132, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1137 = __riscv_vwadd_wv_i32m2(v1135, v1136, 8);
      v736 = v1137;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1138 = v19 + 544;
      const uint8_t* v1139 = (const uint8_t*) v1138;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1140 = __riscv_vle8_v_u8mf2(v1139, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1141 = __riscv_vand_vx_u8mf2(v1140, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1142 = __riscv_vzext_vf2_u16m1(v1141, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1143 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1142, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1144 = __riscv_vsrl_vx_u8mf2(v1140, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1145 = __riscv_vzext_vf2_u16m1(v1144, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1146 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1145, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1147 = v21 + 46;
      const int8_t* v1148 = (const int8_t*) v1147;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1149 = *(const int8_t *)(v1148);
      const uint8_t* v1150 = v21 + 62;
      const int8_t* v1151 = (const int8_t*) v1150;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1152 = *(const int8_t *)(v1151);
      vint32m2_t v1153 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1154 = __riscv_vwmul_vx_i16m1(v1143, v1149, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1155 = __riscv_vwadd_wv_i32m2(v1153, v1154, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1156 = __riscv_vwmul_vx_i16m1(v1146, v1152, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1157 = __riscv_vwadd_wv_i32m2(v1155, v1156, 8);
      v734 = v1157;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1158 = v19 + 552;
      const uint8_t* v1159 = (const uint8_t*) v1158;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1160 = __riscv_vle8_v_u8mf2(v1159, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1161 = __riscv_vand_vx_u8mf2(v1160, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1162 = __riscv_vzext_vf2_u16m1(v1161, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1163 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1162, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1164 = __riscv_vsrl_vx_u8mf2(v1160, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1165 = __riscv_vzext_vf2_u16m1(v1164, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1166 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1165, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1167 = v21 + 46;
      const int8_t* v1168 = (const int8_t*) v1167;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1169 = *(const int8_t *)(v1168);
      const uint8_t* v1170 = v21 + 62;
      const int8_t* v1171 = (const int8_t*) v1170;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1172 = *(const int8_t *)(v1171);
      vint32m2_t v1173 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1174 = __riscv_vwmul_vx_i16m1(v1163, v1169, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1175 = __riscv_vwadd_wv_i32m2(v1173, v1174, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1176 = __riscv_vwmul_vx_i16m1(v1166, v1172, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1177 = __riscv_vwadd_wv_i32m2(v1175, v1176, 8);
      v736 = v1177;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1178 = v19 + 560;
      const uint8_t* v1179 = (const uint8_t*) v1178;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1180 = __riscv_vle8_v_u8mf2(v1179, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1181 = __riscv_vand_vx_u8mf2(v1180, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1182 = __riscv_vzext_vf2_u16m1(v1181, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1183 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1182, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1184 = __riscv_vsrl_vx_u8mf2(v1180, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1185 = __riscv_vzext_vf2_u16m1(v1184, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1186 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1185, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1187 = v21 + 47;
      const int8_t* v1188 = (const int8_t*) v1187;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1189 = *(const int8_t *)(v1188);
      const uint8_t* v1190 = v21 + 63;
      const int8_t* v1191 = (const int8_t*) v1190;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1192 = *(const int8_t *)(v1191);
      vint32m2_t v1193 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1194 = __riscv_vwmul_vx_i16m1(v1183, v1189, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1195 = __riscv_vwadd_wv_i32m2(v1193, v1194, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1196 = __riscv_vwmul_vx_i16m1(v1186, v1192, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1197 = __riscv_vwadd_wv_i32m2(v1195, v1196, 8);
      v734 = v1197;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1198 = v19 + 568;
      const uint8_t* v1199 = (const uint8_t*) v1198;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1200 = __riscv_vle8_v_u8mf2(v1199, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1201 = __riscv_vand_vx_u8mf2(v1200, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1202 = __riscv_vzext_vf2_u16m1(v1201, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1203 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1202, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1204 = __riscv_vsrl_vx_u8mf2(v1200, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1205 = __riscv_vzext_vf2_u16m1(v1204, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1206 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1205, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1207 = v21 + 47;
      const int8_t* v1208 = (const int8_t*) v1207;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1209 = *(const int8_t *)(v1208);
      const uint8_t* v1210 = v21 + 63;
      const int8_t* v1211 = (const int8_t*) v1210;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1212 = *(const int8_t *)(v1211);
      vint32m2_t v1213 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1214 = __riscv_vwmul_vx_i16m1(v1203, v1209, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1215 = __riscv_vwadd_wv_i32m2(v1213, v1214, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1216 = __riscv_vwmul_vx_i16m1(v1206, v1212, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1217 = __riscv_vwadd_wv_i32m2(v1215, v1216, 8);
      v736 = v1217;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1218 = v19 + 576;
      const uint8_t* v1219 = (const uint8_t*) v1218;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1220 = __riscv_vle8_v_u8mf2(v1219, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1221 = __riscv_vand_vx_u8mf2(v1220, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1222 = __riscv_vzext_vf2_u16m1(v1221, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1223 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1222, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1224 = __riscv_vsrl_vx_u8mf2(v1220, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1225 = __riscv_vzext_vf2_u16m1(v1224, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1226 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1225, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1227 = v21 + 48;
      const int8_t* v1228 = (const int8_t*) v1227;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1229 = *(const int8_t *)(v1228);
      const uint8_t* v1230 = v21 + 64;
      const int8_t* v1231 = (const int8_t*) v1230;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1232 = *(const int8_t *)(v1231);
      vint32m2_t v1233 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1234 = __riscv_vwmul_vx_i16m1(v1223, v1229, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1235 = __riscv_vwadd_wv_i32m2(v1233, v1234, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1236 = __riscv_vwmul_vx_i16m1(v1226, v1232, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1237 = __riscv_vwadd_wv_i32m2(v1235, v1236, 8);
      v734 = v1237;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1238 = v19 + 584;
      const uint8_t* v1239 = (const uint8_t*) v1238;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1240 = __riscv_vle8_v_u8mf2(v1239, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1241 = __riscv_vand_vx_u8mf2(v1240, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1242 = __riscv_vzext_vf2_u16m1(v1241, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1243 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1242, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1244 = __riscv_vsrl_vx_u8mf2(v1240, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1245 = __riscv_vzext_vf2_u16m1(v1244, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1246 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1245, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1247 = v21 + 48;
      const int8_t* v1248 = (const int8_t*) v1247;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1249 = *(const int8_t *)(v1248);
      const uint8_t* v1250 = v21 + 64;
      const int8_t* v1251 = (const int8_t*) v1250;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1252 = *(const int8_t *)(v1251);
      vint32m2_t v1253 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1254 = __riscv_vwmul_vx_i16m1(v1243, v1249, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1255 = __riscv_vwadd_wv_i32m2(v1253, v1254, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1256 = __riscv_vwmul_vx_i16m1(v1246, v1252, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1257 = __riscv_vwadd_wv_i32m2(v1255, v1256, 8);
      v736 = v1257;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1258 = v19 + 592;
      const uint8_t* v1259 = (const uint8_t*) v1258;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1260 = __riscv_vle8_v_u8mf2(v1259, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1261 = __riscv_vand_vx_u8mf2(v1260, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1262 = __riscv_vzext_vf2_u16m1(v1261, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1263 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1262, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1264 = __riscv_vsrl_vx_u8mf2(v1260, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1265 = __riscv_vzext_vf2_u16m1(v1264, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1266 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1265, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1267 = v21 + 49;
      const int8_t* v1268 = (const int8_t*) v1267;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1269 = *(const int8_t *)(v1268);
      const uint8_t* v1270 = v21 + 65;
      const int8_t* v1271 = (const int8_t*) v1270;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1272 = *(const int8_t *)(v1271);
      vint32m2_t v1273 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1274 = __riscv_vwmul_vx_i16m1(v1263, v1269, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1275 = __riscv_vwadd_wv_i32m2(v1273, v1274, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1276 = __riscv_vwmul_vx_i16m1(v1266, v1272, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1277 = __riscv_vwadd_wv_i32m2(v1275, v1276, 8);
      v734 = v1277;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1278 = v19 + 600;
      const uint8_t* v1279 = (const uint8_t*) v1278;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1280 = __riscv_vle8_v_u8mf2(v1279, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1281 = __riscv_vand_vx_u8mf2(v1280, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1282 = __riscv_vzext_vf2_u16m1(v1281, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1283 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1282, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1284 = __riscv_vsrl_vx_u8mf2(v1280, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1285 = __riscv_vzext_vf2_u16m1(v1284, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1286 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1285, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1287 = v21 + 49;
      const int8_t* v1288 = (const int8_t*) v1287;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1289 = *(const int8_t *)(v1288);
      const uint8_t* v1290 = v21 + 65;
      const int8_t* v1291 = (const int8_t*) v1290;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1292 = *(const int8_t *)(v1291);
      vint32m2_t v1293 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1294 = __riscv_vwmul_vx_i16m1(v1283, v1289, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1295 = __riscv_vwadd_wv_i32m2(v1293, v1294, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1296 = __riscv_vwmul_vx_i16m1(v1286, v1292, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1297 = __riscv_vwadd_wv_i32m2(v1295, v1296, 8);
      v736 = v1297;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1298 = v19 + 608;
      const uint8_t* v1299 = (const uint8_t*) v1298;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1300 = __riscv_vle8_v_u8mf2(v1299, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1301 = __riscv_vand_vx_u8mf2(v1300, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1302 = __riscv_vzext_vf2_u16m1(v1301, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1303 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1302, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1304 = __riscv_vsrl_vx_u8mf2(v1300, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1305 = __riscv_vzext_vf2_u16m1(v1304, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1306 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1305, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1307 = v21 + 50;
      const int8_t* v1308 = (const int8_t*) v1307;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1309 = *(const int8_t *)(v1308);
      const uint8_t* v1310 = v21 + 66;
      const int8_t* v1311 = (const int8_t*) v1310;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1312 = *(const int8_t *)(v1311);
      vint32m2_t v1313 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1314 = __riscv_vwmul_vx_i16m1(v1303, v1309, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1315 = __riscv_vwadd_wv_i32m2(v1313, v1314, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1316 = __riscv_vwmul_vx_i16m1(v1306, v1312, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1317 = __riscv_vwadd_wv_i32m2(v1315, v1316, 8);
      v734 = v1317;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1318 = v19 + 616;
      const uint8_t* v1319 = (const uint8_t*) v1318;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1320 = __riscv_vle8_v_u8mf2(v1319, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1321 = __riscv_vand_vx_u8mf2(v1320, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1322 = __riscv_vzext_vf2_u16m1(v1321, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1323 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1322, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1324 = __riscv_vsrl_vx_u8mf2(v1320, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1325 = __riscv_vzext_vf2_u16m1(v1324, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1326 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1325, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1327 = v21 + 50;
      const int8_t* v1328 = (const int8_t*) v1327;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1329 = *(const int8_t *)(v1328);
      const uint8_t* v1330 = v21 + 66;
      const int8_t* v1331 = (const int8_t*) v1330;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1332 = *(const int8_t *)(v1331);
      vint32m2_t v1333 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1334 = __riscv_vwmul_vx_i16m1(v1323, v1329, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1335 = __riscv_vwadd_wv_i32m2(v1333, v1334, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1336 = __riscv_vwmul_vx_i16m1(v1326, v1332, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1337 = __riscv_vwadd_wv_i32m2(v1335, v1336, 8);
      v736 = v1337;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1338 = v19 + 624;
      const uint8_t* v1339 = (const uint8_t*) v1338;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1340 = __riscv_vle8_v_u8mf2(v1339, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1341 = __riscv_vand_vx_u8mf2(v1340, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1342 = __riscv_vzext_vf2_u16m1(v1341, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1343 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1342, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1344 = __riscv_vsrl_vx_u8mf2(v1340, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1345 = __riscv_vzext_vf2_u16m1(v1344, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1346 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1345, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1347 = v21 + 51;
      const int8_t* v1348 = (const int8_t*) v1347;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1349 = *(const int8_t *)(v1348);
      const uint8_t* v1350 = v21 + 67;
      const int8_t* v1351 = (const int8_t*) v1350;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1352 = *(const int8_t *)(v1351);
      vint32m2_t v1353 = v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1354 = __riscv_vwmul_vx_i16m1(v1343, v1349, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1355 = __riscv_vwadd_wv_i32m2(v1353, v1354, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1356 = __riscv_vwmul_vx_i16m1(v1346, v1352, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1357 = __riscv_vwadd_wv_i32m2(v1355, v1356, 8);
      v734 = v1357;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1358 = v19 + 632;
      const uint8_t* v1359 = (const uint8_t*) v1358;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1360 = __riscv_vle8_v_u8mf2(v1359, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1361 = __riscv_vand_vx_u8mf2(v1360, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1362 = __riscv_vzext_vf2_u16m1(v1361, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1363 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1362, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1364 = __riscv_vsrl_vx_u8mf2(v1360, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1365 = __riscv_vzext_vf2_u16m1(v1364, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1366 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1365, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1367 = v21 + 51;
      const int8_t* v1368 = (const int8_t*) v1367;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1369 = *(const int8_t *)(v1368);
      const uint8_t* v1370 = v21 + 67;
      const int8_t* v1371 = (const int8_t*) v1370;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1372 = *(const int8_t *)(v1371);
      vint32m2_t v1373 = v736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1374 = __riscv_vwmul_vx_i16m1(v1363, v1369, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1375 = __riscv_vwadd_wv_i32m2(v1373, v1374, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1376 = __riscv_vwmul_vx_i16m1(v1366, v1372, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1377 = __riscv_vwadd_wv_i32m2(v1375, v1376, 8);
      v736 = v1377;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_scale_fold
      vint32m2_t v1378 = v734;
      vint32m2_t v1379 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v1380 = __riscv_vmacc_vv_i32m2(v1379, v719, v1378, 8);
      v24 = v1380;
      vint32m2_t v1381 = v736;
      vint32m2_t v1382 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v1383 = __riscv_vmacc_vv_i32m2(v1382, v733, v1381, 8);
      v26 = v1383;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_signed_scale
      const uint8_t* v1384 = v19 + 80;
      const uint8_t* v1385 = (const uint8_t*) v1384;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1386 = __riscv_vle8_v_u8mf2(v1385, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1387 = __riscv_vand_vx_u8mf2(v1386, 0x0F, 8);
      const uint8_t* v1388 = v19 + 32;
      const uint8_t* v1389 = (const uint8_t*) v1388;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1390 = __riscv_vle8_v_u8mf2(v1389, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1391 = __riscv_vsrl_vx_u8mf2(v1390, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1392 = __riscv_vand_vx_u8mf2(v1391, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v1393 = __riscv_vsll_vx_u8mf2(v1392, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v1394 = __riscv_vor_vv_u8mf2(v1393, v1387, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1395 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1394);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v1396 = __riscv_vsub_vx_i8mf2(v1395, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v1397 = __riscv_vsext_vf4_i32m2(v1396, 8);
      const uint8_t* v1398 = v19 + 88;
      const uint8_t* v1399 = (const uint8_t*) v1398;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1400 = __riscv_vle8_v_u8mf2(v1399, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1401 = __riscv_vand_vx_u8mf2(v1400, 0x0F, 8);
      const uint8_t* v1402 = v19 + 40;
      const uint8_t* v1403 = (const uint8_t*) v1402;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1404 = __riscv_vle8_v_u8mf2(v1403, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1405 = __riscv_vsrl_vx_u8mf2(v1404, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1406 = __riscv_vand_vx_u8mf2(v1405, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v1407 = __riscv_vsll_vx_u8mf2(v1406, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v1408 = __riscv_vor_vv_u8mf2(v1407, v1401, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1409 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1408);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v1410 = __riscv_vsub_vx_i8mf2(v1409, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v1411 = __riscv_vsext_vf4_i32m2(v1410, 8);
      vint32m2_t v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v1413 = __riscv_vmv_v_x_i32m2(0, 8);
      v1412 = v1413;
      vint32m2_t v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v1415 = __riscv_vmv_v_x_i32m2(0, 8);
      v1414 = v1415;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1416 = v19 + 640;
      const uint8_t* v1417 = (const uint8_t*) v1416;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1418 = __riscv_vle8_v_u8mf2(v1417, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1419 = __riscv_vand_vx_u8mf2(v1418, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1420 = __riscv_vzext_vf2_u16m1(v1419, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1421 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1420, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1422 = __riscv_vsrl_vx_u8mf2(v1418, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1423 = __riscv_vzext_vf2_u16m1(v1422, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1424 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1423, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1425 = v21 + 68;
      const int8_t* v1426 = (const int8_t*) v1425;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1427 = *(const int8_t *)(v1426);
      const uint8_t* v1428 = v21 + 84;
      const int8_t* v1429 = (const int8_t*) v1428;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1430 = *(const int8_t *)(v1429);
      vint32m2_t v1431 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1432 = __riscv_vwmul_vx_i16m1(v1421, v1427, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1433 = __riscv_vwadd_wv_i32m2(v1431, v1432, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1434 = __riscv_vwmul_vx_i16m1(v1424, v1430, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1435 = __riscv_vwadd_wv_i32m2(v1433, v1434, 8);
      v1412 = v1435;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1436 = v19 + 648;
      const uint8_t* v1437 = (const uint8_t*) v1436;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1438 = __riscv_vle8_v_u8mf2(v1437, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1439 = __riscv_vand_vx_u8mf2(v1438, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1440 = __riscv_vzext_vf2_u16m1(v1439, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1441 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1440, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1442 = __riscv_vsrl_vx_u8mf2(v1438, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1443 = __riscv_vzext_vf2_u16m1(v1442, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1444 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1443, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1445 = v21 + 68;
      const int8_t* v1446 = (const int8_t*) v1445;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1447 = *(const int8_t *)(v1446);
      const uint8_t* v1448 = v21 + 84;
      const int8_t* v1449 = (const int8_t*) v1448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1450 = *(const int8_t *)(v1449);
      vint32m2_t v1451 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1452 = __riscv_vwmul_vx_i16m1(v1441, v1447, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1453 = __riscv_vwadd_wv_i32m2(v1451, v1452, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1454 = __riscv_vwmul_vx_i16m1(v1444, v1450, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1455 = __riscv_vwadd_wv_i32m2(v1453, v1454, 8);
      v1414 = v1455;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1456 = v19 + 656;
      const uint8_t* v1457 = (const uint8_t*) v1456;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1458 = __riscv_vle8_v_u8mf2(v1457, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1459 = __riscv_vand_vx_u8mf2(v1458, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1460 = __riscv_vzext_vf2_u16m1(v1459, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1461 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1460, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1462 = __riscv_vsrl_vx_u8mf2(v1458, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1463 = __riscv_vzext_vf2_u16m1(v1462, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1464 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1463, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1465 = v21 + 69;
      const int8_t* v1466 = (const int8_t*) v1465;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1467 = *(const int8_t *)(v1466);
      const uint8_t* v1468 = v21 + 85;
      const int8_t* v1469 = (const int8_t*) v1468;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1470 = *(const int8_t *)(v1469);
      vint32m2_t v1471 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1472 = __riscv_vwmul_vx_i16m1(v1461, v1467, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1473 = __riscv_vwadd_wv_i32m2(v1471, v1472, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1474 = __riscv_vwmul_vx_i16m1(v1464, v1470, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1475 = __riscv_vwadd_wv_i32m2(v1473, v1474, 8);
      v1412 = v1475;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1476 = v19 + 664;
      const uint8_t* v1477 = (const uint8_t*) v1476;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1478 = __riscv_vle8_v_u8mf2(v1477, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1479 = __riscv_vand_vx_u8mf2(v1478, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1480 = __riscv_vzext_vf2_u16m1(v1479, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1481 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1480, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1482 = __riscv_vsrl_vx_u8mf2(v1478, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1483 = __riscv_vzext_vf2_u16m1(v1482, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1484 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1483, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1485 = v21 + 69;
      const int8_t* v1486 = (const int8_t*) v1485;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1487 = *(const int8_t *)(v1486);
      const uint8_t* v1488 = v21 + 85;
      const int8_t* v1489 = (const int8_t*) v1488;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1490 = *(const int8_t *)(v1489);
      vint32m2_t v1491 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1492 = __riscv_vwmul_vx_i16m1(v1481, v1487, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1493 = __riscv_vwadd_wv_i32m2(v1491, v1492, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1494 = __riscv_vwmul_vx_i16m1(v1484, v1490, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1495 = __riscv_vwadd_wv_i32m2(v1493, v1494, 8);
      v1414 = v1495;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1496 = v19 + 672;
      const uint8_t* v1497 = (const uint8_t*) v1496;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1498 = __riscv_vle8_v_u8mf2(v1497, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1499 = __riscv_vand_vx_u8mf2(v1498, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1500 = __riscv_vzext_vf2_u16m1(v1499, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1501 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1500, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1502 = __riscv_vsrl_vx_u8mf2(v1498, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1503 = __riscv_vzext_vf2_u16m1(v1502, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1504 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1503, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1505 = v21 + 70;
      const int8_t* v1506 = (const int8_t*) v1505;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1507 = *(const int8_t *)(v1506);
      const uint8_t* v1508 = v21 + 86;
      const int8_t* v1509 = (const int8_t*) v1508;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1510 = *(const int8_t *)(v1509);
      vint32m2_t v1511 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1512 = __riscv_vwmul_vx_i16m1(v1501, v1507, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1513 = __riscv_vwadd_wv_i32m2(v1511, v1512, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1514 = __riscv_vwmul_vx_i16m1(v1504, v1510, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1515 = __riscv_vwadd_wv_i32m2(v1513, v1514, 8);
      v1412 = v1515;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1516 = v19 + 680;
      const uint8_t* v1517 = (const uint8_t*) v1516;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1518 = __riscv_vle8_v_u8mf2(v1517, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1519 = __riscv_vand_vx_u8mf2(v1518, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1520 = __riscv_vzext_vf2_u16m1(v1519, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1521 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1520, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1522 = __riscv_vsrl_vx_u8mf2(v1518, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1523 = __riscv_vzext_vf2_u16m1(v1522, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1524 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1523, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1525 = v21 + 70;
      const int8_t* v1526 = (const int8_t*) v1525;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1527 = *(const int8_t *)(v1526);
      const uint8_t* v1528 = v21 + 86;
      const int8_t* v1529 = (const int8_t*) v1528;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1530 = *(const int8_t *)(v1529);
      vint32m2_t v1531 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1532 = __riscv_vwmul_vx_i16m1(v1521, v1527, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1533 = __riscv_vwadd_wv_i32m2(v1531, v1532, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1534 = __riscv_vwmul_vx_i16m1(v1524, v1530, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1535 = __riscv_vwadd_wv_i32m2(v1533, v1534, 8);
      v1414 = v1535;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1536 = v19 + 688;
      const uint8_t* v1537 = (const uint8_t*) v1536;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1538 = __riscv_vle8_v_u8mf2(v1537, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1539 = __riscv_vand_vx_u8mf2(v1538, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1540 = __riscv_vzext_vf2_u16m1(v1539, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1541 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1540, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1542 = __riscv_vsrl_vx_u8mf2(v1538, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1543 = __riscv_vzext_vf2_u16m1(v1542, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1544 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1543, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1545 = v21 + 71;
      const int8_t* v1546 = (const int8_t*) v1545;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1547 = *(const int8_t *)(v1546);
      const uint8_t* v1548 = v21 + 87;
      const int8_t* v1549 = (const int8_t*) v1548;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1550 = *(const int8_t *)(v1549);
      vint32m2_t v1551 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1552 = __riscv_vwmul_vx_i16m1(v1541, v1547, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1553 = __riscv_vwadd_wv_i32m2(v1551, v1552, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1554 = __riscv_vwmul_vx_i16m1(v1544, v1550, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1555 = __riscv_vwadd_wv_i32m2(v1553, v1554, 8);
      v1412 = v1555;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1556 = v19 + 696;
      const uint8_t* v1557 = (const uint8_t*) v1556;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1558 = __riscv_vle8_v_u8mf2(v1557, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1559 = __riscv_vand_vx_u8mf2(v1558, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1560 = __riscv_vzext_vf2_u16m1(v1559, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1561 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1560, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1562 = __riscv_vsrl_vx_u8mf2(v1558, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1563 = __riscv_vzext_vf2_u16m1(v1562, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1564 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1563, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1565 = v21 + 71;
      const int8_t* v1566 = (const int8_t*) v1565;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1567 = *(const int8_t *)(v1566);
      const uint8_t* v1568 = v21 + 87;
      const int8_t* v1569 = (const int8_t*) v1568;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1570 = *(const int8_t *)(v1569);
      vint32m2_t v1571 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1572 = __riscv_vwmul_vx_i16m1(v1561, v1567, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1573 = __riscv_vwadd_wv_i32m2(v1571, v1572, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1574 = __riscv_vwmul_vx_i16m1(v1564, v1570, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1575 = __riscv_vwadd_wv_i32m2(v1573, v1574, 8);
      v1414 = v1575;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1576 = v19 + 704;
      const uint8_t* v1577 = (const uint8_t*) v1576;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1578 = __riscv_vle8_v_u8mf2(v1577, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1579 = __riscv_vand_vx_u8mf2(v1578, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1580 = __riscv_vzext_vf2_u16m1(v1579, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1581 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1580, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1582 = __riscv_vsrl_vx_u8mf2(v1578, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1583 = __riscv_vzext_vf2_u16m1(v1582, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1584 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1583, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1585 = v21 + 72;
      const int8_t* v1586 = (const int8_t*) v1585;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1587 = *(const int8_t *)(v1586);
      const uint8_t* v1588 = v21 + 88;
      const int8_t* v1589 = (const int8_t*) v1588;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1590 = *(const int8_t *)(v1589);
      vint32m2_t v1591 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1592 = __riscv_vwmul_vx_i16m1(v1581, v1587, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1593 = __riscv_vwadd_wv_i32m2(v1591, v1592, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1594 = __riscv_vwmul_vx_i16m1(v1584, v1590, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1595 = __riscv_vwadd_wv_i32m2(v1593, v1594, 8);
      v1412 = v1595;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1596 = v19 + 712;
      const uint8_t* v1597 = (const uint8_t*) v1596;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1598 = __riscv_vle8_v_u8mf2(v1597, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1599 = __riscv_vand_vx_u8mf2(v1598, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1600 = __riscv_vzext_vf2_u16m1(v1599, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1601 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1600, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1602 = __riscv_vsrl_vx_u8mf2(v1598, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1603 = __riscv_vzext_vf2_u16m1(v1602, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1604 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1603, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1605 = v21 + 72;
      const int8_t* v1606 = (const int8_t*) v1605;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1607 = *(const int8_t *)(v1606);
      const uint8_t* v1608 = v21 + 88;
      const int8_t* v1609 = (const int8_t*) v1608;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1610 = *(const int8_t *)(v1609);
      vint32m2_t v1611 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1612 = __riscv_vwmul_vx_i16m1(v1601, v1607, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1613 = __riscv_vwadd_wv_i32m2(v1611, v1612, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1614 = __riscv_vwmul_vx_i16m1(v1604, v1610, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1615 = __riscv_vwadd_wv_i32m2(v1613, v1614, 8);
      v1414 = v1615;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1616 = v19 + 720;
      const uint8_t* v1617 = (const uint8_t*) v1616;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1618 = __riscv_vle8_v_u8mf2(v1617, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1619 = __riscv_vand_vx_u8mf2(v1618, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1620 = __riscv_vzext_vf2_u16m1(v1619, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1621 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1620, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1622 = __riscv_vsrl_vx_u8mf2(v1618, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1623 = __riscv_vzext_vf2_u16m1(v1622, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1624 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1623, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1625 = v21 + 73;
      const int8_t* v1626 = (const int8_t*) v1625;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1627 = *(const int8_t *)(v1626);
      const uint8_t* v1628 = v21 + 89;
      const int8_t* v1629 = (const int8_t*) v1628;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1630 = *(const int8_t *)(v1629);
      vint32m2_t v1631 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1632 = __riscv_vwmul_vx_i16m1(v1621, v1627, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1633 = __riscv_vwadd_wv_i32m2(v1631, v1632, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1634 = __riscv_vwmul_vx_i16m1(v1624, v1630, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1635 = __riscv_vwadd_wv_i32m2(v1633, v1634, 8);
      v1412 = v1635;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1636 = v19 + 728;
      const uint8_t* v1637 = (const uint8_t*) v1636;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1638 = __riscv_vle8_v_u8mf2(v1637, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1639 = __riscv_vand_vx_u8mf2(v1638, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1640 = __riscv_vzext_vf2_u16m1(v1639, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1641 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1640, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1642 = __riscv_vsrl_vx_u8mf2(v1638, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1643 = __riscv_vzext_vf2_u16m1(v1642, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1644 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1643, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1645 = v21 + 73;
      const int8_t* v1646 = (const int8_t*) v1645;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1647 = *(const int8_t *)(v1646);
      const uint8_t* v1648 = v21 + 89;
      const int8_t* v1649 = (const int8_t*) v1648;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1650 = *(const int8_t *)(v1649);
      vint32m2_t v1651 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1652 = __riscv_vwmul_vx_i16m1(v1641, v1647, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1653 = __riscv_vwadd_wv_i32m2(v1651, v1652, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1654 = __riscv_vwmul_vx_i16m1(v1644, v1650, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1655 = __riscv_vwadd_wv_i32m2(v1653, v1654, 8);
      v1414 = v1655;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1656 = v19 + 736;
      const uint8_t* v1657 = (const uint8_t*) v1656;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1658 = __riscv_vle8_v_u8mf2(v1657, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1659 = __riscv_vand_vx_u8mf2(v1658, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1660 = __riscv_vzext_vf2_u16m1(v1659, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1661 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1660, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1662 = __riscv_vsrl_vx_u8mf2(v1658, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1663 = __riscv_vzext_vf2_u16m1(v1662, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1664 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1663, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1665 = v21 + 74;
      const int8_t* v1666 = (const int8_t*) v1665;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1667 = *(const int8_t *)(v1666);
      const uint8_t* v1668 = v21 + 90;
      const int8_t* v1669 = (const int8_t*) v1668;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1670 = *(const int8_t *)(v1669);
      vint32m2_t v1671 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1672 = __riscv_vwmul_vx_i16m1(v1661, v1667, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1673 = __riscv_vwadd_wv_i32m2(v1671, v1672, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1674 = __riscv_vwmul_vx_i16m1(v1664, v1670, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1675 = __riscv_vwadd_wv_i32m2(v1673, v1674, 8);
      v1412 = v1675;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1676 = v19 + 744;
      const uint8_t* v1677 = (const uint8_t*) v1676;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1678 = __riscv_vle8_v_u8mf2(v1677, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1679 = __riscv_vand_vx_u8mf2(v1678, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1680 = __riscv_vzext_vf2_u16m1(v1679, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1681 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1680, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1682 = __riscv_vsrl_vx_u8mf2(v1678, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1683 = __riscv_vzext_vf2_u16m1(v1682, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1684 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1683, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1685 = v21 + 74;
      const int8_t* v1686 = (const int8_t*) v1685;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1687 = *(const int8_t *)(v1686);
      const uint8_t* v1688 = v21 + 90;
      const int8_t* v1689 = (const int8_t*) v1688;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1690 = *(const int8_t *)(v1689);
      vint32m2_t v1691 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1692 = __riscv_vwmul_vx_i16m1(v1681, v1687, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1693 = __riscv_vwadd_wv_i32m2(v1691, v1692, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1694 = __riscv_vwmul_vx_i16m1(v1684, v1690, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1695 = __riscv_vwadd_wv_i32m2(v1693, v1694, 8);
      v1414 = v1695;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1696 = v19 + 752;
      const uint8_t* v1697 = (const uint8_t*) v1696;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1698 = __riscv_vle8_v_u8mf2(v1697, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1699 = __riscv_vand_vx_u8mf2(v1698, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1700 = __riscv_vzext_vf2_u16m1(v1699, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1701 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1700, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1702 = __riscv_vsrl_vx_u8mf2(v1698, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1703 = __riscv_vzext_vf2_u16m1(v1702, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1704 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1703, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1705 = v21 + 75;
      const int8_t* v1706 = (const int8_t*) v1705;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1707 = *(const int8_t *)(v1706);
      const uint8_t* v1708 = v21 + 91;
      const int8_t* v1709 = (const int8_t*) v1708;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1710 = *(const int8_t *)(v1709);
      vint32m2_t v1711 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1712 = __riscv_vwmul_vx_i16m1(v1701, v1707, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1713 = __riscv_vwadd_wv_i32m2(v1711, v1712, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1714 = __riscv_vwmul_vx_i16m1(v1704, v1710, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1715 = __riscv_vwadd_wv_i32m2(v1713, v1714, 8);
      v1412 = v1715;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1716 = v19 + 760;
      const uint8_t* v1717 = (const uint8_t*) v1716;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1718 = __riscv_vle8_v_u8mf2(v1717, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1719 = __riscv_vand_vx_u8mf2(v1718, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1720 = __riscv_vzext_vf2_u16m1(v1719, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1721 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1720, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1722 = __riscv_vsrl_vx_u8mf2(v1718, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1723 = __riscv_vzext_vf2_u16m1(v1722, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1724 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1723, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1725 = v21 + 75;
      const int8_t* v1726 = (const int8_t*) v1725;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1727 = *(const int8_t *)(v1726);
      const uint8_t* v1728 = v21 + 91;
      const int8_t* v1729 = (const int8_t*) v1728;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1730 = *(const int8_t *)(v1729);
      vint32m2_t v1731 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1732 = __riscv_vwmul_vx_i16m1(v1721, v1727, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1733 = __riscv_vwadd_wv_i32m2(v1731, v1732, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1734 = __riscv_vwmul_vx_i16m1(v1724, v1730, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1735 = __riscv_vwadd_wv_i32m2(v1733, v1734, 8);
      v1414 = v1735;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1736 = v19 + 768;
      const uint8_t* v1737 = (const uint8_t*) v1736;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1738 = __riscv_vle8_v_u8mf2(v1737, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1739 = __riscv_vand_vx_u8mf2(v1738, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1740 = __riscv_vzext_vf2_u16m1(v1739, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1741 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1740, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1742 = __riscv_vsrl_vx_u8mf2(v1738, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1743 = __riscv_vzext_vf2_u16m1(v1742, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1744 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1743, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1745 = v21 + 76;
      const int8_t* v1746 = (const int8_t*) v1745;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1747 = *(const int8_t *)(v1746);
      const uint8_t* v1748 = v21 + 92;
      const int8_t* v1749 = (const int8_t*) v1748;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1750 = *(const int8_t *)(v1749);
      vint32m2_t v1751 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1752 = __riscv_vwmul_vx_i16m1(v1741, v1747, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1753 = __riscv_vwadd_wv_i32m2(v1751, v1752, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1754 = __riscv_vwmul_vx_i16m1(v1744, v1750, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1755 = __riscv_vwadd_wv_i32m2(v1753, v1754, 8);
      v1412 = v1755;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1756 = v19 + 776;
      const uint8_t* v1757 = (const uint8_t*) v1756;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1758 = __riscv_vle8_v_u8mf2(v1757, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1759 = __riscv_vand_vx_u8mf2(v1758, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1760 = __riscv_vzext_vf2_u16m1(v1759, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1761 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1760, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1762 = __riscv_vsrl_vx_u8mf2(v1758, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1763 = __riscv_vzext_vf2_u16m1(v1762, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1764 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1763, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1765 = v21 + 76;
      const int8_t* v1766 = (const int8_t*) v1765;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1767 = *(const int8_t *)(v1766);
      const uint8_t* v1768 = v21 + 92;
      const int8_t* v1769 = (const int8_t*) v1768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1770 = *(const int8_t *)(v1769);
      vint32m2_t v1771 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1772 = __riscv_vwmul_vx_i16m1(v1761, v1767, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1773 = __riscv_vwadd_wv_i32m2(v1771, v1772, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1774 = __riscv_vwmul_vx_i16m1(v1764, v1770, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1775 = __riscv_vwadd_wv_i32m2(v1773, v1774, 8);
      v1414 = v1775;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1776 = v19 + 784;
      const uint8_t* v1777 = (const uint8_t*) v1776;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1778 = __riscv_vle8_v_u8mf2(v1777, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1779 = __riscv_vand_vx_u8mf2(v1778, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1780 = __riscv_vzext_vf2_u16m1(v1779, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1781 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1780, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1782 = __riscv_vsrl_vx_u8mf2(v1778, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1783 = __riscv_vzext_vf2_u16m1(v1782, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1784 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1783, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1785 = v21 + 77;
      const int8_t* v1786 = (const int8_t*) v1785;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1787 = *(const int8_t *)(v1786);
      const uint8_t* v1788 = v21 + 93;
      const int8_t* v1789 = (const int8_t*) v1788;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1790 = *(const int8_t *)(v1789);
      vint32m2_t v1791 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1792 = __riscv_vwmul_vx_i16m1(v1781, v1787, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1793 = __riscv_vwadd_wv_i32m2(v1791, v1792, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1794 = __riscv_vwmul_vx_i16m1(v1784, v1790, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1795 = __riscv_vwadd_wv_i32m2(v1793, v1794, 8);
      v1412 = v1795;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1796 = v19 + 792;
      const uint8_t* v1797 = (const uint8_t*) v1796;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1798 = __riscv_vle8_v_u8mf2(v1797, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1799 = __riscv_vand_vx_u8mf2(v1798, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1800 = __riscv_vzext_vf2_u16m1(v1799, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1801 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1800, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1802 = __riscv_vsrl_vx_u8mf2(v1798, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1803 = __riscv_vzext_vf2_u16m1(v1802, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1804 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1803, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1805 = v21 + 77;
      const int8_t* v1806 = (const int8_t*) v1805;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1807 = *(const int8_t *)(v1806);
      const uint8_t* v1808 = v21 + 93;
      const int8_t* v1809 = (const int8_t*) v1808;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1810 = *(const int8_t *)(v1809);
      vint32m2_t v1811 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1812 = __riscv_vwmul_vx_i16m1(v1801, v1807, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1813 = __riscv_vwadd_wv_i32m2(v1811, v1812, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1814 = __riscv_vwmul_vx_i16m1(v1804, v1810, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1815 = __riscv_vwadd_wv_i32m2(v1813, v1814, 8);
      v1414 = v1815;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1816 = v19 + 800;
      const uint8_t* v1817 = (const uint8_t*) v1816;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1818 = __riscv_vle8_v_u8mf2(v1817, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1819 = __riscv_vand_vx_u8mf2(v1818, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1820 = __riscv_vzext_vf2_u16m1(v1819, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1821 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1820, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1822 = __riscv_vsrl_vx_u8mf2(v1818, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1823 = __riscv_vzext_vf2_u16m1(v1822, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1824 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1823, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1825 = v21 + 78;
      const int8_t* v1826 = (const int8_t*) v1825;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1827 = *(const int8_t *)(v1826);
      const uint8_t* v1828 = v21 + 94;
      const int8_t* v1829 = (const int8_t*) v1828;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1830 = *(const int8_t *)(v1829);
      vint32m2_t v1831 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1832 = __riscv_vwmul_vx_i16m1(v1821, v1827, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1833 = __riscv_vwadd_wv_i32m2(v1831, v1832, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1834 = __riscv_vwmul_vx_i16m1(v1824, v1830, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1835 = __riscv_vwadd_wv_i32m2(v1833, v1834, 8);
      v1412 = v1835;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1836 = v19 + 808;
      const uint8_t* v1837 = (const uint8_t*) v1836;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1838 = __riscv_vle8_v_u8mf2(v1837, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1839 = __riscv_vand_vx_u8mf2(v1838, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1840 = __riscv_vzext_vf2_u16m1(v1839, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1841 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1840, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1842 = __riscv_vsrl_vx_u8mf2(v1838, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1843 = __riscv_vzext_vf2_u16m1(v1842, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1844 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1843, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1845 = v21 + 78;
      const int8_t* v1846 = (const int8_t*) v1845;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1847 = *(const int8_t *)(v1846);
      const uint8_t* v1848 = v21 + 94;
      const int8_t* v1849 = (const int8_t*) v1848;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1850 = *(const int8_t *)(v1849);
      vint32m2_t v1851 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1852 = __riscv_vwmul_vx_i16m1(v1841, v1847, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1853 = __riscv_vwadd_wv_i32m2(v1851, v1852, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1854 = __riscv_vwmul_vx_i16m1(v1844, v1850, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1855 = __riscv_vwadd_wv_i32m2(v1853, v1854, 8);
      v1414 = v1855;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1856 = v19 + 816;
      const uint8_t* v1857 = (const uint8_t*) v1856;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1858 = __riscv_vle8_v_u8mf2(v1857, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1859 = __riscv_vand_vx_u8mf2(v1858, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1860 = __riscv_vzext_vf2_u16m1(v1859, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1861 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1860, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1862 = __riscv_vsrl_vx_u8mf2(v1858, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1863 = __riscv_vzext_vf2_u16m1(v1862, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1864 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1863, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1865 = v21 + 79;
      const int8_t* v1866 = (const int8_t*) v1865;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1867 = *(const int8_t *)(v1866);
      const uint8_t* v1868 = v21 + 95;
      const int8_t* v1869 = (const int8_t*) v1868;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1870 = *(const int8_t *)(v1869);
      vint32m2_t v1871 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1872 = __riscv_vwmul_vx_i16m1(v1861, v1867, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1873 = __riscv_vwadd_wv_i32m2(v1871, v1872, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1874 = __riscv_vwmul_vx_i16m1(v1864, v1870, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1875 = __riscv_vwadd_wv_i32m2(v1873, v1874, 8);
      v1412 = v1875;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1876 = v19 + 824;
      const uint8_t* v1877 = (const uint8_t*) v1876;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1878 = __riscv_vle8_v_u8mf2(v1877, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1879 = __riscv_vand_vx_u8mf2(v1878, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1880 = __riscv_vzext_vf2_u16m1(v1879, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1881 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1880, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1882 = __riscv_vsrl_vx_u8mf2(v1878, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1883 = __riscv_vzext_vf2_u16m1(v1882, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1884 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1883, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1885 = v21 + 79;
      const int8_t* v1886 = (const int8_t*) v1885;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1887 = *(const int8_t *)(v1886);
      const uint8_t* v1888 = v21 + 95;
      const int8_t* v1889 = (const int8_t*) v1888;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1890 = *(const int8_t *)(v1889);
      vint32m2_t v1891 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1892 = __riscv_vwmul_vx_i16m1(v1881, v1887, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1893 = __riscv_vwadd_wv_i32m2(v1891, v1892, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1894 = __riscv_vwmul_vx_i16m1(v1884, v1890, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1895 = __riscv_vwadd_wv_i32m2(v1893, v1894, 8);
      v1414 = v1895;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1896 = v19 + 832;
      const uint8_t* v1897 = (const uint8_t*) v1896;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1898 = __riscv_vle8_v_u8mf2(v1897, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1899 = __riscv_vand_vx_u8mf2(v1898, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1900 = __riscv_vzext_vf2_u16m1(v1899, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1901 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1900, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1902 = __riscv_vsrl_vx_u8mf2(v1898, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1903 = __riscv_vzext_vf2_u16m1(v1902, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1904 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1903, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1905 = v21 + 80;
      const int8_t* v1906 = (const int8_t*) v1905;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1907 = *(const int8_t *)(v1906);
      const uint8_t* v1908 = v21 + 96;
      const int8_t* v1909 = (const int8_t*) v1908;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1910 = *(const int8_t *)(v1909);
      vint32m2_t v1911 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1912 = __riscv_vwmul_vx_i16m1(v1901, v1907, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1913 = __riscv_vwadd_wv_i32m2(v1911, v1912, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1914 = __riscv_vwmul_vx_i16m1(v1904, v1910, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1915 = __riscv_vwadd_wv_i32m2(v1913, v1914, 8);
      v1412 = v1915;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1916 = v19 + 840;
      const uint8_t* v1917 = (const uint8_t*) v1916;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1918 = __riscv_vle8_v_u8mf2(v1917, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1919 = __riscv_vand_vx_u8mf2(v1918, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1920 = __riscv_vzext_vf2_u16m1(v1919, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1921 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1920, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1922 = __riscv_vsrl_vx_u8mf2(v1918, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1923 = __riscv_vzext_vf2_u16m1(v1922, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1924 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1923, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1925 = v21 + 80;
      const int8_t* v1926 = (const int8_t*) v1925;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1927 = *(const int8_t *)(v1926);
      const uint8_t* v1928 = v21 + 96;
      const int8_t* v1929 = (const int8_t*) v1928;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1930 = *(const int8_t *)(v1929);
      vint32m2_t v1931 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1932 = __riscv_vwmul_vx_i16m1(v1921, v1927, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1933 = __riscv_vwadd_wv_i32m2(v1931, v1932, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1934 = __riscv_vwmul_vx_i16m1(v1924, v1930, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1935 = __riscv_vwadd_wv_i32m2(v1933, v1934, 8);
      v1414 = v1935;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1936 = v19 + 848;
      const uint8_t* v1937 = (const uint8_t*) v1936;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1938 = __riscv_vle8_v_u8mf2(v1937, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1939 = __riscv_vand_vx_u8mf2(v1938, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1940 = __riscv_vzext_vf2_u16m1(v1939, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1941 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1940, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1942 = __riscv_vsrl_vx_u8mf2(v1938, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1943 = __riscv_vzext_vf2_u16m1(v1942, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1944 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1943, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1945 = v21 + 81;
      const int8_t* v1946 = (const int8_t*) v1945;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1947 = *(const int8_t *)(v1946);
      const uint8_t* v1948 = v21 + 97;
      const int8_t* v1949 = (const int8_t*) v1948;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1950 = *(const int8_t *)(v1949);
      vint32m2_t v1951 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1952 = __riscv_vwmul_vx_i16m1(v1941, v1947, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1953 = __riscv_vwadd_wv_i32m2(v1951, v1952, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1954 = __riscv_vwmul_vx_i16m1(v1944, v1950, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1955 = __riscv_vwadd_wv_i32m2(v1953, v1954, 8);
      v1412 = v1955;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1956 = v19 + 856;
      const uint8_t* v1957 = (const uint8_t*) v1956;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1958 = __riscv_vle8_v_u8mf2(v1957, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1959 = __riscv_vand_vx_u8mf2(v1958, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1960 = __riscv_vzext_vf2_u16m1(v1959, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1961 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1960, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1962 = __riscv_vsrl_vx_u8mf2(v1958, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1963 = __riscv_vzext_vf2_u16m1(v1962, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1964 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1963, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1965 = v21 + 81;
      const int8_t* v1966 = (const int8_t*) v1965;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1967 = *(const int8_t *)(v1966);
      const uint8_t* v1968 = v21 + 97;
      const int8_t* v1969 = (const int8_t*) v1968;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1970 = *(const int8_t *)(v1969);
      vint32m2_t v1971 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1972 = __riscv_vwmul_vx_i16m1(v1961, v1967, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1973 = __riscv_vwadd_wv_i32m2(v1971, v1972, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1974 = __riscv_vwmul_vx_i16m1(v1964, v1970, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1975 = __riscv_vwadd_wv_i32m2(v1973, v1974, 8);
      v1414 = v1975;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1976 = v19 + 864;
      const uint8_t* v1977 = (const uint8_t*) v1976;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1978 = __riscv_vle8_v_u8mf2(v1977, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1979 = __riscv_vand_vx_u8mf2(v1978, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1980 = __riscv_vzext_vf2_u16m1(v1979, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1981 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1980, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1982 = __riscv_vsrl_vx_u8mf2(v1978, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v1983 = __riscv_vzext_vf2_u16m1(v1982, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v1984 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v1983, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1985 = v21 + 82;
      const int8_t* v1986 = (const int8_t*) v1985;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1987 = *(const int8_t *)(v1986);
      const uint8_t* v1988 = v21 + 98;
      const int8_t* v1989 = (const int8_t*) v1988;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1990 = *(const int8_t *)(v1989);
      vint32m2_t v1991 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1992 = __riscv_vwmul_vx_i16m1(v1981, v1987, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1993 = __riscv_vwadd_wv_i32m2(v1991, v1992, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v1994 = __riscv_vwmul_vx_i16m1(v1984, v1990, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v1995 = __riscv_vwadd_wv_i32m2(v1993, v1994, 8);
      v1412 = v1995;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1996 = v19 + 872;
      const uint8_t* v1997 = (const uint8_t*) v1996;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1998 = __riscv_vle8_v_u8mf2(v1997, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1999 = __riscv_vand_vx_u8mf2(v1998, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2000 = __riscv_vzext_vf2_u16m1(v1999, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2001 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2000, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2002 = __riscv_vsrl_vx_u8mf2(v1998, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2003 = __riscv_vzext_vf2_u16m1(v2002, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2004 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2003, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2005 = v21 + 82;
      const int8_t* v2006 = (const int8_t*) v2005;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2007 = *(const int8_t *)(v2006);
      const uint8_t* v2008 = v21 + 98;
      const int8_t* v2009 = (const int8_t*) v2008;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2010 = *(const int8_t *)(v2009);
      vint32m2_t v2011 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2012 = __riscv_vwmul_vx_i16m1(v2001, v2007, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2013 = __riscv_vwadd_wv_i32m2(v2011, v2012, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2014 = __riscv_vwmul_vx_i16m1(v2004, v2010, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2015 = __riscv_vwadd_wv_i32m2(v2013, v2014, 8);
      v1414 = v2015;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2016 = v19 + 880;
      const uint8_t* v2017 = (const uint8_t*) v2016;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2018 = __riscv_vle8_v_u8mf2(v2017, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2019 = __riscv_vand_vx_u8mf2(v2018, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2020 = __riscv_vzext_vf2_u16m1(v2019, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2021 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2020, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2022 = __riscv_vsrl_vx_u8mf2(v2018, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2023 = __riscv_vzext_vf2_u16m1(v2022, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2024 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2023, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2025 = v21 + 83;
      const int8_t* v2026 = (const int8_t*) v2025;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2027 = *(const int8_t *)(v2026);
      const uint8_t* v2028 = v21 + 99;
      const int8_t* v2029 = (const int8_t*) v2028;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2030 = *(const int8_t *)(v2029);
      vint32m2_t v2031 = v1412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2032 = __riscv_vwmul_vx_i16m1(v2021, v2027, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2033 = __riscv_vwadd_wv_i32m2(v2031, v2032, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2034 = __riscv_vwmul_vx_i16m1(v2024, v2030, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2035 = __riscv_vwadd_wv_i32m2(v2033, v2034, 8);
      v1412 = v2035;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2036 = v19 + 888;
      const uint8_t* v2037 = (const uint8_t*) v2036;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2038 = __riscv_vle8_v_u8mf2(v2037, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2039 = __riscv_vand_vx_u8mf2(v2038, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2040 = __riscv_vzext_vf2_u16m1(v2039, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2041 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2040, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2042 = __riscv_vsrl_vx_u8mf2(v2038, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2043 = __riscv_vzext_vf2_u16m1(v2042, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2044 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2043, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2045 = v21 + 83;
      const int8_t* v2046 = (const int8_t*) v2045;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2047 = *(const int8_t *)(v2046);
      const uint8_t* v2048 = v21 + 99;
      const int8_t* v2049 = (const int8_t*) v2048;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2050 = *(const int8_t *)(v2049);
      vint32m2_t v2051 = v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2052 = __riscv_vwmul_vx_i16m1(v2041, v2047, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2053 = __riscv_vwadd_wv_i32m2(v2051, v2052, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2054 = __riscv_vwmul_vx_i16m1(v2044, v2050, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2055 = __riscv_vwadd_wv_i32m2(v2053, v2054, 8);
      v1414 = v2055;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_scale_fold
      vint32m2_t v2056 = v1412;
      vint32m2_t v2057 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v2058 = __riscv_vmacc_vv_i32m2(v2057, v1397, v2056, 8);
      v24 = v2058;
      vint32m2_t v2059 = v1414;
      vint32m2_t v2060 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v2061 = __riscv_vmacc_vv_i32m2(v2060, v1411, v2059, 8);
      v26 = v2061;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_signed_scale
      const uint8_t* v2062 = v19 + 80;
      const uint8_t* v2063 = (const uint8_t*) v2062;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2064 = __riscv_vle8_v_u8mf2(v2063, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2065 = __riscv_vsrl_vx_u8mf2(v2064, 4, 8);
      const uint8_t* v2066 = v19 + 32;
      const uint8_t* v2067 = (const uint8_t*) v2066;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2068 = __riscv_vle8_v_u8mf2(v2067, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2069 = __riscv_vsrl_vx_u8mf2(v2068, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2070 = __riscv_vand_vx_u8mf2(v2069, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v2071 = __riscv_vsll_vx_u8mf2(v2070, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2072 = __riscv_vor_vv_u8mf2(v2071, v2065, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2073 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2072);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v2074 = __riscv_vsub_vx_i8mf2(v2073, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v2075 = __riscv_vsext_vf4_i32m2(v2074, 8);
      const uint8_t* v2076 = v19 + 88;
      const uint8_t* v2077 = (const uint8_t*) v2076;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2078 = __riscv_vle8_v_u8mf2(v2077, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2079 = __riscv_vsrl_vx_u8mf2(v2078, 4, 8);
      const uint8_t* v2080 = v19 + 40;
      const uint8_t* v2081 = (const uint8_t*) v2080;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2082 = __riscv_vle8_v_u8mf2(v2081, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2083 = __riscv_vsrl_vx_u8mf2(v2082, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2084 = __riscv_vand_vx_u8mf2(v2083, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v2085 = __riscv_vsll_vx_u8mf2(v2084, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2086 = __riscv_vor_vv_u8mf2(v2085, v2079, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2087 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2086);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v2088 = __riscv_vsub_vx_i8mf2(v2087, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v2089 = __riscv_vsext_vf4_i32m2(v2088, 8);
      vint32m2_t v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v2091 = __riscv_vmv_v_x_i32m2(0, 8);
      v2090 = v2091;
      vint32m2_t v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v2093 = __riscv_vmv_v_x_i32m2(0, 8);
      v2092 = v2093;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2094 = v19 + 896;
      const uint8_t* v2095 = (const uint8_t*) v2094;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2096 = __riscv_vle8_v_u8mf2(v2095, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2097 = __riscv_vand_vx_u8mf2(v2096, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2098 = __riscv_vzext_vf2_u16m1(v2097, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2099 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2098, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2100 = __riscv_vsrl_vx_u8mf2(v2096, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2101 = __riscv_vzext_vf2_u16m1(v2100, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2102 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2101, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2103 = v21 + 100;
      const int8_t* v2104 = (const int8_t*) v2103;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2105 = *(const int8_t *)(v2104);
      const uint8_t* v2106 = v21 + 116;
      const int8_t* v2107 = (const int8_t*) v2106;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2108 = *(const int8_t *)(v2107);
      vint32m2_t v2109 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2110 = __riscv_vwmul_vx_i16m1(v2099, v2105, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2111 = __riscv_vwadd_wv_i32m2(v2109, v2110, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2112 = __riscv_vwmul_vx_i16m1(v2102, v2108, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2113 = __riscv_vwadd_wv_i32m2(v2111, v2112, 8);
      v2090 = v2113;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2114 = v19 + 904;
      const uint8_t* v2115 = (const uint8_t*) v2114;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2116 = __riscv_vle8_v_u8mf2(v2115, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2117 = __riscv_vand_vx_u8mf2(v2116, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2118 = __riscv_vzext_vf2_u16m1(v2117, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2119 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2118, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2120 = __riscv_vsrl_vx_u8mf2(v2116, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2121 = __riscv_vzext_vf2_u16m1(v2120, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2122 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2121, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2123 = v21 + 100;
      const int8_t* v2124 = (const int8_t*) v2123;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2125 = *(const int8_t *)(v2124);
      const uint8_t* v2126 = v21 + 116;
      const int8_t* v2127 = (const int8_t*) v2126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2128 = *(const int8_t *)(v2127);
      vint32m2_t v2129 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2130 = __riscv_vwmul_vx_i16m1(v2119, v2125, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2131 = __riscv_vwadd_wv_i32m2(v2129, v2130, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2132 = __riscv_vwmul_vx_i16m1(v2122, v2128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2133 = __riscv_vwadd_wv_i32m2(v2131, v2132, 8);
      v2092 = v2133;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2134 = v19 + 912;
      const uint8_t* v2135 = (const uint8_t*) v2134;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2136 = __riscv_vle8_v_u8mf2(v2135, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2137 = __riscv_vand_vx_u8mf2(v2136, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2138 = __riscv_vzext_vf2_u16m1(v2137, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2139 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2138, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2140 = __riscv_vsrl_vx_u8mf2(v2136, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2141 = __riscv_vzext_vf2_u16m1(v2140, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2142 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2141, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2143 = v21 + 101;
      const int8_t* v2144 = (const int8_t*) v2143;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2145 = *(const int8_t *)(v2144);
      const uint8_t* v2146 = v21 + 117;
      const int8_t* v2147 = (const int8_t*) v2146;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2148 = *(const int8_t *)(v2147);
      vint32m2_t v2149 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2150 = __riscv_vwmul_vx_i16m1(v2139, v2145, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2151 = __riscv_vwadd_wv_i32m2(v2149, v2150, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2152 = __riscv_vwmul_vx_i16m1(v2142, v2148, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2153 = __riscv_vwadd_wv_i32m2(v2151, v2152, 8);
      v2090 = v2153;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2154 = v19 + 920;
      const uint8_t* v2155 = (const uint8_t*) v2154;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2156 = __riscv_vle8_v_u8mf2(v2155, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2157 = __riscv_vand_vx_u8mf2(v2156, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2158 = __riscv_vzext_vf2_u16m1(v2157, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2159 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2158, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2160 = __riscv_vsrl_vx_u8mf2(v2156, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2161 = __riscv_vzext_vf2_u16m1(v2160, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2162 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2161, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2163 = v21 + 101;
      const int8_t* v2164 = (const int8_t*) v2163;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2165 = *(const int8_t *)(v2164);
      const uint8_t* v2166 = v21 + 117;
      const int8_t* v2167 = (const int8_t*) v2166;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2168 = *(const int8_t *)(v2167);
      vint32m2_t v2169 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2170 = __riscv_vwmul_vx_i16m1(v2159, v2165, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2171 = __riscv_vwadd_wv_i32m2(v2169, v2170, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2172 = __riscv_vwmul_vx_i16m1(v2162, v2168, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2173 = __riscv_vwadd_wv_i32m2(v2171, v2172, 8);
      v2092 = v2173;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2174 = v19 + 928;
      const uint8_t* v2175 = (const uint8_t*) v2174;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2176 = __riscv_vle8_v_u8mf2(v2175, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2177 = __riscv_vand_vx_u8mf2(v2176, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2178 = __riscv_vzext_vf2_u16m1(v2177, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2179 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2178, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2180 = __riscv_vsrl_vx_u8mf2(v2176, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2181 = __riscv_vzext_vf2_u16m1(v2180, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2182 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2181, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2183 = v21 + 102;
      const int8_t* v2184 = (const int8_t*) v2183;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2185 = *(const int8_t *)(v2184);
      const uint8_t* v2186 = v21 + 118;
      const int8_t* v2187 = (const int8_t*) v2186;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2188 = *(const int8_t *)(v2187);
      vint32m2_t v2189 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2190 = __riscv_vwmul_vx_i16m1(v2179, v2185, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2191 = __riscv_vwadd_wv_i32m2(v2189, v2190, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2192 = __riscv_vwmul_vx_i16m1(v2182, v2188, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2193 = __riscv_vwadd_wv_i32m2(v2191, v2192, 8);
      v2090 = v2193;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2194 = v19 + 936;
      const uint8_t* v2195 = (const uint8_t*) v2194;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2196 = __riscv_vle8_v_u8mf2(v2195, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2197 = __riscv_vand_vx_u8mf2(v2196, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2198 = __riscv_vzext_vf2_u16m1(v2197, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2199 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2198, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2200 = __riscv_vsrl_vx_u8mf2(v2196, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2201 = __riscv_vzext_vf2_u16m1(v2200, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2202 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2201, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2203 = v21 + 102;
      const int8_t* v2204 = (const int8_t*) v2203;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2205 = *(const int8_t *)(v2204);
      const uint8_t* v2206 = v21 + 118;
      const int8_t* v2207 = (const int8_t*) v2206;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2208 = *(const int8_t *)(v2207);
      vint32m2_t v2209 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2210 = __riscv_vwmul_vx_i16m1(v2199, v2205, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2211 = __riscv_vwadd_wv_i32m2(v2209, v2210, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2212 = __riscv_vwmul_vx_i16m1(v2202, v2208, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2213 = __riscv_vwadd_wv_i32m2(v2211, v2212, 8);
      v2092 = v2213;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2214 = v19 + 944;
      const uint8_t* v2215 = (const uint8_t*) v2214;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2216 = __riscv_vle8_v_u8mf2(v2215, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2217 = __riscv_vand_vx_u8mf2(v2216, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2218 = __riscv_vzext_vf2_u16m1(v2217, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2219 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2218, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2220 = __riscv_vsrl_vx_u8mf2(v2216, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2221 = __riscv_vzext_vf2_u16m1(v2220, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2222 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2221, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2223 = v21 + 103;
      const int8_t* v2224 = (const int8_t*) v2223;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2225 = *(const int8_t *)(v2224);
      const uint8_t* v2226 = v21 + 119;
      const int8_t* v2227 = (const int8_t*) v2226;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2228 = *(const int8_t *)(v2227);
      vint32m2_t v2229 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2230 = __riscv_vwmul_vx_i16m1(v2219, v2225, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2231 = __riscv_vwadd_wv_i32m2(v2229, v2230, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2232 = __riscv_vwmul_vx_i16m1(v2222, v2228, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2233 = __riscv_vwadd_wv_i32m2(v2231, v2232, 8);
      v2090 = v2233;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2234 = v19 + 952;
      const uint8_t* v2235 = (const uint8_t*) v2234;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2236 = __riscv_vle8_v_u8mf2(v2235, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2237 = __riscv_vand_vx_u8mf2(v2236, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2238 = __riscv_vzext_vf2_u16m1(v2237, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2239 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2238, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2240 = __riscv_vsrl_vx_u8mf2(v2236, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2241 = __riscv_vzext_vf2_u16m1(v2240, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2242 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2241, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2243 = v21 + 103;
      const int8_t* v2244 = (const int8_t*) v2243;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2245 = *(const int8_t *)(v2244);
      const uint8_t* v2246 = v21 + 119;
      const int8_t* v2247 = (const int8_t*) v2246;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2248 = *(const int8_t *)(v2247);
      vint32m2_t v2249 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2250 = __riscv_vwmul_vx_i16m1(v2239, v2245, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2251 = __riscv_vwadd_wv_i32m2(v2249, v2250, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2252 = __riscv_vwmul_vx_i16m1(v2242, v2248, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2253 = __riscv_vwadd_wv_i32m2(v2251, v2252, 8);
      v2092 = v2253;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2254 = v19 + 960;
      const uint8_t* v2255 = (const uint8_t*) v2254;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2256 = __riscv_vle8_v_u8mf2(v2255, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2257 = __riscv_vand_vx_u8mf2(v2256, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2258 = __riscv_vzext_vf2_u16m1(v2257, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2259 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2258, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2260 = __riscv_vsrl_vx_u8mf2(v2256, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2261 = __riscv_vzext_vf2_u16m1(v2260, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2262 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2261, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2263 = v21 + 104;
      const int8_t* v2264 = (const int8_t*) v2263;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2265 = *(const int8_t *)(v2264);
      const uint8_t* v2266 = v21 + 120;
      const int8_t* v2267 = (const int8_t*) v2266;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2268 = *(const int8_t *)(v2267);
      vint32m2_t v2269 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2270 = __riscv_vwmul_vx_i16m1(v2259, v2265, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2271 = __riscv_vwadd_wv_i32m2(v2269, v2270, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2272 = __riscv_vwmul_vx_i16m1(v2262, v2268, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2273 = __riscv_vwadd_wv_i32m2(v2271, v2272, 8);
      v2090 = v2273;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2274 = v19 + 968;
      const uint8_t* v2275 = (const uint8_t*) v2274;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2276 = __riscv_vle8_v_u8mf2(v2275, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2277 = __riscv_vand_vx_u8mf2(v2276, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2278 = __riscv_vzext_vf2_u16m1(v2277, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2279 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2278, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2280 = __riscv_vsrl_vx_u8mf2(v2276, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2281 = __riscv_vzext_vf2_u16m1(v2280, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2282 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2281, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2283 = v21 + 104;
      const int8_t* v2284 = (const int8_t*) v2283;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2285 = *(const int8_t *)(v2284);
      const uint8_t* v2286 = v21 + 120;
      const int8_t* v2287 = (const int8_t*) v2286;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2288 = *(const int8_t *)(v2287);
      vint32m2_t v2289 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2290 = __riscv_vwmul_vx_i16m1(v2279, v2285, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2291 = __riscv_vwadd_wv_i32m2(v2289, v2290, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2292 = __riscv_vwmul_vx_i16m1(v2282, v2288, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2293 = __riscv_vwadd_wv_i32m2(v2291, v2292, 8);
      v2092 = v2293;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2294 = v19 + 976;
      const uint8_t* v2295 = (const uint8_t*) v2294;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2296 = __riscv_vle8_v_u8mf2(v2295, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2297 = __riscv_vand_vx_u8mf2(v2296, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2298 = __riscv_vzext_vf2_u16m1(v2297, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2299 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2298, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2300 = __riscv_vsrl_vx_u8mf2(v2296, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2301 = __riscv_vzext_vf2_u16m1(v2300, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2302 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2301, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2303 = v21 + 105;
      const int8_t* v2304 = (const int8_t*) v2303;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2305 = *(const int8_t *)(v2304);
      const uint8_t* v2306 = v21 + 121;
      const int8_t* v2307 = (const int8_t*) v2306;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2308 = *(const int8_t *)(v2307);
      vint32m2_t v2309 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2310 = __riscv_vwmul_vx_i16m1(v2299, v2305, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2311 = __riscv_vwadd_wv_i32m2(v2309, v2310, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2312 = __riscv_vwmul_vx_i16m1(v2302, v2308, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2313 = __riscv_vwadd_wv_i32m2(v2311, v2312, 8);
      v2090 = v2313;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2314 = v19 + 984;
      const uint8_t* v2315 = (const uint8_t*) v2314;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2316 = __riscv_vle8_v_u8mf2(v2315, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2317 = __riscv_vand_vx_u8mf2(v2316, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2318 = __riscv_vzext_vf2_u16m1(v2317, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2319 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2318, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2320 = __riscv_vsrl_vx_u8mf2(v2316, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2321 = __riscv_vzext_vf2_u16m1(v2320, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2322 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2321, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2323 = v21 + 105;
      const int8_t* v2324 = (const int8_t*) v2323;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2325 = *(const int8_t *)(v2324);
      const uint8_t* v2326 = v21 + 121;
      const int8_t* v2327 = (const int8_t*) v2326;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2328 = *(const int8_t *)(v2327);
      vint32m2_t v2329 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2330 = __riscv_vwmul_vx_i16m1(v2319, v2325, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2331 = __riscv_vwadd_wv_i32m2(v2329, v2330, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2332 = __riscv_vwmul_vx_i16m1(v2322, v2328, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2333 = __riscv_vwadd_wv_i32m2(v2331, v2332, 8);
      v2092 = v2333;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2334 = v19 + 992;
      const uint8_t* v2335 = (const uint8_t*) v2334;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2336 = __riscv_vle8_v_u8mf2(v2335, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2337 = __riscv_vand_vx_u8mf2(v2336, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2338 = __riscv_vzext_vf2_u16m1(v2337, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2339 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2338, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2340 = __riscv_vsrl_vx_u8mf2(v2336, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2341 = __riscv_vzext_vf2_u16m1(v2340, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2342 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2341, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2343 = v21 + 106;
      const int8_t* v2344 = (const int8_t*) v2343;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2345 = *(const int8_t *)(v2344);
      const uint8_t* v2346 = v21 + 122;
      const int8_t* v2347 = (const int8_t*) v2346;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2348 = *(const int8_t *)(v2347);
      vint32m2_t v2349 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2350 = __riscv_vwmul_vx_i16m1(v2339, v2345, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2351 = __riscv_vwadd_wv_i32m2(v2349, v2350, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2352 = __riscv_vwmul_vx_i16m1(v2342, v2348, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2353 = __riscv_vwadd_wv_i32m2(v2351, v2352, 8);
      v2090 = v2353;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2354 = v19 + 1000;
      const uint8_t* v2355 = (const uint8_t*) v2354;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2356 = __riscv_vle8_v_u8mf2(v2355, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2357 = __riscv_vand_vx_u8mf2(v2356, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2358 = __riscv_vzext_vf2_u16m1(v2357, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2359 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2358, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2360 = __riscv_vsrl_vx_u8mf2(v2356, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2361 = __riscv_vzext_vf2_u16m1(v2360, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2362 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2361, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2363 = v21 + 106;
      const int8_t* v2364 = (const int8_t*) v2363;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2365 = *(const int8_t *)(v2364);
      const uint8_t* v2366 = v21 + 122;
      const int8_t* v2367 = (const int8_t*) v2366;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2368 = *(const int8_t *)(v2367);
      vint32m2_t v2369 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2370 = __riscv_vwmul_vx_i16m1(v2359, v2365, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2371 = __riscv_vwadd_wv_i32m2(v2369, v2370, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2372 = __riscv_vwmul_vx_i16m1(v2362, v2368, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2373 = __riscv_vwadd_wv_i32m2(v2371, v2372, 8);
      v2092 = v2373;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2374 = v19 + 1008;
      const uint8_t* v2375 = (const uint8_t*) v2374;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2376 = __riscv_vle8_v_u8mf2(v2375, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2377 = __riscv_vand_vx_u8mf2(v2376, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2378 = __riscv_vzext_vf2_u16m1(v2377, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2379 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2378, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2380 = __riscv_vsrl_vx_u8mf2(v2376, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2381 = __riscv_vzext_vf2_u16m1(v2380, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2382 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2381, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2383 = v21 + 107;
      const int8_t* v2384 = (const int8_t*) v2383;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2385 = *(const int8_t *)(v2384);
      const uint8_t* v2386 = v21 + 123;
      const int8_t* v2387 = (const int8_t*) v2386;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2388 = *(const int8_t *)(v2387);
      vint32m2_t v2389 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2390 = __riscv_vwmul_vx_i16m1(v2379, v2385, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2391 = __riscv_vwadd_wv_i32m2(v2389, v2390, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2392 = __riscv_vwmul_vx_i16m1(v2382, v2388, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2393 = __riscv_vwadd_wv_i32m2(v2391, v2392, 8);
      v2090 = v2393;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2394 = v19 + 1016;
      const uint8_t* v2395 = (const uint8_t*) v2394;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2396 = __riscv_vle8_v_u8mf2(v2395, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2397 = __riscv_vand_vx_u8mf2(v2396, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2398 = __riscv_vzext_vf2_u16m1(v2397, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2399 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2398, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2400 = __riscv_vsrl_vx_u8mf2(v2396, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2401 = __riscv_vzext_vf2_u16m1(v2400, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2402 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2401, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2403 = v21 + 107;
      const int8_t* v2404 = (const int8_t*) v2403;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2405 = *(const int8_t *)(v2404);
      const uint8_t* v2406 = v21 + 123;
      const int8_t* v2407 = (const int8_t*) v2406;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2408 = *(const int8_t *)(v2407);
      vint32m2_t v2409 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2410 = __riscv_vwmul_vx_i16m1(v2399, v2405, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2411 = __riscv_vwadd_wv_i32m2(v2409, v2410, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2412 = __riscv_vwmul_vx_i16m1(v2402, v2408, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2413 = __riscv_vwadd_wv_i32m2(v2411, v2412, 8);
      v2092 = v2413;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2414 = v19 + 1024;
      const uint8_t* v2415 = (const uint8_t*) v2414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2416 = __riscv_vle8_v_u8mf2(v2415, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2417 = __riscv_vand_vx_u8mf2(v2416, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2418 = __riscv_vzext_vf2_u16m1(v2417, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2419 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2418, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2420 = __riscv_vsrl_vx_u8mf2(v2416, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2421 = __riscv_vzext_vf2_u16m1(v2420, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2422 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2421, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2423 = v21 + 108;
      const int8_t* v2424 = (const int8_t*) v2423;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2425 = *(const int8_t *)(v2424);
      const uint8_t* v2426 = v21 + 124;
      const int8_t* v2427 = (const int8_t*) v2426;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2428 = *(const int8_t *)(v2427);
      vint32m2_t v2429 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2430 = __riscv_vwmul_vx_i16m1(v2419, v2425, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2431 = __riscv_vwadd_wv_i32m2(v2429, v2430, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2432 = __riscv_vwmul_vx_i16m1(v2422, v2428, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2433 = __riscv_vwadd_wv_i32m2(v2431, v2432, 8);
      v2090 = v2433;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2434 = v19 + 1032;
      const uint8_t* v2435 = (const uint8_t*) v2434;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2436 = __riscv_vle8_v_u8mf2(v2435, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2437 = __riscv_vand_vx_u8mf2(v2436, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2438 = __riscv_vzext_vf2_u16m1(v2437, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2439 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2438, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2440 = __riscv_vsrl_vx_u8mf2(v2436, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2441 = __riscv_vzext_vf2_u16m1(v2440, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2442 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2441, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2443 = v21 + 108;
      const int8_t* v2444 = (const int8_t*) v2443;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2445 = *(const int8_t *)(v2444);
      const uint8_t* v2446 = v21 + 124;
      const int8_t* v2447 = (const int8_t*) v2446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2448 = *(const int8_t *)(v2447);
      vint32m2_t v2449 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2450 = __riscv_vwmul_vx_i16m1(v2439, v2445, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2451 = __riscv_vwadd_wv_i32m2(v2449, v2450, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2452 = __riscv_vwmul_vx_i16m1(v2442, v2448, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2453 = __riscv_vwadd_wv_i32m2(v2451, v2452, 8);
      v2092 = v2453;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2454 = v19 + 1040;
      const uint8_t* v2455 = (const uint8_t*) v2454;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2456 = __riscv_vle8_v_u8mf2(v2455, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2457 = __riscv_vand_vx_u8mf2(v2456, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2458 = __riscv_vzext_vf2_u16m1(v2457, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2459 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2458, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2460 = __riscv_vsrl_vx_u8mf2(v2456, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2461 = __riscv_vzext_vf2_u16m1(v2460, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2462 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2461, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2463 = v21 + 109;
      const int8_t* v2464 = (const int8_t*) v2463;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2465 = *(const int8_t *)(v2464);
      const uint8_t* v2466 = v21 + 125;
      const int8_t* v2467 = (const int8_t*) v2466;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2468 = *(const int8_t *)(v2467);
      vint32m2_t v2469 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2470 = __riscv_vwmul_vx_i16m1(v2459, v2465, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2471 = __riscv_vwadd_wv_i32m2(v2469, v2470, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2472 = __riscv_vwmul_vx_i16m1(v2462, v2468, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2473 = __riscv_vwadd_wv_i32m2(v2471, v2472, 8);
      v2090 = v2473;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2474 = v19 + 1048;
      const uint8_t* v2475 = (const uint8_t*) v2474;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2476 = __riscv_vle8_v_u8mf2(v2475, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2477 = __riscv_vand_vx_u8mf2(v2476, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2478 = __riscv_vzext_vf2_u16m1(v2477, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2479 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2478, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2480 = __riscv_vsrl_vx_u8mf2(v2476, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2481 = __riscv_vzext_vf2_u16m1(v2480, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2482 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2481, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2483 = v21 + 109;
      const int8_t* v2484 = (const int8_t*) v2483;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2485 = *(const int8_t *)(v2484);
      const uint8_t* v2486 = v21 + 125;
      const int8_t* v2487 = (const int8_t*) v2486;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2488 = *(const int8_t *)(v2487);
      vint32m2_t v2489 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2490 = __riscv_vwmul_vx_i16m1(v2479, v2485, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2491 = __riscv_vwadd_wv_i32m2(v2489, v2490, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2492 = __riscv_vwmul_vx_i16m1(v2482, v2488, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2493 = __riscv_vwadd_wv_i32m2(v2491, v2492, 8);
      v2092 = v2493;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2494 = v19 + 1056;
      const uint8_t* v2495 = (const uint8_t*) v2494;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2496 = __riscv_vle8_v_u8mf2(v2495, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2497 = __riscv_vand_vx_u8mf2(v2496, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2498 = __riscv_vzext_vf2_u16m1(v2497, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2499 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2498, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2500 = __riscv_vsrl_vx_u8mf2(v2496, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2501 = __riscv_vzext_vf2_u16m1(v2500, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2502 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2501, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2503 = v21 + 110;
      const int8_t* v2504 = (const int8_t*) v2503;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2505 = *(const int8_t *)(v2504);
      const uint8_t* v2506 = v21 + 126;
      const int8_t* v2507 = (const int8_t*) v2506;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2508 = *(const int8_t *)(v2507);
      vint32m2_t v2509 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2510 = __riscv_vwmul_vx_i16m1(v2499, v2505, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2511 = __riscv_vwadd_wv_i32m2(v2509, v2510, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2512 = __riscv_vwmul_vx_i16m1(v2502, v2508, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2513 = __riscv_vwadd_wv_i32m2(v2511, v2512, 8);
      v2090 = v2513;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2514 = v19 + 1064;
      const uint8_t* v2515 = (const uint8_t*) v2514;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2516 = __riscv_vle8_v_u8mf2(v2515, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2517 = __riscv_vand_vx_u8mf2(v2516, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2518 = __riscv_vzext_vf2_u16m1(v2517, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2519 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2518, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2520 = __riscv_vsrl_vx_u8mf2(v2516, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2521 = __riscv_vzext_vf2_u16m1(v2520, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2522 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2521, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2523 = v21 + 110;
      const int8_t* v2524 = (const int8_t*) v2523;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2525 = *(const int8_t *)(v2524);
      const uint8_t* v2526 = v21 + 126;
      const int8_t* v2527 = (const int8_t*) v2526;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2528 = *(const int8_t *)(v2527);
      vint32m2_t v2529 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2530 = __riscv_vwmul_vx_i16m1(v2519, v2525, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2531 = __riscv_vwadd_wv_i32m2(v2529, v2530, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2532 = __riscv_vwmul_vx_i16m1(v2522, v2528, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2533 = __riscv_vwadd_wv_i32m2(v2531, v2532, 8);
      v2092 = v2533;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2534 = v19 + 1072;
      const uint8_t* v2535 = (const uint8_t*) v2534;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2536 = __riscv_vle8_v_u8mf2(v2535, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2537 = __riscv_vand_vx_u8mf2(v2536, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2538 = __riscv_vzext_vf2_u16m1(v2537, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2539 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2538, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2540 = __riscv_vsrl_vx_u8mf2(v2536, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2541 = __riscv_vzext_vf2_u16m1(v2540, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2542 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2541, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2543 = v21 + 111;
      const int8_t* v2544 = (const int8_t*) v2543;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2545 = *(const int8_t *)(v2544);
      const uint8_t* v2546 = v21 + 127;
      const int8_t* v2547 = (const int8_t*) v2546;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2548 = *(const int8_t *)(v2547);
      vint32m2_t v2549 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2550 = __riscv_vwmul_vx_i16m1(v2539, v2545, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2551 = __riscv_vwadd_wv_i32m2(v2549, v2550, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2552 = __riscv_vwmul_vx_i16m1(v2542, v2548, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2553 = __riscv_vwadd_wv_i32m2(v2551, v2552, 8);
      v2090 = v2553;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2554 = v19 + 1080;
      const uint8_t* v2555 = (const uint8_t*) v2554;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2556 = __riscv_vle8_v_u8mf2(v2555, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2557 = __riscv_vand_vx_u8mf2(v2556, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2558 = __riscv_vzext_vf2_u16m1(v2557, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2559 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2558, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2560 = __riscv_vsrl_vx_u8mf2(v2556, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2561 = __riscv_vzext_vf2_u16m1(v2560, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2562 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2561, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2563 = v21 + 111;
      const int8_t* v2564 = (const int8_t*) v2563;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2565 = *(const int8_t *)(v2564);
      const uint8_t* v2566 = v21 + 127;
      const int8_t* v2567 = (const int8_t*) v2566;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2568 = *(const int8_t *)(v2567);
      vint32m2_t v2569 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2570 = __riscv_vwmul_vx_i16m1(v2559, v2565, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2571 = __riscv_vwadd_wv_i32m2(v2569, v2570, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2572 = __riscv_vwmul_vx_i16m1(v2562, v2568, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2573 = __riscv_vwadd_wv_i32m2(v2571, v2572, 8);
      v2092 = v2573;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2574 = v19 + 1088;
      const uint8_t* v2575 = (const uint8_t*) v2574;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2576 = __riscv_vle8_v_u8mf2(v2575, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2577 = __riscv_vand_vx_u8mf2(v2576, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2578 = __riscv_vzext_vf2_u16m1(v2577, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2579 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2578, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2580 = __riscv_vsrl_vx_u8mf2(v2576, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2581 = __riscv_vzext_vf2_u16m1(v2580, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2582 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2581, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2583 = v21 + 112;
      const int8_t* v2584 = (const int8_t*) v2583;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2585 = *(const int8_t *)(v2584);
      const uint8_t* v2586 = v21 + 128;
      const int8_t* v2587 = (const int8_t*) v2586;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2588 = *(const int8_t *)(v2587);
      vint32m2_t v2589 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2590 = __riscv_vwmul_vx_i16m1(v2579, v2585, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2591 = __riscv_vwadd_wv_i32m2(v2589, v2590, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2592 = __riscv_vwmul_vx_i16m1(v2582, v2588, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2593 = __riscv_vwadd_wv_i32m2(v2591, v2592, 8);
      v2090 = v2593;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2594 = v19 + 1096;
      const uint8_t* v2595 = (const uint8_t*) v2594;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2596 = __riscv_vle8_v_u8mf2(v2595, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2597 = __riscv_vand_vx_u8mf2(v2596, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2598 = __riscv_vzext_vf2_u16m1(v2597, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2599 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2598, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2600 = __riscv_vsrl_vx_u8mf2(v2596, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2601 = __riscv_vzext_vf2_u16m1(v2600, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2602 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2601, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2603 = v21 + 112;
      const int8_t* v2604 = (const int8_t*) v2603;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2605 = *(const int8_t *)(v2604);
      const uint8_t* v2606 = v21 + 128;
      const int8_t* v2607 = (const int8_t*) v2606;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2608 = *(const int8_t *)(v2607);
      vint32m2_t v2609 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2610 = __riscv_vwmul_vx_i16m1(v2599, v2605, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2611 = __riscv_vwadd_wv_i32m2(v2609, v2610, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2612 = __riscv_vwmul_vx_i16m1(v2602, v2608, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2613 = __riscv_vwadd_wv_i32m2(v2611, v2612, 8);
      v2092 = v2613;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2614 = v19 + 1104;
      const uint8_t* v2615 = (const uint8_t*) v2614;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2616 = __riscv_vle8_v_u8mf2(v2615, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2617 = __riscv_vand_vx_u8mf2(v2616, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2618 = __riscv_vzext_vf2_u16m1(v2617, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2619 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2618, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2620 = __riscv_vsrl_vx_u8mf2(v2616, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2621 = __riscv_vzext_vf2_u16m1(v2620, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2622 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2621, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2623 = v21 + 113;
      const int8_t* v2624 = (const int8_t*) v2623;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2625 = *(const int8_t *)(v2624);
      const uint8_t* v2626 = v21 + 129;
      const int8_t* v2627 = (const int8_t*) v2626;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2628 = *(const int8_t *)(v2627);
      vint32m2_t v2629 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2630 = __riscv_vwmul_vx_i16m1(v2619, v2625, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2631 = __riscv_vwadd_wv_i32m2(v2629, v2630, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2632 = __riscv_vwmul_vx_i16m1(v2622, v2628, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2633 = __riscv_vwadd_wv_i32m2(v2631, v2632, 8);
      v2090 = v2633;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2634 = v19 + 1112;
      const uint8_t* v2635 = (const uint8_t*) v2634;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2636 = __riscv_vle8_v_u8mf2(v2635, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2637 = __riscv_vand_vx_u8mf2(v2636, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2638 = __riscv_vzext_vf2_u16m1(v2637, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2639 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2638, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2640 = __riscv_vsrl_vx_u8mf2(v2636, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2641 = __riscv_vzext_vf2_u16m1(v2640, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2642 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2641, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2643 = v21 + 113;
      const int8_t* v2644 = (const int8_t*) v2643;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2645 = *(const int8_t *)(v2644);
      const uint8_t* v2646 = v21 + 129;
      const int8_t* v2647 = (const int8_t*) v2646;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2648 = *(const int8_t *)(v2647);
      vint32m2_t v2649 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2650 = __riscv_vwmul_vx_i16m1(v2639, v2645, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2651 = __riscv_vwadd_wv_i32m2(v2649, v2650, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2652 = __riscv_vwmul_vx_i16m1(v2642, v2648, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2653 = __riscv_vwadd_wv_i32m2(v2651, v2652, 8);
      v2092 = v2653;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2654 = v19 + 1120;
      const uint8_t* v2655 = (const uint8_t*) v2654;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2656 = __riscv_vle8_v_u8mf2(v2655, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2657 = __riscv_vand_vx_u8mf2(v2656, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2658 = __riscv_vzext_vf2_u16m1(v2657, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2659 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2658, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2660 = __riscv_vsrl_vx_u8mf2(v2656, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2661 = __riscv_vzext_vf2_u16m1(v2660, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2662 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2661, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2663 = v21 + 114;
      const int8_t* v2664 = (const int8_t*) v2663;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2665 = *(const int8_t *)(v2664);
      const uint8_t* v2666 = v21 + 130;
      const int8_t* v2667 = (const int8_t*) v2666;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2668 = *(const int8_t *)(v2667);
      vint32m2_t v2669 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2670 = __riscv_vwmul_vx_i16m1(v2659, v2665, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2671 = __riscv_vwadd_wv_i32m2(v2669, v2670, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2672 = __riscv_vwmul_vx_i16m1(v2662, v2668, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2673 = __riscv_vwadd_wv_i32m2(v2671, v2672, 8);
      v2090 = v2673;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2674 = v19 + 1128;
      const uint8_t* v2675 = (const uint8_t*) v2674;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2676 = __riscv_vle8_v_u8mf2(v2675, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2677 = __riscv_vand_vx_u8mf2(v2676, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2678 = __riscv_vzext_vf2_u16m1(v2677, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2679 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2678, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2680 = __riscv_vsrl_vx_u8mf2(v2676, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2681 = __riscv_vzext_vf2_u16m1(v2680, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2682 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2681, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2683 = v21 + 114;
      const int8_t* v2684 = (const int8_t*) v2683;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2685 = *(const int8_t *)(v2684);
      const uint8_t* v2686 = v21 + 130;
      const int8_t* v2687 = (const int8_t*) v2686;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2688 = *(const int8_t *)(v2687);
      vint32m2_t v2689 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2690 = __riscv_vwmul_vx_i16m1(v2679, v2685, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2691 = __riscv_vwadd_wv_i32m2(v2689, v2690, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2692 = __riscv_vwmul_vx_i16m1(v2682, v2688, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2693 = __riscv_vwadd_wv_i32m2(v2691, v2692, 8);
      v2092 = v2693;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2694 = v19 + 1136;
      const uint8_t* v2695 = (const uint8_t*) v2694;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2696 = __riscv_vle8_v_u8mf2(v2695, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2697 = __riscv_vand_vx_u8mf2(v2696, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2698 = __riscv_vzext_vf2_u16m1(v2697, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2699 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2698, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2700 = __riscv_vsrl_vx_u8mf2(v2696, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2701 = __riscv_vzext_vf2_u16m1(v2700, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2702 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2701, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2703 = v21 + 115;
      const int8_t* v2704 = (const int8_t*) v2703;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2705 = *(const int8_t *)(v2704);
      const uint8_t* v2706 = v21 + 131;
      const int8_t* v2707 = (const int8_t*) v2706;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2708 = *(const int8_t *)(v2707);
      vint32m2_t v2709 = v2090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2710 = __riscv_vwmul_vx_i16m1(v2699, v2705, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2711 = __riscv_vwadd_wv_i32m2(v2709, v2710, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2712 = __riscv_vwmul_vx_i16m1(v2702, v2708, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2713 = __riscv_vwadd_wv_i32m2(v2711, v2712, 8);
      v2090 = v2713;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2714 = v19 + 1144;
      const uint8_t* v2715 = (const uint8_t*) v2714;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2716 = __riscv_vle8_v_u8mf2(v2715, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2717 = __riscv_vand_vx_u8mf2(v2716, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2718 = __riscv_vzext_vf2_u16m1(v2717, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2719 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2718, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2720 = __riscv_vsrl_vx_u8mf2(v2716, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2721 = __riscv_vzext_vf2_u16m1(v2720, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2722 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2721, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2723 = v21 + 115;
      const int8_t* v2724 = (const int8_t*) v2723;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2725 = *(const int8_t *)(v2724);
      const uint8_t* v2726 = v21 + 131;
      const int8_t* v2727 = (const int8_t*) v2726;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2728 = *(const int8_t *)(v2727);
      vint32m2_t v2729 = v2092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2730 = __riscv_vwmul_vx_i16m1(v2719, v2725, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2731 = __riscv_vwadd_wv_i32m2(v2729, v2730, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2732 = __riscv_vwmul_vx_i16m1(v2722, v2728, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2733 = __riscv_vwadd_wv_i32m2(v2731, v2732, 8);
      v2092 = v2733;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_scale_fold
      vint32m2_t v2734 = v2090;
      vint32m2_t v2735 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v2736 = __riscv_vmacc_vv_i32m2(v2735, v2075, v2734, 8);
      v24 = v2736;
      vint32m2_t v2737 = v2092;
      vint32m2_t v2738 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v2739 = __riscv_vmacc_vv_i32m2(v2738, v2089, v2737, 8);
      v26 = v2739;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_signed_scale
      const uint8_t* v2740 = v19 + 96;
      const uint8_t* v2741 = (const uint8_t*) v2740;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2742 = __riscv_vle8_v_u8mf2(v2741, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2743 = __riscv_vand_vx_u8mf2(v2742, 0x0F, 8);
      const uint8_t* v2744 = v19 + 48;
      const uint8_t* v2745 = (const uint8_t*) v2744;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2746 = __riscv_vle8_v_u8mf2(v2745, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2747 = __riscv_vsrl_vx_u8mf2(v2746, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2748 = __riscv_vand_vx_u8mf2(v2747, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v2749 = __riscv_vsll_vx_u8mf2(v2748, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2750 = __riscv_vor_vv_u8mf2(v2749, v2743, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2751 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2750);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v2752 = __riscv_vsub_vx_i8mf2(v2751, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v2753 = __riscv_vsext_vf4_i32m2(v2752, 8);
      const uint8_t* v2754 = v19 + 104;
      const uint8_t* v2755 = (const uint8_t*) v2754;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2756 = __riscv_vle8_v_u8mf2(v2755, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2757 = __riscv_vand_vx_u8mf2(v2756, 0x0F, 8);
      const uint8_t* v2758 = v19 + 56;
      const uint8_t* v2759 = (const uint8_t*) v2758;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2760 = __riscv_vle8_v_u8mf2(v2759, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2761 = __riscv_vsrl_vx_u8mf2(v2760, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2762 = __riscv_vand_vx_u8mf2(v2761, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v2763 = __riscv_vsll_vx_u8mf2(v2762, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2764 = __riscv_vor_vv_u8mf2(v2763, v2757, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2765 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2764);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v2766 = __riscv_vsub_vx_i8mf2(v2765, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v2767 = __riscv_vsext_vf4_i32m2(v2766, 8);
      vint32m2_t v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v2769 = __riscv_vmv_v_x_i32m2(0, 8);
      v2768 = v2769;
      vint32m2_t v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v2771 = __riscv_vmv_v_x_i32m2(0, 8);
      v2770 = v2771;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2772 = v19 + 1152;
      const uint8_t* v2773 = (const uint8_t*) v2772;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2774 = __riscv_vle8_v_u8mf2(v2773, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2775 = __riscv_vand_vx_u8mf2(v2774, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2776 = __riscv_vzext_vf2_u16m1(v2775, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2777 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2776, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2778 = __riscv_vsrl_vx_u8mf2(v2774, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2779 = __riscv_vzext_vf2_u16m1(v2778, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2780 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2779, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2781 = v21 + 132;
      const int8_t* v2782 = (const int8_t*) v2781;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2783 = *(const int8_t *)(v2782);
      const uint8_t* v2784 = v21 + 148;
      const int8_t* v2785 = (const int8_t*) v2784;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2786 = *(const int8_t *)(v2785);
      vint32m2_t v2787 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2788 = __riscv_vwmul_vx_i16m1(v2777, v2783, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2789 = __riscv_vwadd_wv_i32m2(v2787, v2788, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2790 = __riscv_vwmul_vx_i16m1(v2780, v2786, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2791 = __riscv_vwadd_wv_i32m2(v2789, v2790, 8);
      v2768 = v2791;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2792 = v19 + 1160;
      const uint8_t* v2793 = (const uint8_t*) v2792;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2794 = __riscv_vle8_v_u8mf2(v2793, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2795 = __riscv_vand_vx_u8mf2(v2794, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2796 = __riscv_vzext_vf2_u16m1(v2795, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2797 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2796, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2798 = __riscv_vsrl_vx_u8mf2(v2794, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2799 = __riscv_vzext_vf2_u16m1(v2798, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2800 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2799, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2801 = v21 + 132;
      const int8_t* v2802 = (const int8_t*) v2801;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2803 = *(const int8_t *)(v2802);
      const uint8_t* v2804 = v21 + 148;
      const int8_t* v2805 = (const int8_t*) v2804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2806 = *(const int8_t *)(v2805);
      vint32m2_t v2807 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2808 = __riscv_vwmul_vx_i16m1(v2797, v2803, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2809 = __riscv_vwadd_wv_i32m2(v2807, v2808, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2810 = __riscv_vwmul_vx_i16m1(v2800, v2806, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2811 = __riscv_vwadd_wv_i32m2(v2809, v2810, 8);
      v2770 = v2811;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2812 = v19 + 1168;
      const uint8_t* v2813 = (const uint8_t*) v2812;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2814 = __riscv_vle8_v_u8mf2(v2813, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2815 = __riscv_vand_vx_u8mf2(v2814, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2816 = __riscv_vzext_vf2_u16m1(v2815, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2817 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2816, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2818 = __riscv_vsrl_vx_u8mf2(v2814, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2819 = __riscv_vzext_vf2_u16m1(v2818, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2820 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2819, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2821 = v21 + 133;
      const int8_t* v2822 = (const int8_t*) v2821;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2823 = *(const int8_t *)(v2822);
      const uint8_t* v2824 = v21 + 149;
      const int8_t* v2825 = (const int8_t*) v2824;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2826 = *(const int8_t *)(v2825);
      vint32m2_t v2827 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2828 = __riscv_vwmul_vx_i16m1(v2817, v2823, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2829 = __riscv_vwadd_wv_i32m2(v2827, v2828, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2830 = __riscv_vwmul_vx_i16m1(v2820, v2826, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2831 = __riscv_vwadd_wv_i32m2(v2829, v2830, 8);
      v2768 = v2831;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2832 = v19 + 1176;
      const uint8_t* v2833 = (const uint8_t*) v2832;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2834 = __riscv_vle8_v_u8mf2(v2833, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2835 = __riscv_vand_vx_u8mf2(v2834, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2836 = __riscv_vzext_vf2_u16m1(v2835, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2837 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2836, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2838 = __riscv_vsrl_vx_u8mf2(v2834, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2839 = __riscv_vzext_vf2_u16m1(v2838, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2840 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2839, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2841 = v21 + 133;
      const int8_t* v2842 = (const int8_t*) v2841;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2843 = *(const int8_t *)(v2842);
      const uint8_t* v2844 = v21 + 149;
      const int8_t* v2845 = (const int8_t*) v2844;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2846 = *(const int8_t *)(v2845);
      vint32m2_t v2847 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2848 = __riscv_vwmul_vx_i16m1(v2837, v2843, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2849 = __riscv_vwadd_wv_i32m2(v2847, v2848, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2850 = __riscv_vwmul_vx_i16m1(v2840, v2846, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2851 = __riscv_vwadd_wv_i32m2(v2849, v2850, 8);
      v2770 = v2851;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2852 = v19 + 1184;
      const uint8_t* v2853 = (const uint8_t*) v2852;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2854 = __riscv_vle8_v_u8mf2(v2853, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2855 = __riscv_vand_vx_u8mf2(v2854, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2856 = __riscv_vzext_vf2_u16m1(v2855, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2857 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2856, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2858 = __riscv_vsrl_vx_u8mf2(v2854, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2859 = __riscv_vzext_vf2_u16m1(v2858, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2860 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2859, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2861 = v21 + 134;
      const int8_t* v2862 = (const int8_t*) v2861;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2863 = *(const int8_t *)(v2862);
      const uint8_t* v2864 = v21 + 150;
      const int8_t* v2865 = (const int8_t*) v2864;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2866 = *(const int8_t *)(v2865);
      vint32m2_t v2867 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2868 = __riscv_vwmul_vx_i16m1(v2857, v2863, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2869 = __riscv_vwadd_wv_i32m2(v2867, v2868, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2870 = __riscv_vwmul_vx_i16m1(v2860, v2866, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2871 = __riscv_vwadd_wv_i32m2(v2869, v2870, 8);
      v2768 = v2871;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2872 = v19 + 1192;
      const uint8_t* v2873 = (const uint8_t*) v2872;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2874 = __riscv_vle8_v_u8mf2(v2873, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2875 = __riscv_vand_vx_u8mf2(v2874, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2876 = __riscv_vzext_vf2_u16m1(v2875, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2877 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2876, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2878 = __riscv_vsrl_vx_u8mf2(v2874, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2879 = __riscv_vzext_vf2_u16m1(v2878, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2880 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2879, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2881 = v21 + 134;
      const int8_t* v2882 = (const int8_t*) v2881;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2883 = *(const int8_t *)(v2882);
      const uint8_t* v2884 = v21 + 150;
      const int8_t* v2885 = (const int8_t*) v2884;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2886 = *(const int8_t *)(v2885);
      vint32m2_t v2887 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2888 = __riscv_vwmul_vx_i16m1(v2877, v2883, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2889 = __riscv_vwadd_wv_i32m2(v2887, v2888, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2890 = __riscv_vwmul_vx_i16m1(v2880, v2886, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2891 = __riscv_vwadd_wv_i32m2(v2889, v2890, 8);
      v2770 = v2891;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2892 = v19 + 1200;
      const uint8_t* v2893 = (const uint8_t*) v2892;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2894 = __riscv_vle8_v_u8mf2(v2893, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2895 = __riscv_vand_vx_u8mf2(v2894, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2896 = __riscv_vzext_vf2_u16m1(v2895, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2897 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2896, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2898 = __riscv_vsrl_vx_u8mf2(v2894, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2899 = __riscv_vzext_vf2_u16m1(v2898, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2900 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2899, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2901 = v21 + 135;
      const int8_t* v2902 = (const int8_t*) v2901;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2903 = *(const int8_t *)(v2902);
      const uint8_t* v2904 = v21 + 151;
      const int8_t* v2905 = (const int8_t*) v2904;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2906 = *(const int8_t *)(v2905);
      vint32m2_t v2907 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2908 = __riscv_vwmul_vx_i16m1(v2897, v2903, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2909 = __riscv_vwadd_wv_i32m2(v2907, v2908, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2910 = __riscv_vwmul_vx_i16m1(v2900, v2906, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2911 = __riscv_vwadd_wv_i32m2(v2909, v2910, 8);
      v2768 = v2911;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2912 = v19 + 1208;
      const uint8_t* v2913 = (const uint8_t*) v2912;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2914 = __riscv_vle8_v_u8mf2(v2913, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2915 = __riscv_vand_vx_u8mf2(v2914, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2916 = __riscv_vzext_vf2_u16m1(v2915, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2917 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2916, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2918 = __riscv_vsrl_vx_u8mf2(v2914, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2919 = __riscv_vzext_vf2_u16m1(v2918, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2920 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2919, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2921 = v21 + 135;
      const int8_t* v2922 = (const int8_t*) v2921;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2923 = *(const int8_t *)(v2922);
      const uint8_t* v2924 = v21 + 151;
      const int8_t* v2925 = (const int8_t*) v2924;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2926 = *(const int8_t *)(v2925);
      vint32m2_t v2927 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2928 = __riscv_vwmul_vx_i16m1(v2917, v2923, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2929 = __riscv_vwadd_wv_i32m2(v2927, v2928, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2930 = __riscv_vwmul_vx_i16m1(v2920, v2926, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2931 = __riscv_vwadd_wv_i32m2(v2929, v2930, 8);
      v2770 = v2931;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2932 = v19 + 1216;
      const uint8_t* v2933 = (const uint8_t*) v2932;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2934 = __riscv_vle8_v_u8mf2(v2933, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2935 = __riscv_vand_vx_u8mf2(v2934, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2936 = __riscv_vzext_vf2_u16m1(v2935, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2937 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2936, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2938 = __riscv_vsrl_vx_u8mf2(v2934, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2939 = __riscv_vzext_vf2_u16m1(v2938, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2940 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2939, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2941 = v21 + 136;
      const int8_t* v2942 = (const int8_t*) v2941;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2943 = *(const int8_t *)(v2942);
      const uint8_t* v2944 = v21 + 152;
      const int8_t* v2945 = (const int8_t*) v2944;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2946 = *(const int8_t *)(v2945);
      vint32m2_t v2947 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2948 = __riscv_vwmul_vx_i16m1(v2937, v2943, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2949 = __riscv_vwadd_wv_i32m2(v2947, v2948, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2950 = __riscv_vwmul_vx_i16m1(v2940, v2946, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2951 = __riscv_vwadd_wv_i32m2(v2949, v2950, 8);
      v2768 = v2951;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2952 = v19 + 1224;
      const uint8_t* v2953 = (const uint8_t*) v2952;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2954 = __riscv_vle8_v_u8mf2(v2953, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2955 = __riscv_vand_vx_u8mf2(v2954, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2956 = __riscv_vzext_vf2_u16m1(v2955, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2957 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2956, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2958 = __riscv_vsrl_vx_u8mf2(v2954, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2959 = __riscv_vzext_vf2_u16m1(v2958, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2960 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2959, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2961 = v21 + 136;
      const int8_t* v2962 = (const int8_t*) v2961;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2963 = *(const int8_t *)(v2962);
      const uint8_t* v2964 = v21 + 152;
      const int8_t* v2965 = (const int8_t*) v2964;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2966 = *(const int8_t *)(v2965);
      vint32m2_t v2967 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2968 = __riscv_vwmul_vx_i16m1(v2957, v2963, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2969 = __riscv_vwadd_wv_i32m2(v2967, v2968, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2970 = __riscv_vwmul_vx_i16m1(v2960, v2966, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2971 = __riscv_vwadd_wv_i32m2(v2969, v2970, 8);
      v2770 = v2971;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2972 = v19 + 1232;
      const uint8_t* v2973 = (const uint8_t*) v2972;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2974 = __riscv_vle8_v_u8mf2(v2973, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2975 = __riscv_vand_vx_u8mf2(v2974, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2976 = __riscv_vzext_vf2_u16m1(v2975, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2977 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2976, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2978 = __riscv_vsrl_vx_u8mf2(v2974, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2979 = __riscv_vzext_vf2_u16m1(v2978, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2980 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2979, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2981 = v21 + 137;
      const int8_t* v2982 = (const int8_t*) v2981;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2983 = *(const int8_t *)(v2982);
      const uint8_t* v2984 = v21 + 153;
      const int8_t* v2985 = (const int8_t*) v2984;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2986 = *(const int8_t *)(v2985);
      vint32m2_t v2987 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2988 = __riscv_vwmul_vx_i16m1(v2977, v2983, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2989 = __riscv_vwadd_wv_i32m2(v2987, v2988, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v2990 = __riscv_vwmul_vx_i16m1(v2980, v2986, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v2991 = __riscv_vwadd_wv_i32m2(v2989, v2990, 8);
      v2768 = v2991;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2992 = v19 + 1240;
      const uint8_t* v2993 = (const uint8_t*) v2992;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2994 = __riscv_vle8_v_u8mf2(v2993, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2995 = __riscv_vand_vx_u8mf2(v2994, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2996 = __riscv_vzext_vf2_u16m1(v2995, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v2997 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2996, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2998 = __riscv_vsrl_vx_u8mf2(v2994, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v2999 = __riscv_vzext_vf2_u16m1(v2998, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3000 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v2999, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3001 = v21 + 137;
      const int8_t* v3002 = (const int8_t*) v3001;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3003 = *(const int8_t *)(v3002);
      const uint8_t* v3004 = v21 + 153;
      const int8_t* v3005 = (const int8_t*) v3004;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3006 = *(const int8_t *)(v3005);
      vint32m2_t v3007 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3008 = __riscv_vwmul_vx_i16m1(v2997, v3003, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3009 = __riscv_vwadd_wv_i32m2(v3007, v3008, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3010 = __riscv_vwmul_vx_i16m1(v3000, v3006, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3011 = __riscv_vwadd_wv_i32m2(v3009, v3010, 8);
      v2770 = v3011;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3012 = v19 + 1248;
      const uint8_t* v3013 = (const uint8_t*) v3012;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3014 = __riscv_vle8_v_u8mf2(v3013, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3015 = __riscv_vand_vx_u8mf2(v3014, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3016 = __riscv_vzext_vf2_u16m1(v3015, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3017 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3016, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3018 = __riscv_vsrl_vx_u8mf2(v3014, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3019 = __riscv_vzext_vf2_u16m1(v3018, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3020 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3019, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3021 = v21 + 138;
      const int8_t* v3022 = (const int8_t*) v3021;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3023 = *(const int8_t *)(v3022);
      const uint8_t* v3024 = v21 + 154;
      const int8_t* v3025 = (const int8_t*) v3024;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3026 = *(const int8_t *)(v3025);
      vint32m2_t v3027 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3028 = __riscv_vwmul_vx_i16m1(v3017, v3023, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3029 = __riscv_vwadd_wv_i32m2(v3027, v3028, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3030 = __riscv_vwmul_vx_i16m1(v3020, v3026, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3031 = __riscv_vwadd_wv_i32m2(v3029, v3030, 8);
      v2768 = v3031;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3032 = v19 + 1256;
      const uint8_t* v3033 = (const uint8_t*) v3032;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3034 = __riscv_vle8_v_u8mf2(v3033, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3035 = __riscv_vand_vx_u8mf2(v3034, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3036 = __riscv_vzext_vf2_u16m1(v3035, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3037 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3036, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3038 = __riscv_vsrl_vx_u8mf2(v3034, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3039 = __riscv_vzext_vf2_u16m1(v3038, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3040 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3039, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3041 = v21 + 138;
      const int8_t* v3042 = (const int8_t*) v3041;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3043 = *(const int8_t *)(v3042);
      const uint8_t* v3044 = v21 + 154;
      const int8_t* v3045 = (const int8_t*) v3044;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3046 = *(const int8_t *)(v3045);
      vint32m2_t v3047 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3048 = __riscv_vwmul_vx_i16m1(v3037, v3043, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3049 = __riscv_vwadd_wv_i32m2(v3047, v3048, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3050 = __riscv_vwmul_vx_i16m1(v3040, v3046, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3051 = __riscv_vwadd_wv_i32m2(v3049, v3050, 8);
      v2770 = v3051;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3052 = v19 + 1264;
      const uint8_t* v3053 = (const uint8_t*) v3052;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3054 = __riscv_vle8_v_u8mf2(v3053, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3055 = __riscv_vand_vx_u8mf2(v3054, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3056 = __riscv_vzext_vf2_u16m1(v3055, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3057 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3056, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3058 = __riscv_vsrl_vx_u8mf2(v3054, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3059 = __riscv_vzext_vf2_u16m1(v3058, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3060 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3059, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3061 = v21 + 139;
      const int8_t* v3062 = (const int8_t*) v3061;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3063 = *(const int8_t *)(v3062);
      const uint8_t* v3064 = v21 + 155;
      const int8_t* v3065 = (const int8_t*) v3064;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3066 = *(const int8_t *)(v3065);
      vint32m2_t v3067 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3068 = __riscv_vwmul_vx_i16m1(v3057, v3063, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3069 = __riscv_vwadd_wv_i32m2(v3067, v3068, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3070 = __riscv_vwmul_vx_i16m1(v3060, v3066, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3071 = __riscv_vwadd_wv_i32m2(v3069, v3070, 8);
      v2768 = v3071;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3072 = v19 + 1272;
      const uint8_t* v3073 = (const uint8_t*) v3072;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3074 = __riscv_vle8_v_u8mf2(v3073, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3075 = __riscv_vand_vx_u8mf2(v3074, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3076 = __riscv_vzext_vf2_u16m1(v3075, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3077 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3076, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3078 = __riscv_vsrl_vx_u8mf2(v3074, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3079 = __riscv_vzext_vf2_u16m1(v3078, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3080 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3079, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3081 = v21 + 139;
      const int8_t* v3082 = (const int8_t*) v3081;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3083 = *(const int8_t *)(v3082);
      const uint8_t* v3084 = v21 + 155;
      const int8_t* v3085 = (const int8_t*) v3084;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3086 = *(const int8_t *)(v3085);
      vint32m2_t v3087 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3088 = __riscv_vwmul_vx_i16m1(v3077, v3083, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3089 = __riscv_vwadd_wv_i32m2(v3087, v3088, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3090 = __riscv_vwmul_vx_i16m1(v3080, v3086, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3091 = __riscv_vwadd_wv_i32m2(v3089, v3090, 8);
      v2770 = v3091;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3092 = v19 + 1280;
      const uint8_t* v3093 = (const uint8_t*) v3092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3094 = __riscv_vle8_v_u8mf2(v3093, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3095 = __riscv_vand_vx_u8mf2(v3094, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3096 = __riscv_vzext_vf2_u16m1(v3095, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3097 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3096, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3098 = __riscv_vsrl_vx_u8mf2(v3094, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3099 = __riscv_vzext_vf2_u16m1(v3098, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3100 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3099, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3101 = v21 + 140;
      const int8_t* v3102 = (const int8_t*) v3101;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3103 = *(const int8_t *)(v3102);
      const uint8_t* v3104 = v21 + 156;
      const int8_t* v3105 = (const int8_t*) v3104;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3106 = *(const int8_t *)(v3105);
      vint32m2_t v3107 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3108 = __riscv_vwmul_vx_i16m1(v3097, v3103, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3109 = __riscv_vwadd_wv_i32m2(v3107, v3108, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3110 = __riscv_vwmul_vx_i16m1(v3100, v3106, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3111 = __riscv_vwadd_wv_i32m2(v3109, v3110, 8);
      v2768 = v3111;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3112 = v19 + 1288;
      const uint8_t* v3113 = (const uint8_t*) v3112;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3114 = __riscv_vle8_v_u8mf2(v3113, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3115 = __riscv_vand_vx_u8mf2(v3114, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3116 = __riscv_vzext_vf2_u16m1(v3115, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3117 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3116, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3118 = __riscv_vsrl_vx_u8mf2(v3114, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3119 = __riscv_vzext_vf2_u16m1(v3118, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3120 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3119, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3121 = v21 + 140;
      const int8_t* v3122 = (const int8_t*) v3121;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3123 = *(const int8_t *)(v3122);
      const uint8_t* v3124 = v21 + 156;
      const int8_t* v3125 = (const int8_t*) v3124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3126 = *(const int8_t *)(v3125);
      vint32m2_t v3127 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3128 = __riscv_vwmul_vx_i16m1(v3117, v3123, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3129 = __riscv_vwadd_wv_i32m2(v3127, v3128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3130 = __riscv_vwmul_vx_i16m1(v3120, v3126, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3131 = __riscv_vwadd_wv_i32m2(v3129, v3130, 8);
      v2770 = v3131;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3132 = v19 + 1296;
      const uint8_t* v3133 = (const uint8_t*) v3132;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3134 = __riscv_vle8_v_u8mf2(v3133, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3135 = __riscv_vand_vx_u8mf2(v3134, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3136 = __riscv_vzext_vf2_u16m1(v3135, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3137 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3136, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3138 = __riscv_vsrl_vx_u8mf2(v3134, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3139 = __riscv_vzext_vf2_u16m1(v3138, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3140 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3139, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3141 = v21 + 141;
      const int8_t* v3142 = (const int8_t*) v3141;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3143 = *(const int8_t *)(v3142);
      const uint8_t* v3144 = v21 + 157;
      const int8_t* v3145 = (const int8_t*) v3144;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3146 = *(const int8_t *)(v3145);
      vint32m2_t v3147 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3148 = __riscv_vwmul_vx_i16m1(v3137, v3143, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3149 = __riscv_vwadd_wv_i32m2(v3147, v3148, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3150 = __riscv_vwmul_vx_i16m1(v3140, v3146, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3151 = __riscv_vwadd_wv_i32m2(v3149, v3150, 8);
      v2768 = v3151;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3152 = v19 + 1304;
      const uint8_t* v3153 = (const uint8_t*) v3152;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3154 = __riscv_vle8_v_u8mf2(v3153, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3155 = __riscv_vand_vx_u8mf2(v3154, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3156 = __riscv_vzext_vf2_u16m1(v3155, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3157 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3156, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3158 = __riscv_vsrl_vx_u8mf2(v3154, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3159 = __riscv_vzext_vf2_u16m1(v3158, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3160 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3159, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3161 = v21 + 141;
      const int8_t* v3162 = (const int8_t*) v3161;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3163 = *(const int8_t *)(v3162);
      const uint8_t* v3164 = v21 + 157;
      const int8_t* v3165 = (const int8_t*) v3164;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3166 = *(const int8_t *)(v3165);
      vint32m2_t v3167 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3168 = __riscv_vwmul_vx_i16m1(v3157, v3163, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3169 = __riscv_vwadd_wv_i32m2(v3167, v3168, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3170 = __riscv_vwmul_vx_i16m1(v3160, v3166, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3171 = __riscv_vwadd_wv_i32m2(v3169, v3170, 8);
      v2770 = v3171;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3172 = v19 + 1312;
      const uint8_t* v3173 = (const uint8_t*) v3172;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3174 = __riscv_vle8_v_u8mf2(v3173, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3175 = __riscv_vand_vx_u8mf2(v3174, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3176 = __riscv_vzext_vf2_u16m1(v3175, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3177 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3176, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3178 = __riscv_vsrl_vx_u8mf2(v3174, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3179 = __riscv_vzext_vf2_u16m1(v3178, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3180 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3179, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3181 = v21 + 142;
      const int8_t* v3182 = (const int8_t*) v3181;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3183 = *(const int8_t *)(v3182);
      const uint8_t* v3184 = v21 + 158;
      const int8_t* v3185 = (const int8_t*) v3184;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3186 = *(const int8_t *)(v3185);
      vint32m2_t v3187 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3188 = __riscv_vwmul_vx_i16m1(v3177, v3183, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3189 = __riscv_vwadd_wv_i32m2(v3187, v3188, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3190 = __riscv_vwmul_vx_i16m1(v3180, v3186, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3191 = __riscv_vwadd_wv_i32m2(v3189, v3190, 8);
      v2768 = v3191;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3192 = v19 + 1320;
      const uint8_t* v3193 = (const uint8_t*) v3192;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3194 = __riscv_vle8_v_u8mf2(v3193, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3195 = __riscv_vand_vx_u8mf2(v3194, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3196 = __riscv_vzext_vf2_u16m1(v3195, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3197 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3196, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3198 = __riscv_vsrl_vx_u8mf2(v3194, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3199 = __riscv_vzext_vf2_u16m1(v3198, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3200 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3199, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3201 = v21 + 142;
      const int8_t* v3202 = (const int8_t*) v3201;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3203 = *(const int8_t *)(v3202);
      const uint8_t* v3204 = v21 + 158;
      const int8_t* v3205 = (const int8_t*) v3204;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3206 = *(const int8_t *)(v3205);
      vint32m2_t v3207 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3208 = __riscv_vwmul_vx_i16m1(v3197, v3203, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3209 = __riscv_vwadd_wv_i32m2(v3207, v3208, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3210 = __riscv_vwmul_vx_i16m1(v3200, v3206, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3211 = __riscv_vwadd_wv_i32m2(v3209, v3210, 8);
      v2770 = v3211;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3212 = v19 + 1328;
      const uint8_t* v3213 = (const uint8_t*) v3212;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3214 = __riscv_vle8_v_u8mf2(v3213, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3215 = __riscv_vand_vx_u8mf2(v3214, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3216 = __riscv_vzext_vf2_u16m1(v3215, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3217 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3216, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3218 = __riscv_vsrl_vx_u8mf2(v3214, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3219 = __riscv_vzext_vf2_u16m1(v3218, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3220 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3219, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3221 = v21 + 143;
      const int8_t* v3222 = (const int8_t*) v3221;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3223 = *(const int8_t *)(v3222);
      const uint8_t* v3224 = v21 + 159;
      const int8_t* v3225 = (const int8_t*) v3224;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3226 = *(const int8_t *)(v3225);
      vint32m2_t v3227 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3228 = __riscv_vwmul_vx_i16m1(v3217, v3223, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3229 = __riscv_vwadd_wv_i32m2(v3227, v3228, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3230 = __riscv_vwmul_vx_i16m1(v3220, v3226, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3231 = __riscv_vwadd_wv_i32m2(v3229, v3230, 8);
      v2768 = v3231;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3232 = v19 + 1336;
      const uint8_t* v3233 = (const uint8_t*) v3232;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3234 = __riscv_vle8_v_u8mf2(v3233, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3235 = __riscv_vand_vx_u8mf2(v3234, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3236 = __riscv_vzext_vf2_u16m1(v3235, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3237 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3236, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3238 = __riscv_vsrl_vx_u8mf2(v3234, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3239 = __riscv_vzext_vf2_u16m1(v3238, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3240 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3239, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3241 = v21 + 143;
      const int8_t* v3242 = (const int8_t*) v3241;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3243 = *(const int8_t *)(v3242);
      const uint8_t* v3244 = v21 + 159;
      const int8_t* v3245 = (const int8_t*) v3244;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3246 = *(const int8_t *)(v3245);
      vint32m2_t v3247 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3248 = __riscv_vwmul_vx_i16m1(v3237, v3243, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3249 = __riscv_vwadd_wv_i32m2(v3247, v3248, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3250 = __riscv_vwmul_vx_i16m1(v3240, v3246, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3251 = __riscv_vwadd_wv_i32m2(v3249, v3250, 8);
      v2770 = v3251;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3252 = v19 + 1344;
      const uint8_t* v3253 = (const uint8_t*) v3252;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3254 = __riscv_vle8_v_u8mf2(v3253, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3255 = __riscv_vand_vx_u8mf2(v3254, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3256 = __riscv_vzext_vf2_u16m1(v3255, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3257 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3256, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3258 = __riscv_vsrl_vx_u8mf2(v3254, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3259 = __riscv_vzext_vf2_u16m1(v3258, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3260 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3259, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3261 = v21 + 144;
      const int8_t* v3262 = (const int8_t*) v3261;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3263 = *(const int8_t *)(v3262);
      const uint8_t* v3264 = v21 + 160;
      const int8_t* v3265 = (const int8_t*) v3264;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3266 = *(const int8_t *)(v3265);
      vint32m2_t v3267 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3268 = __riscv_vwmul_vx_i16m1(v3257, v3263, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3269 = __riscv_vwadd_wv_i32m2(v3267, v3268, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3270 = __riscv_vwmul_vx_i16m1(v3260, v3266, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3271 = __riscv_vwadd_wv_i32m2(v3269, v3270, 8);
      v2768 = v3271;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3272 = v19 + 1352;
      const uint8_t* v3273 = (const uint8_t*) v3272;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3274 = __riscv_vle8_v_u8mf2(v3273, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3275 = __riscv_vand_vx_u8mf2(v3274, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3276 = __riscv_vzext_vf2_u16m1(v3275, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3277 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3276, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3278 = __riscv_vsrl_vx_u8mf2(v3274, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3279 = __riscv_vzext_vf2_u16m1(v3278, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3280 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3279, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3281 = v21 + 144;
      const int8_t* v3282 = (const int8_t*) v3281;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3283 = *(const int8_t *)(v3282);
      const uint8_t* v3284 = v21 + 160;
      const int8_t* v3285 = (const int8_t*) v3284;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3286 = *(const int8_t *)(v3285);
      vint32m2_t v3287 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3288 = __riscv_vwmul_vx_i16m1(v3277, v3283, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3289 = __riscv_vwadd_wv_i32m2(v3287, v3288, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3290 = __riscv_vwmul_vx_i16m1(v3280, v3286, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3291 = __riscv_vwadd_wv_i32m2(v3289, v3290, 8);
      v2770 = v3291;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3292 = v19 + 1360;
      const uint8_t* v3293 = (const uint8_t*) v3292;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3294 = __riscv_vle8_v_u8mf2(v3293, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3295 = __riscv_vand_vx_u8mf2(v3294, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3296 = __riscv_vzext_vf2_u16m1(v3295, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3297 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3296, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3298 = __riscv_vsrl_vx_u8mf2(v3294, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3299 = __riscv_vzext_vf2_u16m1(v3298, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3300 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3299, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3301 = v21 + 145;
      const int8_t* v3302 = (const int8_t*) v3301;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3303 = *(const int8_t *)(v3302);
      const uint8_t* v3304 = v21 + 161;
      const int8_t* v3305 = (const int8_t*) v3304;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3306 = *(const int8_t *)(v3305);
      vint32m2_t v3307 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3308 = __riscv_vwmul_vx_i16m1(v3297, v3303, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3309 = __riscv_vwadd_wv_i32m2(v3307, v3308, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3310 = __riscv_vwmul_vx_i16m1(v3300, v3306, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3311 = __riscv_vwadd_wv_i32m2(v3309, v3310, 8);
      v2768 = v3311;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3312 = v19 + 1368;
      const uint8_t* v3313 = (const uint8_t*) v3312;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3314 = __riscv_vle8_v_u8mf2(v3313, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3315 = __riscv_vand_vx_u8mf2(v3314, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3316 = __riscv_vzext_vf2_u16m1(v3315, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3317 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3316, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3318 = __riscv_vsrl_vx_u8mf2(v3314, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3319 = __riscv_vzext_vf2_u16m1(v3318, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3320 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3319, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3321 = v21 + 145;
      const int8_t* v3322 = (const int8_t*) v3321;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3323 = *(const int8_t *)(v3322);
      const uint8_t* v3324 = v21 + 161;
      const int8_t* v3325 = (const int8_t*) v3324;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3326 = *(const int8_t *)(v3325);
      vint32m2_t v3327 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3328 = __riscv_vwmul_vx_i16m1(v3317, v3323, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3329 = __riscv_vwadd_wv_i32m2(v3327, v3328, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3330 = __riscv_vwmul_vx_i16m1(v3320, v3326, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3331 = __riscv_vwadd_wv_i32m2(v3329, v3330, 8);
      v2770 = v3331;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3332 = v19 + 1376;
      const uint8_t* v3333 = (const uint8_t*) v3332;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3334 = __riscv_vle8_v_u8mf2(v3333, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3335 = __riscv_vand_vx_u8mf2(v3334, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3336 = __riscv_vzext_vf2_u16m1(v3335, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3337 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3336, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3338 = __riscv_vsrl_vx_u8mf2(v3334, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3339 = __riscv_vzext_vf2_u16m1(v3338, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3340 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3339, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3341 = v21 + 146;
      const int8_t* v3342 = (const int8_t*) v3341;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3343 = *(const int8_t *)(v3342);
      const uint8_t* v3344 = v21 + 162;
      const int8_t* v3345 = (const int8_t*) v3344;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3346 = *(const int8_t *)(v3345);
      vint32m2_t v3347 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3348 = __riscv_vwmul_vx_i16m1(v3337, v3343, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3349 = __riscv_vwadd_wv_i32m2(v3347, v3348, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3350 = __riscv_vwmul_vx_i16m1(v3340, v3346, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3351 = __riscv_vwadd_wv_i32m2(v3349, v3350, 8);
      v2768 = v3351;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3352 = v19 + 1384;
      const uint8_t* v3353 = (const uint8_t*) v3352;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3354 = __riscv_vle8_v_u8mf2(v3353, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3355 = __riscv_vand_vx_u8mf2(v3354, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3356 = __riscv_vzext_vf2_u16m1(v3355, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3357 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3356, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3358 = __riscv_vsrl_vx_u8mf2(v3354, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3359 = __riscv_vzext_vf2_u16m1(v3358, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3360 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3359, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3361 = v21 + 146;
      const int8_t* v3362 = (const int8_t*) v3361;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3363 = *(const int8_t *)(v3362);
      const uint8_t* v3364 = v21 + 162;
      const int8_t* v3365 = (const int8_t*) v3364;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3366 = *(const int8_t *)(v3365);
      vint32m2_t v3367 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3368 = __riscv_vwmul_vx_i16m1(v3357, v3363, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3369 = __riscv_vwadd_wv_i32m2(v3367, v3368, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3370 = __riscv_vwmul_vx_i16m1(v3360, v3366, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3371 = __riscv_vwadd_wv_i32m2(v3369, v3370, 8);
      v2770 = v3371;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3372 = v19 + 1392;
      const uint8_t* v3373 = (const uint8_t*) v3372;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3374 = __riscv_vle8_v_u8mf2(v3373, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3375 = __riscv_vand_vx_u8mf2(v3374, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3376 = __riscv_vzext_vf2_u16m1(v3375, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3377 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3376, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3378 = __riscv_vsrl_vx_u8mf2(v3374, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3379 = __riscv_vzext_vf2_u16m1(v3378, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3380 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3379, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3381 = v21 + 147;
      const int8_t* v3382 = (const int8_t*) v3381;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3383 = *(const int8_t *)(v3382);
      const uint8_t* v3384 = v21 + 163;
      const int8_t* v3385 = (const int8_t*) v3384;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3386 = *(const int8_t *)(v3385);
      vint32m2_t v3387 = v2768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3388 = __riscv_vwmul_vx_i16m1(v3377, v3383, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3389 = __riscv_vwadd_wv_i32m2(v3387, v3388, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3390 = __riscv_vwmul_vx_i16m1(v3380, v3386, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3391 = __riscv_vwadd_wv_i32m2(v3389, v3390, 8);
      v2768 = v3391;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3392 = v19 + 1400;
      const uint8_t* v3393 = (const uint8_t*) v3392;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3394 = __riscv_vle8_v_u8mf2(v3393, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3395 = __riscv_vand_vx_u8mf2(v3394, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3396 = __riscv_vzext_vf2_u16m1(v3395, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3397 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3396, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3398 = __riscv_vsrl_vx_u8mf2(v3394, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3399 = __riscv_vzext_vf2_u16m1(v3398, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3400 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3399, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3401 = v21 + 147;
      const int8_t* v3402 = (const int8_t*) v3401;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3403 = *(const int8_t *)(v3402);
      const uint8_t* v3404 = v21 + 163;
      const int8_t* v3405 = (const int8_t*) v3404;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3406 = *(const int8_t *)(v3405);
      vint32m2_t v3407 = v2770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3408 = __riscv_vwmul_vx_i16m1(v3397, v3403, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3409 = __riscv_vwadd_wv_i32m2(v3407, v3408, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3410 = __riscv_vwmul_vx_i16m1(v3400, v3406, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3411 = __riscv_vwadd_wv_i32m2(v3409, v3410, 8);
      v2770 = v3411;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_scale_fold
      vint32m2_t v3412 = v2768;
      vint32m2_t v3413 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v3414 = __riscv_vmacc_vv_i32m2(v3413, v2753, v3412, 8);
      v24 = v3414;
      vint32m2_t v3415 = v2770;
      vint32m2_t v3416 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v3417 = __riscv_vmacc_vv_i32m2(v3416, v2767, v3415, 8);
      v26 = v3417;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_signed_scale
      const uint8_t* v3418 = v19 + 96;
      const uint8_t* v3419 = (const uint8_t*) v3418;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3420 = __riscv_vle8_v_u8mf2(v3419, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3421 = __riscv_vsrl_vx_u8mf2(v3420, 4, 8);
      const uint8_t* v3422 = v19 + 48;
      const uint8_t* v3423 = (const uint8_t*) v3422;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3424 = __riscv_vle8_v_u8mf2(v3423, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3425 = __riscv_vsrl_vx_u8mf2(v3424, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3426 = __riscv_vand_vx_u8mf2(v3425, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v3427 = __riscv_vsll_vx_u8mf2(v3426, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v3428 = __riscv_vor_vv_u8mf2(v3427, v3421, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3429 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3428);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v3430 = __riscv_vsub_vx_i8mf2(v3429, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v3431 = __riscv_vsext_vf4_i32m2(v3430, 8);
      const uint8_t* v3432 = v19 + 104;
      const uint8_t* v3433 = (const uint8_t*) v3432;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3434 = __riscv_vle8_v_u8mf2(v3433, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3435 = __riscv_vsrl_vx_u8mf2(v3434, 4, 8);
      const uint8_t* v3436 = v19 + 56;
      const uint8_t* v3437 = (const uint8_t*) v3436;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3438 = __riscv_vle8_v_u8mf2(v3437, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3439 = __riscv_vsrl_vx_u8mf2(v3438, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3440 = __riscv_vand_vx_u8mf2(v3439, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v3441 = __riscv_vsll_vx_u8mf2(v3440, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v3442 = __riscv_vor_vv_u8mf2(v3441, v3435, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3443 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3442);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v3444 = __riscv_vsub_vx_i8mf2(v3443, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v3445 = __riscv_vsext_vf4_i32m2(v3444, 8);
      vint32m2_t v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v3447 = __riscv_vmv_v_x_i32m2(0, 8);
      v3446 = v3447;
      vint32m2_t v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v3449 = __riscv_vmv_v_x_i32m2(0, 8);
      v3448 = v3449;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3450 = v19 + 1408;
      const uint8_t* v3451 = (const uint8_t*) v3450;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3452 = __riscv_vle8_v_u8mf2(v3451, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3453 = __riscv_vand_vx_u8mf2(v3452, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3454 = __riscv_vzext_vf2_u16m1(v3453, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3455 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3454, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3456 = __riscv_vsrl_vx_u8mf2(v3452, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3457 = __riscv_vzext_vf2_u16m1(v3456, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3458 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3457, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3459 = v21 + 164;
      const int8_t* v3460 = (const int8_t*) v3459;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3461 = *(const int8_t *)(v3460);
      const uint8_t* v3462 = v21 + 180;
      const int8_t* v3463 = (const int8_t*) v3462;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3464 = *(const int8_t *)(v3463);
      vint32m2_t v3465 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3466 = __riscv_vwmul_vx_i16m1(v3455, v3461, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3467 = __riscv_vwadd_wv_i32m2(v3465, v3466, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3468 = __riscv_vwmul_vx_i16m1(v3458, v3464, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3469 = __riscv_vwadd_wv_i32m2(v3467, v3468, 8);
      v3446 = v3469;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3470 = v19 + 1416;
      const uint8_t* v3471 = (const uint8_t*) v3470;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3472 = __riscv_vle8_v_u8mf2(v3471, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3473 = __riscv_vand_vx_u8mf2(v3472, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3474 = __riscv_vzext_vf2_u16m1(v3473, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3475 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3474, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3476 = __riscv_vsrl_vx_u8mf2(v3472, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3477 = __riscv_vzext_vf2_u16m1(v3476, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3478 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3477, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3479 = v21 + 164;
      const int8_t* v3480 = (const int8_t*) v3479;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3481 = *(const int8_t *)(v3480);
      const uint8_t* v3482 = v21 + 180;
      const int8_t* v3483 = (const int8_t*) v3482;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3484 = *(const int8_t *)(v3483);
      vint32m2_t v3485 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3486 = __riscv_vwmul_vx_i16m1(v3475, v3481, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3487 = __riscv_vwadd_wv_i32m2(v3485, v3486, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3488 = __riscv_vwmul_vx_i16m1(v3478, v3484, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3489 = __riscv_vwadd_wv_i32m2(v3487, v3488, 8);
      v3448 = v3489;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3490 = v19 + 1424;
      const uint8_t* v3491 = (const uint8_t*) v3490;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3492 = __riscv_vle8_v_u8mf2(v3491, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3493 = __riscv_vand_vx_u8mf2(v3492, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3494 = __riscv_vzext_vf2_u16m1(v3493, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3495 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3494, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3496 = __riscv_vsrl_vx_u8mf2(v3492, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3497 = __riscv_vzext_vf2_u16m1(v3496, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3498 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3497, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3499 = v21 + 165;
      const int8_t* v3500 = (const int8_t*) v3499;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3501 = *(const int8_t *)(v3500);
      const uint8_t* v3502 = v21 + 181;
      const int8_t* v3503 = (const int8_t*) v3502;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3504 = *(const int8_t *)(v3503);
      vint32m2_t v3505 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3506 = __riscv_vwmul_vx_i16m1(v3495, v3501, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3507 = __riscv_vwadd_wv_i32m2(v3505, v3506, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3508 = __riscv_vwmul_vx_i16m1(v3498, v3504, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3509 = __riscv_vwadd_wv_i32m2(v3507, v3508, 8);
      v3446 = v3509;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3510 = v19 + 1432;
      const uint8_t* v3511 = (const uint8_t*) v3510;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3512 = __riscv_vle8_v_u8mf2(v3511, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3513 = __riscv_vand_vx_u8mf2(v3512, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3514 = __riscv_vzext_vf2_u16m1(v3513, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3515 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3514, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3516 = __riscv_vsrl_vx_u8mf2(v3512, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3517 = __riscv_vzext_vf2_u16m1(v3516, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3518 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3517, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3519 = v21 + 165;
      const int8_t* v3520 = (const int8_t*) v3519;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3521 = *(const int8_t *)(v3520);
      const uint8_t* v3522 = v21 + 181;
      const int8_t* v3523 = (const int8_t*) v3522;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3524 = *(const int8_t *)(v3523);
      vint32m2_t v3525 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3526 = __riscv_vwmul_vx_i16m1(v3515, v3521, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3527 = __riscv_vwadd_wv_i32m2(v3525, v3526, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3528 = __riscv_vwmul_vx_i16m1(v3518, v3524, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3529 = __riscv_vwadd_wv_i32m2(v3527, v3528, 8);
      v3448 = v3529;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3530 = v19 + 1440;
      const uint8_t* v3531 = (const uint8_t*) v3530;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3532 = __riscv_vle8_v_u8mf2(v3531, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3533 = __riscv_vand_vx_u8mf2(v3532, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3534 = __riscv_vzext_vf2_u16m1(v3533, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3535 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3534, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3536 = __riscv_vsrl_vx_u8mf2(v3532, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3537 = __riscv_vzext_vf2_u16m1(v3536, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3538 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3537, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3539 = v21 + 166;
      const int8_t* v3540 = (const int8_t*) v3539;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3541 = *(const int8_t *)(v3540);
      const uint8_t* v3542 = v21 + 182;
      const int8_t* v3543 = (const int8_t*) v3542;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3544 = *(const int8_t *)(v3543);
      vint32m2_t v3545 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3546 = __riscv_vwmul_vx_i16m1(v3535, v3541, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3547 = __riscv_vwadd_wv_i32m2(v3545, v3546, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3548 = __riscv_vwmul_vx_i16m1(v3538, v3544, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3549 = __riscv_vwadd_wv_i32m2(v3547, v3548, 8);
      v3446 = v3549;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3550 = v19 + 1448;
      const uint8_t* v3551 = (const uint8_t*) v3550;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3552 = __riscv_vle8_v_u8mf2(v3551, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3553 = __riscv_vand_vx_u8mf2(v3552, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3554 = __riscv_vzext_vf2_u16m1(v3553, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3555 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3554, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3556 = __riscv_vsrl_vx_u8mf2(v3552, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3557 = __riscv_vzext_vf2_u16m1(v3556, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3558 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3557, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3559 = v21 + 166;
      const int8_t* v3560 = (const int8_t*) v3559;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3561 = *(const int8_t *)(v3560);
      const uint8_t* v3562 = v21 + 182;
      const int8_t* v3563 = (const int8_t*) v3562;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3564 = *(const int8_t *)(v3563);
      vint32m2_t v3565 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3566 = __riscv_vwmul_vx_i16m1(v3555, v3561, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3567 = __riscv_vwadd_wv_i32m2(v3565, v3566, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3568 = __riscv_vwmul_vx_i16m1(v3558, v3564, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3569 = __riscv_vwadd_wv_i32m2(v3567, v3568, 8);
      v3448 = v3569;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3570 = v19 + 1456;
      const uint8_t* v3571 = (const uint8_t*) v3570;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3572 = __riscv_vle8_v_u8mf2(v3571, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3573 = __riscv_vand_vx_u8mf2(v3572, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3574 = __riscv_vzext_vf2_u16m1(v3573, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3575 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3574, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3576 = __riscv_vsrl_vx_u8mf2(v3572, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3577 = __riscv_vzext_vf2_u16m1(v3576, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3578 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3577, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3579 = v21 + 167;
      const int8_t* v3580 = (const int8_t*) v3579;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3581 = *(const int8_t *)(v3580);
      const uint8_t* v3582 = v21 + 183;
      const int8_t* v3583 = (const int8_t*) v3582;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3584 = *(const int8_t *)(v3583);
      vint32m2_t v3585 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3586 = __riscv_vwmul_vx_i16m1(v3575, v3581, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3587 = __riscv_vwadd_wv_i32m2(v3585, v3586, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3588 = __riscv_vwmul_vx_i16m1(v3578, v3584, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3589 = __riscv_vwadd_wv_i32m2(v3587, v3588, 8);
      v3446 = v3589;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3590 = v19 + 1464;
      const uint8_t* v3591 = (const uint8_t*) v3590;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3592 = __riscv_vle8_v_u8mf2(v3591, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3593 = __riscv_vand_vx_u8mf2(v3592, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3594 = __riscv_vzext_vf2_u16m1(v3593, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3595 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3594, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3596 = __riscv_vsrl_vx_u8mf2(v3592, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3597 = __riscv_vzext_vf2_u16m1(v3596, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3598 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3597, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3599 = v21 + 167;
      const int8_t* v3600 = (const int8_t*) v3599;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3601 = *(const int8_t *)(v3600);
      const uint8_t* v3602 = v21 + 183;
      const int8_t* v3603 = (const int8_t*) v3602;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3604 = *(const int8_t *)(v3603);
      vint32m2_t v3605 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3606 = __riscv_vwmul_vx_i16m1(v3595, v3601, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3607 = __riscv_vwadd_wv_i32m2(v3605, v3606, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3608 = __riscv_vwmul_vx_i16m1(v3598, v3604, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3609 = __riscv_vwadd_wv_i32m2(v3607, v3608, 8);
      v3448 = v3609;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3610 = v19 + 1472;
      const uint8_t* v3611 = (const uint8_t*) v3610;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3612 = __riscv_vle8_v_u8mf2(v3611, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3613 = __riscv_vand_vx_u8mf2(v3612, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3614 = __riscv_vzext_vf2_u16m1(v3613, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3615 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3614, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3616 = __riscv_vsrl_vx_u8mf2(v3612, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3617 = __riscv_vzext_vf2_u16m1(v3616, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3618 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3617, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3619 = v21 + 168;
      const int8_t* v3620 = (const int8_t*) v3619;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3621 = *(const int8_t *)(v3620);
      const uint8_t* v3622 = v21 + 184;
      const int8_t* v3623 = (const int8_t*) v3622;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3624 = *(const int8_t *)(v3623);
      vint32m2_t v3625 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3626 = __riscv_vwmul_vx_i16m1(v3615, v3621, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3627 = __riscv_vwadd_wv_i32m2(v3625, v3626, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3628 = __riscv_vwmul_vx_i16m1(v3618, v3624, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3629 = __riscv_vwadd_wv_i32m2(v3627, v3628, 8);
      v3446 = v3629;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3630 = v19 + 1480;
      const uint8_t* v3631 = (const uint8_t*) v3630;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3632 = __riscv_vle8_v_u8mf2(v3631, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3633 = __riscv_vand_vx_u8mf2(v3632, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3634 = __riscv_vzext_vf2_u16m1(v3633, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3635 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3634, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3636 = __riscv_vsrl_vx_u8mf2(v3632, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3637 = __riscv_vzext_vf2_u16m1(v3636, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3638 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3637, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3639 = v21 + 168;
      const int8_t* v3640 = (const int8_t*) v3639;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3641 = *(const int8_t *)(v3640);
      const uint8_t* v3642 = v21 + 184;
      const int8_t* v3643 = (const int8_t*) v3642;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3644 = *(const int8_t *)(v3643);
      vint32m2_t v3645 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3646 = __riscv_vwmul_vx_i16m1(v3635, v3641, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3647 = __riscv_vwadd_wv_i32m2(v3645, v3646, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3648 = __riscv_vwmul_vx_i16m1(v3638, v3644, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3649 = __riscv_vwadd_wv_i32m2(v3647, v3648, 8);
      v3448 = v3649;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3650 = v19 + 1488;
      const uint8_t* v3651 = (const uint8_t*) v3650;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3652 = __riscv_vle8_v_u8mf2(v3651, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3653 = __riscv_vand_vx_u8mf2(v3652, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3654 = __riscv_vzext_vf2_u16m1(v3653, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3655 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3654, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3656 = __riscv_vsrl_vx_u8mf2(v3652, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3657 = __riscv_vzext_vf2_u16m1(v3656, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3658 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3657, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3659 = v21 + 169;
      const int8_t* v3660 = (const int8_t*) v3659;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3661 = *(const int8_t *)(v3660);
      const uint8_t* v3662 = v21 + 185;
      const int8_t* v3663 = (const int8_t*) v3662;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3664 = *(const int8_t *)(v3663);
      vint32m2_t v3665 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3666 = __riscv_vwmul_vx_i16m1(v3655, v3661, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3667 = __riscv_vwadd_wv_i32m2(v3665, v3666, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3668 = __riscv_vwmul_vx_i16m1(v3658, v3664, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3669 = __riscv_vwadd_wv_i32m2(v3667, v3668, 8);
      v3446 = v3669;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3670 = v19 + 1496;
      const uint8_t* v3671 = (const uint8_t*) v3670;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3672 = __riscv_vle8_v_u8mf2(v3671, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3673 = __riscv_vand_vx_u8mf2(v3672, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3674 = __riscv_vzext_vf2_u16m1(v3673, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3675 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3674, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3676 = __riscv_vsrl_vx_u8mf2(v3672, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3677 = __riscv_vzext_vf2_u16m1(v3676, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3678 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3677, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3679 = v21 + 169;
      const int8_t* v3680 = (const int8_t*) v3679;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3681 = *(const int8_t *)(v3680);
      const uint8_t* v3682 = v21 + 185;
      const int8_t* v3683 = (const int8_t*) v3682;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3684 = *(const int8_t *)(v3683);
      vint32m2_t v3685 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3686 = __riscv_vwmul_vx_i16m1(v3675, v3681, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3687 = __riscv_vwadd_wv_i32m2(v3685, v3686, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3688 = __riscv_vwmul_vx_i16m1(v3678, v3684, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3689 = __riscv_vwadd_wv_i32m2(v3687, v3688, 8);
      v3448 = v3689;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3690 = v19 + 1504;
      const uint8_t* v3691 = (const uint8_t*) v3690;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3692 = __riscv_vle8_v_u8mf2(v3691, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3693 = __riscv_vand_vx_u8mf2(v3692, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3694 = __riscv_vzext_vf2_u16m1(v3693, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3695 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3694, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3696 = __riscv_vsrl_vx_u8mf2(v3692, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3697 = __riscv_vzext_vf2_u16m1(v3696, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3698 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3697, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3699 = v21 + 170;
      const int8_t* v3700 = (const int8_t*) v3699;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3701 = *(const int8_t *)(v3700);
      const uint8_t* v3702 = v21 + 186;
      const int8_t* v3703 = (const int8_t*) v3702;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3704 = *(const int8_t *)(v3703);
      vint32m2_t v3705 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3706 = __riscv_vwmul_vx_i16m1(v3695, v3701, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3707 = __riscv_vwadd_wv_i32m2(v3705, v3706, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3708 = __riscv_vwmul_vx_i16m1(v3698, v3704, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3709 = __riscv_vwadd_wv_i32m2(v3707, v3708, 8);
      v3446 = v3709;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3710 = v19 + 1512;
      const uint8_t* v3711 = (const uint8_t*) v3710;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3712 = __riscv_vle8_v_u8mf2(v3711, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3713 = __riscv_vand_vx_u8mf2(v3712, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3714 = __riscv_vzext_vf2_u16m1(v3713, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3715 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3714, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3716 = __riscv_vsrl_vx_u8mf2(v3712, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3717 = __riscv_vzext_vf2_u16m1(v3716, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3718 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3717, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3719 = v21 + 170;
      const int8_t* v3720 = (const int8_t*) v3719;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3721 = *(const int8_t *)(v3720);
      const uint8_t* v3722 = v21 + 186;
      const int8_t* v3723 = (const int8_t*) v3722;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3724 = *(const int8_t *)(v3723);
      vint32m2_t v3725 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3726 = __riscv_vwmul_vx_i16m1(v3715, v3721, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3727 = __riscv_vwadd_wv_i32m2(v3725, v3726, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3728 = __riscv_vwmul_vx_i16m1(v3718, v3724, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3729 = __riscv_vwadd_wv_i32m2(v3727, v3728, 8);
      v3448 = v3729;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3730 = v19 + 1520;
      const uint8_t* v3731 = (const uint8_t*) v3730;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3732 = __riscv_vle8_v_u8mf2(v3731, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3733 = __riscv_vand_vx_u8mf2(v3732, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3734 = __riscv_vzext_vf2_u16m1(v3733, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3735 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3734, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3736 = __riscv_vsrl_vx_u8mf2(v3732, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3737 = __riscv_vzext_vf2_u16m1(v3736, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3738 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3737, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3739 = v21 + 171;
      const int8_t* v3740 = (const int8_t*) v3739;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3741 = *(const int8_t *)(v3740);
      const uint8_t* v3742 = v21 + 187;
      const int8_t* v3743 = (const int8_t*) v3742;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3744 = *(const int8_t *)(v3743);
      vint32m2_t v3745 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3746 = __riscv_vwmul_vx_i16m1(v3735, v3741, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3747 = __riscv_vwadd_wv_i32m2(v3745, v3746, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3748 = __riscv_vwmul_vx_i16m1(v3738, v3744, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3749 = __riscv_vwadd_wv_i32m2(v3747, v3748, 8);
      v3446 = v3749;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3750 = v19 + 1528;
      const uint8_t* v3751 = (const uint8_t*) v3750;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3752 = __riscv_vle8_v_u8mf2(v3751, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3753 = __riscv_vand_vx_u8mf2(v3752, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3754 = __riscv_vzext_vf2_u16m1(v3753, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3755 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3754, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3756 = __riscv_vsrl_vx_u8mf2(v3752, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3757 = __riscv_vzext_vf2_u16m1(v3756, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3758 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3757, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3759 = v21 + 171;
      const int8_t* v3760 = (const int8_t*) v3759;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3761 = *(const int8_t *)(v3760);
      const uint8_t* v3762 = v21 + 187;
      const int8_t* v3763 = (const int8_t*) v3762;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3764 = *(const int8_t *)(v3763);
      vint32m2_t v3765 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3766 = __riscv_vwmul_vx_i16m1(v3755, v3761, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3767 = __riscv_vwadd_wv_i32m2(v3765, v3766, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3768 = __riscv_vwmul_vx_i16m1(v3758, v3764, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3769 = __riscv_vwadd_wv_i32m2(v3767, v3768, 8);
      v3448 = v3769;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3770 = v19 + 1536;
      const uint8_t* v3771 = (const uint8_t*) v3770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3772 = __riscv_vle8_v_u8mf2(v3771, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3773 = __riscv_vand_vx_u8mf2(v3772, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3774 = __riscv_vzext_vf2_u16m1(v3773, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3775 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3774, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3776 = __riscv_vsrl_vx_u8mf2(v3772, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3777 = __riscv_vzext_vf2_u16m1(v3776, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3778 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3777, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3779 = v21 + 172;
      const int8_t* v3780 = (const int8_t*) v3779;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3781 = *(const int8_t *)(v3780);
      const uint8_t* v3782 = v21 + 188;
      const int8_t* v3783 = (const int8_t*) v3782;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3784 = *(const int8_t *)(v3783);
      vint32m2_t v3785 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3786 = __riscv_vwmul_vx_i16m1(v3775, v3781, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3787 = __riscv_vwadd_wv_i32m2(v3785, v3786, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3788 = __riscv_vwmul_vx_i16m1(v3778, v3784, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3789 = __riscv_vwadd_wv_i32m2(v3787, v3788, 8);
      v3446 = v3789;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3790 = v19 + 1544;
      const uint8_t* v3791 = (const uint8_t*) v3790;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3792 = __riscv_vle8_v_u8mf2(v3791, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3793 = __riscv_vand_vx_u8mf2(v3792, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3794 = __riscv_vzext_vf2_u16m1(v3793, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3795 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3794, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3796 = __riscv_vsrl_vx_u8mf2(v3792, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3797 = __riscv_vzext_vf2_u16m1(v3796, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3798 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3797, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3799 = v21 + 172;
      const int8_t* v3800 = (const int8_t*) v3799;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3801 = *(const int8_t *)(v3800);
      const uint8_t* v3802 = v21 + 188;
      const int8_t* v3803 = (const int8_t*) v3802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3804 = *(const int8_t *)(v3803);
      vint32m2_t v3805 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3806 = __riscv_vwmul_vx_i16m1(v3795, v3801, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3807 = __riscv_vwadd_wv_i32m2(v3805, v3806, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3808 = __riscv_vwmul_vx_i16m1(v3798, v3804, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3809 = __riscv_vwadd_wv_i32m2(v3807, v3808, 8);
      v3448 = v3809;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3810 = v19 + 1552;
      const uint8_t* v3811 = (const uint8_t*) v3810;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3812 = __riscv_vle8_v_u8mf2(v3811, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3813 = __riscv_vand_vx_u8mf2(v3812, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3814 = __riscv_vzext_vf2_u16m1(v3813, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3815 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3814, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3816 = __riscv_vsrl_vx_u8mf2(v3812, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3817 = __riscv_vzext_vf2_u16m1(v3816, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3818 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3817, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3819 = v21 + 173;
      const int8_t* v3820 = (const int8_t*) v3819;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3821 = *(const int8_t *)(v3820);
      const uint8_t* v3822 = v21 + 189;
      const int8_t* v3823 = (const int8_t*) v3822;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3824 = *(const int8_t *)(v3823);
      vint32m2_t v3825 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3826 = __riscv_vwmul_vx_i16m1(v3815, v3821, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3827 = __riscv_vwadd_wv_i32m2(v3825, v3826, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3828 = __riscv_vwmul_vx_i16m1(v3818, v3824, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3829 = __riscv_vwadd_wv_i32m2(v3827, v3828, 8);
      v3446 = v3829;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3830 = v19 + 1560;
      const uint8_t* v3831 = (const uint8_t*) v3830;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3832 = __riscv_vle8_v_u8mf2(v3831, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3833 = __riscv_vand_vx_u8mf2(v3832, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3834 = __riscv_vzext_vf2_u16m1(v3833, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3835 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3834, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3836 = __riscv_vsrl_vx_u8mf2(v3832, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3837 = __riscv_vzext_vf2_u16m1(v3836, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3838 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3837, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3839 = v21 + 173;
      const int8_t* v3840 = (const int8_t*) v3839;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3841 = *(const int8_t *)(v3840);
      const uint8_t* v3842 = v21 + 189;
      const int8_t* v3843 = (const int8_t*) v3842;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3844 = *(const int8_t *)(v3843);
      vint32m2_t v3845 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3846 = __riscv_vwmul_vx_i16m1(v3835, v3841, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3847 = __riscv_vwadd_wv_i32m2(v3845, v3846, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3848 = __riscv_vwmul_vx_i16m1(v3838, v3844, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3849 = __riscv_vwadd_wv_i32m2(v3847, v3848, 8);
      v3448 = v3849;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3850 = v19 + 1568;
      const uint8_t* v3851 = (const uint8_t*) v3850;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3852 = __riscv_vle8_v_u8mf2(v3851, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3853 = __riscv_vand_vx_u8mf2(v3852, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3854 = __riscv_vzext_vf2_u16m1(v3853, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3855 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3854, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3856 = __riscv_vsrl_vx_u8mf2(v3852, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3857 = __riscv_vzext_vf2_u16m1(v3856, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3858 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3857, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3859 = v21 + 174;
      const int8_t* v3860 = (const int8_t*) v3859;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3861 = *(const int8_t *)(v3860);
      const uint8_t* v3862 = v21 + 190;
      const int8_t* v3863 = (const int8_t*) v3862;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3864 = *(const int8_t *)(v3863);
      vint32m2_t v3865 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3866 = __riscv_vwmul_vx_i16m1(v3855, v3861, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3867 = __riscv_vwadd_wv_i32m2(v3865, v3866, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3868 = __riscv_vwmul_vx_i16m1(v3858, v3864, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3869 = __riscv_vwadd_wv_i32m2(v3867, v3868, 8);
      v3446 = v3869;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3870 = v19 + 1576;
      const uint8_t* v3871 = (const uint8_t*) v3870;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3872 = __riscv_vle8_v_u8mf2(v3871, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3873 = __riscv_vand_vx_u8mf2(v3872, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3874 = __riscv_vzext_vf2_u16m1(v3873, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3875 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3874, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3876 = __riscv_vsrl_vx_u8mf2(v3872, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3877 = __riscv_vzext_vf2_u16m1(v3876, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3878 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3877, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3879 = v21 + 174;
      const int8_t* v3880 = (const int8_t*) v3879;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3881 = *(const int8_t *)(v3880);
      const uint8_t* v3882 = v21 + 190;
      const int8_t* v3883 = (const int8_t*) v3882;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3884 = *(const int8_t *)(v3883);
      vint32m2_t v3885 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3886 = __riscv_vwmul_vx_i16m1(v3875, v3881, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3887 = __riscv_vwadd_wv_i32m2(v3885, v3886, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3888 = __riscv_vwmul_vx_i16m1(v3878, v3884, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3889 = __riscv_vwadd_wv_i32m2(v3887, v3888, 8);
      v3448 = v3889;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3890 = v19 + 1584;
      const uint8_t* v3891 = (const uint8_t*) v3890;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3892 = __riscv_vle8_v_u8mf2(v3891, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3893 = __riscv_vand_vx_u8mf2(v3892, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3894 = __riscv_vzext_vf2_u16m1(v3893, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3895 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3894, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3896 = __riscv_vsrl_vx_u8mf2(v3892, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3897 = __riscv_vzext_vf2_u16m1(v3896, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3898 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3897, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3899 = v21 + 175;
      const int8_t* v3900 = (const int8_t*) v3899;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3901 = *(const int8_t *)(v3900);
      const uint8_t* v3902 = v21 + 191;
      const int8_t* v3903 = (const int8_t*) v3902;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3904 = *(const int8_t *)(v3903);
      vint32m2_t v3905 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3906 = __riscv_vwmul_vx_i16m1(v3895, v3901, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3907 = __riscv_vwadd_wv_i32m2(v3905, v3906, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3908 = __riscv_vwmul_vx_i16m1(v3898, v3904, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3909 = __riscv_vwadd_wv_i32m2(v3907, v3908, 8);
      v3446 = v3909;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3910 = v19 + 1592;
      const uint8_t* v3911 = (const uint8_t*) v3910;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3912 = __riscv_vle8_v_u8mf2(v3911, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3913 = __riscv_vand_vx_u8mf2(v3912, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3914 = __riscv_vzext_vf2_u16m1(v3913, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3915 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3914, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3916 = __riscv_vsrl_vx_u8mf2(v3912, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3917 = __riscv_vzext_vf2_u16m1(v3916, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3918 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3917, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3919 = v21 + 175;
      const int8_t* v3920 = (const int8_t*) v3919;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3921 = *(const int8_t *)(v3920);
      const uint8_t* v3922 = v21 + 191;
      const int8_t* v3923 = (const int8_t*) v3922;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3924 = *(const int8_t *)(v3923);
      vint32m2_t v3925 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3926 = __riscv_vwmul_vx_i16m1(v3915, v3921, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3927 = __riscv_vwadd_wv_i32m2(v3925, v3926, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3928 = __riscv_vwmul_vx_i16m1(v3918, v3924, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3929 = __riscv_vwadd_wv_i32m2(v3927, v3928, 8);
      v3448 = v3929;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3930 = v19 + 1600;
      const uint8_t* v3931 = (const uint8_t*) v3930;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3932 = __riscv_vle8_v_u8mf2(v3931, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3933 = __riscv_vand_vx_u8mf2(v3932, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3934 = __riscv_vzext_vf2_u16m1(v3933, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3935 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3934, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3936 = __riscv_vsrl_vx_u8mf2(v3932, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3937 = __riscv_vzext_vf2_u16m1(v3936, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3938 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3937, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3939 = v21 + 176;
      const int8_t* v3940 = (const int8_t*) v3939;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3941 = *(const int8_t *)(v3940);
      const uint8_t* v3942 = v21 + 192;
      const int8_t* v3943 = (const int8_t*) v3942;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3944 = *(const int8_t *)(v3943);
      vint32m2_t v3945 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3946 = __riscv_vwmul_vx_i16m1(v3935, v3941, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3947 = __riscv_vwadd_wv_i32m2(v3945, v3946, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3948 = __riscv_vwmul_vx_i16m1(v3938, v3944, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3949 = __riscv_vwadd_wv_i32m2(v3947, v3948, 8);
      v3446 = v3949;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3950 = v19 + 1608;
      const uint8_t* v3951 = (const uint8_t*) v3950;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3952 = __riscv_vle8_v_u8mf2(v3951, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3953 = __riscv_vand_vx_u8mf2(v3952, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3954 = __riscv_vzext_vf2_u16m1(v3953, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3955 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3954, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3956 = __riscv_vsrl_vx_u8mf2(v3952, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3957 = __riscv_vzext_vf2_u16m1(v3956, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3958 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3957, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3959 = v21 + 176;
      const int8_t* v3960 = (const int8_t*) v3959;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3961 = *(const int8_t *)(v3960);
      const uint8_t* v3962 = v21 + 192;
      const int8_t* v3963 = (const int8_t*) v3962;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3964 = *(const int8_t *)(v3963);
      vint32m2_t v3965 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3966 = __riscv_vwmul_vx_i16m1(v3955, v3961, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3967 = __riscv_vwadd_wv_i32m2(v3965, v3966, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3968 = __riscv_vwmul_vx_i16m1(v3958, v3964, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3969 = __riscv_vwadd_wv_i32m2(v3967, v3968, 8);
      v3448 = v3969;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3970 = v19 + 1616;
      const uint8_t* v3971 = (const uint8_t*) v3970;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3972 = __riscv_vle8_v_u8mf2(v3971, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3973 = __riscv_vand_vx_u8mf2(v3972, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3974 = __riscv_vzext_vf2_u16m1(v3973, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3975 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3974, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3976 = __riscv_vsrl_vx_u8mf2(v3972, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3977 = __riscv_vzext_vf2_u16m1(v3976, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3978 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3977, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3979 = v21 + 177;
      const int8_t* v3980 = (const int8_t*) v3979;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3981 = *(const int8_t *)(v3980);
      const uint8_t* v3982 = v21 + 193;
      const int8_t* v3983 = (const int8_t*) v3982;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3984 = *(const int8_t *)(v3983);
      vint32m2_t v3985 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3986 = __riscv_vwmul_vx_i16m1(v3975, v3981, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3987 = __riscv_vwadd_wv_i32m2(v3985, v3986, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v3988 = __riscv_vwmul_vx_i16m1(v3978, v3984, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v3989 = __riscv_vwadd_wv_i32m2(v3987, v3988, 8);
      v3446 = v3989;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3990 = v19 + 1624;
      const uint8_t* v3991 = (const uint8_t*) v3990;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3992 = __riscv_vle8_v_u8mf2(v3991, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3993 = __riscv_vand_vx_u8mf2(v3992, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3994 = __riscv_vzext_vf2_u16m1(v3993, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3995 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3994, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3996 = __riscv_vsrl_vx_u8mf2(v3992, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v3997 = __riscv_vzext_vf2_u16m1(v3996, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v3998 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v3997, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3999 = v21 + 177;
      const int8_t* v4000 = (const int8_t*) v3999;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4001 = *(const int8_t *)(v4000);
      const uint8_t* v4002 = v21 + 193;
      const int8_t* v4003 = (const int8_t*) v4002;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4004 = *(const int8_t *)(v4003);
      vint32m2_t v4005 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4006 = __riscv_vwmul_vx_i16m1(v3995, v4001, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4007 = __riscv_vwadd_wv_i32m2(v4005, v4006, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4008 = __riscv_vwmul_vx_i16m1(v3998, v4004, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4009 = __riscv_vwadd_wv_i32m2(v4007, v4008, 8);
      v3448 = v4009;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4010 = v19 + 1632;
      const uint8_t* v4011 = (const uint8_t*) v4010;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4012 = __riscv_vle8_v_u8mf2(v4011, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4013 = __riscv_vand_vx_u8mf2(v4012, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4014 = __riscv_vzext_vf2_u16m1(v4013, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4015 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4014, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4016 = __riscv_vsrl_vx_u8mf2(v4012, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4017 = __riscv_vzext_vf2_u16m1(v4016, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4018 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4017, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4019 = v21 + 178;
      const int8_t* v4020 = (const int8_t*) v4019;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4021 = *(const int8_t *)(v4020);
      const uint8_t* v4022 = v21 + 194;
      const int8_t* v4023 = (const int8_t*) v4022;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4024 = *(const int8_t *)(v4023);
      vint32m2_t v4025 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4026 = __riscv_vwmul_vx_i16m1(v4015, v4021, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4027 = __riscv_vwadd_wv_i32m2(v4025, v4026, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4028 = __riscv_vwmul_vx_i16m1(v4018, v4024, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4029 = __riscv_vwadd_wv_i32m2(v4027, v4028, 8);
      v3446 = v4029;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4030 = v19 + 1640;
      const uint8_t* v4031 = (const uint8_t*) v4030;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4032 = __riscv_vle8_v_u8mf2(v4031, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4033 = __riscv_vand_vx_u8mf2(v4032, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4034 = __riscv_vzext_vf2_u16m1(v4033, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4035 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4034, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4036 = __riscv_vsrl_vx_u8mf2(v4032, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4037 = __riscv_vzext_vf2_u16m1(v4036, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4038 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4037, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4039 = v21 + 178;
      const int8_t* v4040 = (const int8_t*) v4039;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4041 = *(const int8_t *)(v4040);
      const uint8_t* v4042 = v21 + 194;
      const int8_t* v4043 = (const int8_t*) v4042;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4044 = *(const int8_t *)(v4043);
      vint32m2_t v4045 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4046 = __riscv_vwmul_vx_i16m1(v4035, v4041, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4047 = __riscv_vwadd_wv_i32m2(v4045, v4046, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4048 = __riscv_vwmul_vx_i16m1(v4038, v4044, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4049 = __riscv_vwadd_wv_i32m2(v4047, v4048, 8);
      v3448 = v4049;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4050 = v19 + 1648;
      const uint8_t* v4051 = (const uint8_t*) v4050;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4052 = __riscv_vle8_v_u8mf2(v4051, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4053 = __riscv_vand_vx_u8mf2(v4052, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4054 = __riscv_vzext_vf2_u16m1(v4053, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4055 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4054, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4056 = __riscv_vsrl_vx_u8mf2(v4052, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4057 = __riscv_vzext_vf2_u16m1(v4056, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4058 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4057, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4059 = v21 + 179;
      const int8_t* v4060 = (const int8_t*) v4059;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4061 = *(const int8_t *)(v4060);
      const uint8_t* v4062 = v21 + 195;
      const int8_t* v4063 = (const int8_t*) v4062;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4064 = *(const int8_t *)(v4063);
      vint32m2_t v4065 = v3446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4066 = __riscv_vwmul_vx_i16m1(v4055, v4061, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4067 = __riscv_vwadd_wv_i32m2(v4065, v4066, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4068 = __riscv_vwmul_vx_i16m1(v4058, v4064, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4069 = __riscv_vwadd_wv_i32m2(v4067, v4068, 8);
      v3446 = v4069;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4070 = v19 + 1656;
      const uint8_t* v4071 = (const uint8_t*) v4070;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4072 = __riscv_vle8_v_u8mf2(v4071, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4073 = __riscv_vand_vx_u8mf2(v4072, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4074 = __riscv_vzext_vf2_u16m1(v4073, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4075 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4074, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4076 = __riscv_vsrl_vx_u8mf2(v4072, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4077 = __riscv_vzext_vf2_u16m1(v4076, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4078 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4077, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4079 = v21 + 179;
      const int8_t* v4080 = (const int8_t*) v4079;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4081 = *(const int8_t *)(v4080);
      const uint8_t* v4082 = v21 + 195;
      const int8_t* v4083 = (const int8_t*) v4082;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4084 = *(const int8_t *)(v4083);
      vint32m2_t v4085 = v3448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4086 = __riscv_vwmul_vx_i16m1(v4075, v4081, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4087 = __riscv_vwadd_wv_i32m2(v4085, v4086, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4088 = __riscv_vwmul_vx_i16m1(v4078, v4084, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4089 = __riscv_vwadd_wv_i32m2(v4087, v4088, 8);
      v3448 = v4089;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_scale_fold
      vint32m2_t v4090 = v3446;
      vint32m2_t v4091 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v4092 = __riscv_vmacc_vv_i32m2(v4091, v3431, v4090, 8);
      v24 = v4092;
      vint32m2_t v4093 = v3448;
      vint32m2_t v4094 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v4095 = __riscv_vmacc_vv_i32m2(v4094, v3445, v4093, 8);
      v26 = v4095;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_signed_scale
      const uint8_t* v4096 = v19 + 112;
      const uint8_t* v4097 = (const uint8_t*) v4096;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4098 = __riscv_vle8_v_u8mf2(v4097, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4099 = __riscv_vand_vx_u8mf2(v4098, 0x0F, 8);
      const uint8_t* v4100 = v19 + 48;
      const uint8_t* v4101 = (const uint8_t*) v4100;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4102 = __riscv_vle8_v_u8mf2(v4101, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4103 = __riscv_vsrl_vx_u8mf2(v4102, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4104 = __riscv_vand_vx_u8mf2(v4103, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v4105 = __riscv_vsll_vx_u8mf2(v4104, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v4106 = __riscv_vor_vv_u8mf2(v4105, v4099, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4107 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4106);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v4108 = __riscv_vsub_vx_i8mf2(v4107, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v4109 = __riscv_vsext_vf4_i32m2(v4108, 8);
      const uint8_t* v4110 = v19 + 120;
      const uint8_t* v4111 = (const uint8_t*) v4110;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4112 = __riscv_vle8_v_u8mf2(v4111, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4113 = __riscv_vand_vx_u8mf2(v4112, 0x0F, 8);
      const uint8_t* v4114 = v19 + 56;
      const uint8_t* v4115 = (const uint8_t*) v4114;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4116 = __riscv_vle8_v_u8mf2(v4115, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4117 = __riscv_vsrl_vx_u8mf2(v4116, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4118 = __riscv_vand_vx_u8mf2(v4117, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v4119 = __riscv_vsll_vx_u8mf2(v4118, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v4120 = __riscv_vor_vv_u8mf2(v4119, v4113, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4121 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4120);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v4122 = __riscv_vsub_vx_i8mf2(v4121, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v4123 = __riscv_vsext_vf4_i32m2(v4122, 8);
      vint32m2_t v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v4125 = __riscv_vmv_v_x_i32m2(0, 8);
      v4124 = v4125;
      vint32m2_t v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v4127 = __riscv_vmv_v_x_i32m2(0, 8);
      v4126 = v4127;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4128 = v19 + 1664;
      const uint8_t* v4129 = (const uint8_t*) v4128;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4130 = __riscv_vle8_v_u8mf2(v4129, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4131 = __riscv_vand_vx_u8mf2(v4130, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4132 = __riscv_vzext_vf2_u16m1(v4131, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4133 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4132, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4134 = __riscv_vsrl_vx_u8mf2(v4130, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4135 = __riscv_vzext_vf2_u16m1(v4134, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4136 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4135, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4137 = v21 + 196;
      const int8_t* v4138 = (const int8_t*) v4137;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4139 = *(const int8_t *)(v4138);
      const uint8_t* v4140 = v21 + 212;
      const int8_t* v4141 = (const int8_t*) v4140;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4142 = *(const int8_t *)(v4141);
      vint32m2_t v4143 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4144 = __riscv_vwmul_vx_i16m1(v4133, v4139, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4145 = __riscv_vwadd_wv_i32m2(v4143, v4144, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4146 = __riscv_vwmul_vx_i16m1(v4136, v4142, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4147 = __riscv_vwadd_wv_i32m2(v4145, v4146, 8);
      v4124 = v4147;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4148 = v19 + 1672;
      const uint8_t* v4149 = (const uint8_t*) v4148;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4150 = __riscv_vle8_v_u8mf2(v4149, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4151 = __riscv_vand_vx_u8mf2(v4150, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4152 = __riscv_vzext_vf2_u16m1(v4151, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4153 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4152, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4154 = __riscv_vsrl_vx_u8mf2(v4150, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4155 = __riscv_vzext_vf2_u16m1(v4154, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4156 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4155, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4157 = v21 + 196;
      const int8_t* v4158 = (const int8_t*) v4157;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4159 = *(const int8_t *)(v4158);
      const uint8_t* v4160 = v21 + 212;
      const int8_t* v4161 = (const int8_t*) v4160;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4162 = *(const int8_t *)(v4161);
      vint32m2_t v4163 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4164 = __riscv_vwmul_vx_i16m1(v4153, v4159, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4165 = __riscv_vwadd_wv_i32m2(v4163, v4164, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4166 = __riscv_vwmul_vx_i16m1(v4156, v4162, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4167 = __riscv_vwadd_wv_i32m2(v4165, v4166, 8);
      v4126 = v4167;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4168 = v19 + 1680;
      const uint8_t* v4169 = (const uint8_t*) v4168;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4170 = __riscv_vle8_v_u8mf2(v4169, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4171 = __riscv_vand_vx_u8mf2(v4170, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4172 = __riscv_vzext_vf2_u16m1(v4171, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4173 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4172, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4174 = __riscv_vsrl_vx_u8mf2(v4170, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4175 = __riscv_vzext_vf2_u16m1(v4174, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4176 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4175, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4177 = v21 + 197;
      const int8_t* v4178 = (const int8_t*) v4177;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4179 = *(const int8_t *)(v4178);
      const uint8_t* v4180 = v21 + 213;
      const int8_t* v4181 = (const int8_t*) v4180;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4182 = *(const int8_t *)(v4181);
      vint32m2_t v4183 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4184 = __riscv_vwmul_vx_i16m1(v4173, v4179, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4185 = __riscv_vwadd_wv_i32m2(v4183, v4184, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4186 = __riscv_vwmul_vx_i16m1(v4176, v4182, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4187 = __riscv_vwadd_wv_i32m2(v4185, v4186, 8);
      v4124 = v4187;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4188 = v19 + 1688;
      const uint8_t* v4189 = (const uint8_t*) v4188;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4190 = __riscv_vle8_v_u8mf2(v4189, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4191 = __riscv_vand_vx_u8mf2(v4190, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4192 = __riscv_vzext_vf2_u16m1(v4191, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4193 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4192, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4194 = __riscv_vsrl_vx_u8mf2(v4190, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4195 = __riscv_vzext_vf2_u16m1(v4194, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4196 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4195, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4197 = v21 + 197;
      const int8_t* v4198 = (const int8_t*) v4197;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4199 = *(const int8_t *)(v4198);
      const uint8_t* v4200 = v21 + 213;
      const int8_t* v4201 = (const int8_t*) v4200;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4202 = *(const int8_t *)(v4201);
      vint32m2_t v4203 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4204 = __riscv_vwmul_vx_i16m1(v4193, v4199, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4205 = __riscv_vwadd_wv_i32m2(v4203, v4204, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4206 = __riscv_vwmul_vx_i16m1(v4196, v4202, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4207 = __riscv_vwadd_wv_i32m2(v4205, v4206, 8);
      v4126 = v4207;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4208 = v19 + 1696;
      const uint8_t* v4209 = (const uint8_t*) v4208;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4210 = __riscv_vle8_v_u8mf2(v4209, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4211 = __riscv_vand_vx_u8mf2(v4210, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4212 = __riscv_vzext_vf2_u16m1(v4211, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4213 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4212, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4214 = __riscv_vsrl_vx_u8mf2(v4210, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4215 = __riscv_vzext_vf2_u16m1(v4214, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4216 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4215, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4217 = v21 + 198;
      const int8_t* v4218 = (const int8_t*) v4217;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4219 = *(const int8_t *)(v4218);
      const uint8_t* v4220 = v21 + 214;
      const int8_t* v4221 = (const int8_t*) v4220;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4222 = *(const int8_t *)(v4221);
      vint32m2_t v4223 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4224 = __riscv_vwmul_vx_i16m1(v4213, v4219, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4225 = __riscv_vwadd_wv_i32m2(v4223, v4224, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4226 = __riscv_vwmul_vx_i16m1(v4216, v4222, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4227 = __riscv_vwadd_wv_i32m2(v4225, v4226, 8);
      v4124 = v4227;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4228 = v19 + 1704;
      const uint8_t* v4229 = (const uint8_t*) v4228;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4230 = __riscv_vle8_v_u8mf2(v4229, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4231 = __riscv_vand_vx_u8mf2(v4230, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4232 = __riscv_vzext_vf2_u16m1(v4231, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4233 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4232, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4234 = __riscv_vsrl_vx_u8mf2(v4230, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4235 = __riscv_vzext_vf2_u16m1(v4234, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4236 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4235, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4237 = v21 + 198;
      const int8_t* v4238 = (const int8_t*) v4237;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4239 = *(const int8_t *)(v4238);
      const uint8_t* v4240 = v21 + 214;
      const int8_t* v4241 = (const int8_t*) v4240;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4242 = *(const int8_t *)(v4241);
      vint32m2_t v4243 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4244 = __riscv_vwmul_vx_i16m1(v4233, v4239, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4245 = __riscv_vwadd_wv_i32m2(v4243, v4244, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4246 = __riscv_vwmul_vx_i16m1(v4236, v4242, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4247 = __riscv_vwadd_wv_i32m2(v4245, v4246, 8);
      v4126 = v4247;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4248 = v19 + 1712;
      const uint8_t* v4249 = (const uint8_t*) v4248;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4250 = __riscv_vle8_v_u8mf2(v4249, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4251 = __riscv_vand_vx_u8mf2(v4250, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4252 = __riscv_vzext_vf2_u16m1(v4251, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4253 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4252, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4254 = __riscv_vsrl_vx_u8mf2(v4250, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4255 = __riscv_vzext_vf2_u16m1(v4254, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4256 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4255, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4257 = v21 + 199;
      const int8_t* v4258 = (const int8_t*) v4257;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4259 = *(const int8_t *)(v4258);
      const uint8_t* v4260 = v21 + 215;
      const int8_t* v4261 = (const int8_t*) v4260;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4262 = *(const int8_t *)(v4261);
      vint32m2_t v4263 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4264 = __riscv_vwmul_vx_i16m1(v4253, v4259, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4265 = __riscv_vwadd_wv_i32m2(v4263, v4264, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4266 = __riscv_vwmul_vx_i16m1(v4256, v4262, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4267 = __riscv_vwadd_wv_i32m2(v4265, v4266, 8);
      v4124 = v4267;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4268 = v19 + 1720;
      const uint8_t* v4269 = (const uint8_t*) v4268;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4270 = __riscv_vle8_v_u8mf2(v4269, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4271 = __riscv_vand_vx_u8mf2(v4270, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4272 = __riscv_vzext_vf2_u16m1(v4271, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4273 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4272, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4274 = __riscv_vsrl_vx_u8mf2(v4270, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4275 = __riscv_vzext_vf2_u16m1(v4274, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4276 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4275, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4277 = v21 + 199;
      const int8_t* v4278 = (const int8_t*) v4277;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4279 = *(const int8_t *)(v4278);
      const uint8_t* v4280 = v21 + 215;
      const int8_t* v4281 = (const int8_t*) v4280;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4282 = *(const int8_t *)(v4281);
      vint32m2_t v4283 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4284 = __riscv_vwmul_vx_i16m1(v4273, v4279, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4285 = __riscv_vwadd_wv_i32m2(v4283, v4284, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4286 = __riscv_vwmul_vx_i16m1(v4276, v4282, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4287 = __riscv_vwadd_wv_i32m2(v4285, v4286, 8);
      v4126 = v4287;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4288 = v19 + 1728;
      const uint8_t* v4289 = (const uint8_t*) v4288;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4290 = __riscv_vle8_v_u8mf2(v4289, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4291 = __riscv_vand_vx_u8mf2(v4290, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4292 = __riscv_vzext_vf2_u16m1(v4291, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4293 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4292, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4294 = __riscv_vsrl_vx_u8mf2(v4290, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4295 = __riscv_vzext_vf2_u16m1(v4294, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4296 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4295, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4297 = v21 + 200;
      const int8_t* v4298 = (const int8_t*) v4297;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4299 = *(const int8_t *)(v4298);
      const uint8_t* v4300 = v21 + 216;
      const int8_t* v4301 = (const int8_t*) v4300;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4302 = *(const int8_t *)(v4301);
      vint32m2_t v4303 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4304 = __riscv_vwmul_vx_i16m1(v4293, v4299, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4305 = __riscv_vwadd_wv_i32m2(v4303, v4304, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4306 = __riscv_vwmul_vx_i16m1(v4296, v4302, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4307 = __riscv_vwadd_wv_i32m2(v4305, v4306, 8);
      v4124 = v4307;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4308 = v19 + 1736;
      const uint8_t* v4309 = (const uint8_t*) v4308;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4310 = __riscv_vle8_v_u8mf2(v4309, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4311 = __riscv_vand_vx_u8mf2(v4310, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4312 = __riscv_vzext_vf2_u16m1(v4311, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4313 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4312, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4314 = __riscv_vsrl_vx_u8mf2(v4310, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4315 = __riscv_vzext_vf2_u16m1(v4314, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4316 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4315, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4317 = v21 + 200;
      const int8_t* v4318 = (const int8_t*) v4317;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4319 = *(const int8_t *)(v4318);
      const uint8_t* v4320 = v21 + 216;
      const int8_t* v4321 = (const int8_t*) v4320;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4322 = *(const int8_t *)(v4321);
      vint32m2_t v4323 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4324 = __riscv_vwmul_vx_i16m1(v4313, v4319, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4325 = __riscv_vwadd_wv_i32m2(v4323, v4324, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4326 = __riscv_vwmul_vx_i16m1(v4316, v4322, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4327 = __riscv_vwadd_wv_i32m2(v4325, v4326, 8);
      v4126 = v4327;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4328 = v19 + 1744;
      const uint8_t* v4329 = (const uint8_t*) v4328;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4330 = __riscv_vle8_v_u8mf2(v4329, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4331 = __riscv_vand_vx_u8mf2(v4330, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4332 = __riscv_vzext_vf2_u16m1(v4331, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4333 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4332, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4334 = __riscv_vsrl_vx_u8mf2(v4330, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4335 = __riscv_vzext_vf2_u16m1(v4334, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4336 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4335, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4337 = v21 + 201;
      const int8_t* v4338 = (const int8_t*) v4337;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4339 = *(const int8_t *)(v4338);
      const uint8_t* v4340 = v21 + 217;
      const int8_t* v4341 = (const int8_t*) v4340;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4342 = *(const int8_t *)(v4341);
      vint32m2_t v4343 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4344 = __riscv_vwmul_vx_i16m1(v4333, v4339, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4345 = __riscv_vwadd_wv_i32m2(v4343, v4344, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4346 = __riscv_vwmul_vx_i16m1(v4336, v4342, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4347 = __riscv_vwadd_wv_i32m2(v4345, v4346, 8);
      v4124 = v4347;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4348 = v19 + 1752;
      const uint8_t* v4349 = (const uint8_t*) v4348;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4350 = __riscv_vle8_v_u8mf2(v4349, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4351 = __riscv_vand_vx_u8mf2(v4350, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4352 = __riscv_vzext_vf2_u16m1(v4351, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4353 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4352, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4354 = __riscv_vsrl_vx_u8mf2(v4350, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4355 = __riscv_vzext_vf2_u16m1(v4354, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4356 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4355, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4357 = v21 + 201;
      const int8_t* v4358 = (const int8_t*) v4357;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4359 = *(const int8_t *)(v4358);
      const uint8_t* v4360 = v21 + 217;
      const int8_t* v4361 = (const int8_t*) v4360;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4362 = *(const int8_t *)(v4361);
      vint32m2_t v4363 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4364 = __riscv_vwmul_vx_i16m1(v4353, v4359, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4365 = __riscv_vwadd_wv_i32m2(v4363, v4364, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4366 = __riscv_vwmul_vx_i16m1(v4356, v4362, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4367 = __riscv_vwadd_wv_i32m2(v4365, v4366, 8);
      v4126 = v4367;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4368 = v19 + 1760;
      const uint8_t* v4369 = (const uint8_t*) v4368;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4370 = __riscv_vle8_v_u8mf2(v4369, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4371 = __riscv_vand_vx_u8mf2(v4370, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4372 = __riscv_vzext_vf2_u16m1(v4371, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4373 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4372, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4374 = __riscv_vsrl_vx_u8mf2(v4370, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4375 = __riscv_vzext_vf2_u16m1(v4374, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4376 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4375, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4377 = v21 + 202;
      const int8_t* v4378 = (const int8_t*) v4377;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4379 = *(const int8_t *)(v4378);
      const uint8_t* v4380 = v21 + 218;
      const int8_t* v4381 = (const int8_t*) v4380;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4382 = *(const int8_t *)(v4381);
      vint32m2_t v4383 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4384 = __riscv_vwmul_vx_i16m1(v4373, v4379, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4385 = __riscv_vwadd_wv_i32m2(v4383, v4384, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4386 = __riscv_vwmul_vx_i16m1(v4376, v4382, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4387 = __riscv_vwadd_wv_i32m2(v4385, v4386, 8);
      v4124 = v4387;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4388 = v19 + 1768;
      const uint8_t* v4389 = (const uint8_t*) v4388;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4390 = __riscv_vle8_v_u8mf2(v4389, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4391 = __riscv_vand_vx_u8mf2(v4390, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4392 = __riscv_vzext_vf2_u16m1(v4391, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4393 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4392, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4394 = __riscv_vsrl_vx_u8mf2(v4390, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4395 = __riscv_vzext_vf2_u16m1(v4394, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4396 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4395, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4397 = v21 + 202;
      const int8_t* v4398 = (const int8_t*) v4397;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4399 = *(const int8_t *)(v4398);
      const uint8_t* v4400 = v21 + 218;
      const int8_t* v4401 = (const int8_t*) v4400;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4402 = *(const int8_t *)(v4401);
      vint32m2_t v4403 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4404 = __riscv_vwmul_vx_i16m1(v4393, v4399, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4405 = __riscv_vwadd_wv_i32m2(v4403, v4404, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4406 = __riscv_vwmul_vx_i16m1(v4396, v4402, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4407 = __riscv_vwadd_wv_i32m2(v4405, v4406, 8);
      v4126 = v4407;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4408 = v19 + 1776;
      const uint8_t* v4409 = (const uint8_t*) v4408;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4410 = __riscv_vle8_v_u8mf2(v4409, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4411 = __riscv_vand_vx_u8mf2(v4410, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4412 = __riscv_vzext_vf2_u16m1(v4411, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4413 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4412, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4414 = __riscv_vsrl_vx_u8mf2(v4410, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4415 = __riscv_vzext_vf2_u16m1(v4414, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4416 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4415, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4417 = v21 + 203;
      const int8_t* v4418 = (const int8_t*) v4417;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4419 = *(const int8_t *)(v4418);
      const uint8_t* v4420 = v21 + 219;
      const int8_t* v4421 = (const int8_t*) v4420;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4422 = *(const int8_t *)(v4421);
      vint32m2_t v4423 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4424 = __riscv_vwmul_vx_i16m1(v4413, v4419, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4425 = __riscv_vwadd_wv_i32m2(v4423, v4424, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4426 = __riscv_vwmul_vx_i16m1(v4416, v4422, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4427 = __riscv_vwadd_wv_i32m2(v4425, v4426, 8);
      v4124 = v4427;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4428 = v19 + 1784;
      const uint8_t* v4429 = (const uint8_t*) v4428;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4430 = __riscv_vle8_v_u8mf2(v4429, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4431 = __riscv_vand_vx_u8mf2(v4430, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4432 = __riscv_vzext_vf2_u16m1(v4431, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4433 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4432, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4434 = __riscv_vsrl_vx_u8mf2(v4430, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4435 = __riscv_vzext_vf2_u16m1(v4434, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4436 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4435, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4437 = v21 + 203;
      const int8_t* v4438 = (const int8_t*) v4437;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4439 = *(const int8_t *)(v4438);
      const uint8_t* v4440 = v21 + 219;
      const int8_t* v4441 = (const int8_t*) v4440;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4442 = *(const int8_t *)(v4441);
      vint32m2_t v4443 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4444 = __riscv_vwmul_vx_i16m1(v4433, v4439, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4445 = __riscv_vwadd_wv_i32m2(v4443, v4444, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4446 = __riscv_vwmul_vx_i16m1(v4436, v4442, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4447 = __riscv_vwadd_wv_i32m2(v4445, v4446, 8);
      v4126 = v4447;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4448 = v19 + 1792;
      const uint8_t* v4449 = (const uint8_t*) v4448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4450 = __riscv_vle8_v_u8mf2(v4449, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4451 = __riscv_vand_vx_u8mf2(v4450, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4452 = __riscv_vzext_vf2_u16m1(v4451, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4453 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4452, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4454 = __riscv_vsrl_vx_u8mf2(v4450, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4455 = __riscv_vzext_vf2_u16m1(v4454, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4456 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4455, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4457 = v21 + 204;
      const int8_t* v4458 = (const int8_t*) v4457;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4459 = *(const int8_t *)(v4458);
      const uint8_t* v4460 = v21 + 220;
      const int8_t* v4461 = (const int8_t*) v4460;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4462 = *(const int8_t *)(v4461);
      vint32m2_t v4463 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4464 = __riscv_vwmul_vx_i16m1(v4453, v4459, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4465 = __riscv_vwadd_wv_i32m2(v4463, v4464, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4466 = __riscv_vwmul_vx_i16m1(v4456, v4462, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4467 = __riscv_vwadd_wv_i32m2(v4465, v4466, 8);
      v4124 = v4467;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4468 = v19 + 1800;
      const uint8_t* v4469 = (const uint8_t*) v4468;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4470 = __riscv_vle8_v_u8mf2(v4469, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4471 = __riscv_vand_vx_u8mf2(v4470, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4472 = __riscv_vzext_vf2_u16m1(v4471, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4473 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4472, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4474 = __riscv_vsrl_vx_u8mf2(v4470, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4475 = __riscv_vzext_vf2_u16m1(v4474, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4476 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4475, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4477 = v21 + 204;
      const int8_t* v4478 = (const int8_t*) v4477;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4479 = *(const int8_t *)(v4478);
      const uint8_t* v4480 = v21 + 220;
      const int8_t* v4481 = (const int8_t*) v4480;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4482 = *(const int8_t *)(v4481);
      vint32m2_t v4483 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4484 = __riscv_vwmul_vx_i16m1(v4473, v4479, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4485 = __riscv_vwadd_wv_i32m2(v4483, v4484, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4486 = __riscv_vwmul_vx_i16m1(v4476, v4482, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4487 = __riscv_vwadd_wv_i32m2(v4485, v4486, 8);
      v4126 = v4487;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4488 = v19 + 1808;
      const uint8_t* v4489 = (const uint8_t*) v4488;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4490 = __riscv_vle8_v_u8mf2(v4489, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4491 = __riscv_vand_vx_u8mf2(v4490, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4492 = __riscv_vzext_vf2_u16m1(v4491, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4493 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4492, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4494 = __riscv_vsrl_vx_u8mf2(v4490, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4495 = __riscv_vzext_vf2_u16m1(v4494, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4496 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4495, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4497 = v21 + 205;
      const int8_t* v4498 = (const int8_t*) v4497;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4499 = *(const int8_t *)(v4498);
      const uint8_t* v4500 = v21 + 221;
      const int8_t* v4501 = (const int8_t*) v4500;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4502 = *(const int8_t *)(v4501);
      vint32m2_t v4503 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4504 = __riscv_vwmul_vx_i16m1(v4493, v4499, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4505 = __riscv_vwadd_wv_i32m2(v4503, v4504, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4506 = __riscv_vwmul_vx_i16m1(v4496, v4502, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4507 = __riscv_vwadd_wv_i32m2(v4505, v4506, 8);
      v4124 = v4507;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4508 = v19 + 1816;
      const uint8_t* v4509 = (const uint8_t*) v4508;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4510 = __riscv_vle8_v_u8mf2(v4509, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4511 = __riscv_vand_vx_u8mf2(v4510, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4512 = __riscv_vzext_vf2_u16m1(v4511, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4513 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4512, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4514 = __riscv_vsrl_vx_u8mf2(v4510, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4515 = __riscv_vzext_vf2_u16m1(v4514, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4516 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4515, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4517 = v21 + 205;
      const int8_t* v4518 = (const int8_t*) v4517;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4519 = *(const int8_t *)(v4518);
      const uint8_t* v4520 = v21 + 221;
      const int8_t* v4521 = (const int8_t*) v4520;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4522 = *(const int8_t *)(v4521);
      vint32m2_t v4523 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4524 = __riscv_vwmul_vx_i16m1(v4513, v4519, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4525 = __riscv_vwadd_wv_i32m2(v4523, v4524, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4526 = __riscv_vwmul_vx_i16m1(v4516, v4522, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4527 = __riscv_vwadd_wv_i32m2(v4525, v4526, 8);
      v4126 = v4527;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4528 = v19 + 1824;
      const uint8_t* v4529 = (const uint8_t*) v4528;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4530 = __riscv_vle8_v_u8mf2(v4529, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4531 = __riscv_vand_vx_u8mf2(v4530, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4532 = __riscv_vzext_vf2_u16m1(v4531, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4533 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4532, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4534 = __riscv_vsrl_vx_u8mf2(v4530, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4535 = __riscv_vzext_vf2_u16m1(v4534, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4536 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4535, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4537 = v21 + 206;
      const int8_t* v4538 = (const int8_t*) v4537;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4539 = *(const int8_t *)(v4538);
      const uint8_t* v4540 = v21 + 222;
      const int8_t* v4541 = (const int8_t*) v4540;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4542 = *(const int8_t *)(v4541);
      vint32m2_t v4543 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4544 = __riscv_vwmul_vx_i16m1(v4533, v4539, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4545 = __riscv_vwadd_wv_i32m2(v4543, v4544, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4546 = __riscv_vwmul_vx_i16m1(v4536, v4542, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4547 = __riscv_vwadd_wv_i32m2(v4545, v4546, 8);
      v4124 = v4547;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4548 = v19 + 1832;
      const uint8_t* v4549 = (const uint8_t*) v4548;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4550 = __riscv_vle8_v_u8mf2(v4549, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4551 = __riscv_vand_vx_u8mf2(v4550, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4552 = __riscv_vzext_vf2_u16m1(v4551, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4553 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4552, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4554 = __riscv_vsrl_vx_u8mf2(v4550, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4555 = __riscv_vzext_vf2_u16m1(v4554, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4556 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4555, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4557 = v21 + 206;
      const int8_t* v4558 = (const int8_t*) v4557;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4559 = *(const int8_t *)(v4558);
      const uint8_t* v4560 = v21 + 222;
      const int8_t* v4561 = (const int8_t*) v4560;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4562 = *(const int8_t *)(v4561);
      vint32m2_t v4563 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4564 = __riscv_vwmul_vx_i16m1(v4553, v4559, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4565 = __riscv_vwadd_wv_i32m2(v4563, v4564, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4566 = __riscv_vwmul_vx_i16m1(v4556, v4562, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4567 = __riscv_vwadd_wv_i32m2(v4565, v4566, 8);
      v4126 = v4567;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4568 = v19 + 1840;
      const uint8_t* v4569 = (const uint8_t*) v4568;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4570 = __riscv_vle8_v_u8mf2(v4569, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4571 = __riscv_vand_vx_u8mf2(v4570, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4572 = __riscv_vzext_vf2_u16m1(v4571, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4573 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4572, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4574 = __riscv_vsrl_vx_u8mf2(v4570, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4575 = __riscv_vzext_vf2_u16m1(v4574, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4576 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4575, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4577 = v21 + 207;
      const int8_t* v4578 = (const int8_t*) v4577;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4579 = *(const int8_t *)(v4578);
      const uint8_t* v4580 = v21 + 223;
      const int8_t* v4581 = (const int8_t*) v4580;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4582 = *(const int8_t *)(v4581);
      vint32m2_t v4583 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4584 = __riscv_vwmul_vx_i16m1(v4573, v4579, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4585 = __riscv_vwadd_wv_i32m2(v4583, v4584, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4586 = __riscv_vwmul_vx_i16m1(v4576, v4582, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4587 = __riscv_vwadd_wv_i32m2(v4585, v4586, 8);
      v4124 = v4587;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4588 = v19 + 1848;
      const uint8_t* v4589 = (const uint8_t*) v4588;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4590 = __riscv_vle8_v_u8mf2(v4589, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4591 = __riscv_vand_vx_u8mf2(v4590, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4592 = __riscv_vzext_vf2_u16m1(v4591, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4593 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4592, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4594 = __riscv_vsrl_vx_u8mf2(v4590, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4595 = __riscv_vzext_vf2_u16m1(v4594, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4596 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4595, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4597 = v21 + 207;
      const int8_t* v4598 = (const int8_t*) v4597;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4599 = *(const int8_t *)(v4598);
      const uint8_t* v4600 = v21 + 223;
      const int8_t* v4601 = (const int8_t*) v4600;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4602 = *(const int8_t *)(v4601);
      vint32m2_t v4603 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4604 = __riscv_vwmul_vx_i16m1(v4593, v4599, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4605 = __riscv_vwadd_wv_i32m2(v4603, v4604, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4606 = __riscv_vwmul_vx_i16m1(v4596, v4602, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4607 = __riscv_vwadd_wv_i32m2(v4605, v4606, 8);
      v4126 = v4607;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4608 = v19 + 1856;
      const uint8_t* v4609 = (const uint8_t*) v4608;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4610 = __riscv_vle8_v_u8mf2(v4609, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4611 = __riscv_vand_vx_u8mf2(v4610, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4612 = __riscv_vzext_vf2_u16m1(v4611, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4613 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4612, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4614 = __riscv_vsrl_vx_u8mf2(v4610, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4615 = __riscv_vzext_vf2_u16m1(v4614, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4616 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4615, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4617 = v21 + 208;
      const int8_t* v4618 = (const int8_t*) v4617;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4619 = *(const int8_t *)(v4618);
      const uint8_t* v4620 = v21 + 224;
      const int8_t* v4621 = (const int8_t*) v4620;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4622 = *(const int8_t *)(v4621);
      vint32m2_t v4623 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4624 = __riscv_vwmul_vx_i16m1(v4613, v4619, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4625 = __riscv_vwadd_wv_i32m2(v4623, v4624, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4626 = __riscv_vwmul_vx_i16m1(v4616, v4622, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4627 = __riscv_vwadd_wv_i32m2(v4625, v4626, 8);
      v4124 = v4627;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4628 = v19 + 1864;
      const uint8_t* v4629 = (const uint8_t*) v4628;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4630 = __riscv_vle8_v_u8mf2(v4629, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4631 = __riscv_vand_vx_u8mf2(v4630, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4632 = __riscv_vzext_vf2_u16m1(v4631, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4633 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4632, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4634 = __riscv_vsrl_vx_u8mf2(v4630, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4635 = __riscv_vzext_vf2_u16m1(v4634, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4636 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4635, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4637 = v21 + 208;
      const int8_t* v4638 = (const int8_t*) v4637;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4639 = *(const int8_t *)(v4638);
      const uint8_t* v4640 = v21 + 224;
      const int8_t* v4641 = (const int8_t*) v4640;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4642 = *(const int8_t *)(v4641);
      vint32m2_t v4643 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4644 = __riscv_vwmul_vx_i16m1(v4633, v4639, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4645 = __riscv_vwadd_wv_i32m2(v4643, v4644, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4646 = __riscv_vwmul_vx_i16m1(v4636, v4642, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4647 = __riscv_vwadd_wv_i32m2(v4645, v4646, 8);
      v4126 = v4647;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4648 = v19 + 1872;
      const uint8_t* v4649 = (const uint8_t*) v4648;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4650 = __riscv_vle8_v_u8mf2(v4649, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4651 = __riscv_vand_vx_u8mf2(v4650, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4652 = __riscv_vzext_vf2_u16m1(v4651, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4653 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4652, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4654 = __riscv_vsrl_vx_u8mf2(v4650, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4655 = __riscv_vzext_vf2_u16m1(v4654, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4656 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4655, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4657 = v21 + 209;
      const int8_t* v4658 = (const int8_t*) v4657;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4659 = *(const int8_t *)(v4658);
      const uint8_t* v4660 = v21 + 225;
      const int8_t* v4661 = (const int8_t*) v4660;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4662 = *(const int8_t *)(v4661);
      vint32m2_t v4663 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4664 = __riscv_vwmul_vx_i16m1(v4653, v4659, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4665 = __riscv_vwadd_wv_i32m2(v4663, v4664, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4666 = __riscv_vwmul_vx_i16m1(v4656, v4662, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4667 = __riscv_vwadd_wv_i32m2(v4665, v4666, 8);
      v4124 = v4667;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4668 = v19 + 1880;
      const uint8_t* v4669 = (const uint8_t*) v4668;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4670 = __riscv_vle8_v_u8mf2(v4669, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4671 = __riscv_vand_vx_u8mf2(v4670, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4672 = __riscv_vzext_vf2_u16m1(v4671, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4673 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4672, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4674 = __riscv_vsrl_vx_u8mf2(v4670, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4675 = __riscv_vzext_vf2_u16m1(v4674, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4676 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4675, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4677 = v21 + 209;
      const int8_t* v4678 = (const int8_t*) v4677;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4679 = *(const int8_t *)(v4678);
      const uint8_t* v4680 = v21 + 225;
      const int8_t* v4681 = (const int8_t*) v4680;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4682 = *(const int8_t *)(v4681);
      vint32m2_t v4683 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4684 = __riscv_vwmul_vx_i16m1(v4673, v4679, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4685 = __riscv_vwadd_wv_i32m2(v4683, v4684, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4686 = __riscv_vwmul_vx_i16m1(v4676, v4682, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4687 = __riscv_vwadd_wv_i32m2(v4685, v4686, 8);
      v4126 = v4687;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4688 = v19 + 1888;
      const uint8_t* v4689 = (const uint8_t*) v4688;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4690 = __riscv_vle8_v_u8mf2(v4689, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4691 = __riscv_vand_vx_u8mf2(v4690, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4692 = __riscv_vzext_vf2_u16m1(v4691, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4693 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4692, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4694 = __riscv_vsrl_vx_u8mf2(v4690, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4695 = __riscv_vzext_vf2_u16m1(v4694, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4696 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4695, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4697 = v21 + 210;
      const int8_t* v4698 = (const int8_t*) v4697;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4699 = *(const int8_t *)(v4698);
      const uint8_t* v4700 = v21 + 226;
      const int8_t* v4701 = (const int8_t*) v4700;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4702 = *(const int8_t *)(v4701);
      vint32m2_t v4703 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4704 = __riscv_vwmul_vx_i16m1(v4693, v4699, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4705 = __riscv_vwadd_wv_i32m2(v4703, v4704, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4706 = __riscv_vwmul_vx_i16m1(v4696, v4702, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4707 = __riscv_vwadd_wv_i32m2(v4705, v4706, 8);
      v4124 = v4707;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4708 = v19 + 1896;
      const uint8_t* v4709 = (const uint8_t*) v4708;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4710 = __riscv_vle8_v_u8mf2(v4709, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4711 = __riscv_vand_vx_u8mf2(v4710, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4712 = __riscv_vzext_vf2_u16m1(v4711, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4713 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4712, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4714 = __riscv_vsrl_vx_u8mf2(v4710, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4715 = __riscv_vzext_vf2_u16m1(v4714, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4716 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4715, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4717 = v21 + 210;
      const int8_t* v4718 = (const int8_t*) v4717;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4719 = *(const int8_t *)(v4718);
      const uint8_t* v4720 = v21 + 226;
      const int8_t* v4721 = (const int8_t*) v4720;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4722 = *(const int8_t *)(v4721);
      vint32m2_t v4723 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4724 = __riscv_vwmul_vx_i16m1(v4713, v4719, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4725 = __riscv_vwadd_wv_i32m2(v4723, v4724, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4726 = __riscv_vwmul_vx_i16m1(v4716, v4722, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4727 = __riscv_vwadd_wv_i32m2(v4725, v4726, 8);
      v4126 = v4727;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4728 = v19 + 1904;
      const uint8_t* v4729 = (const uint8_t*) v4728;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4730 = __riscv_vle8_v_u8mf2(v4729, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4731 = __riscv_vand_vx_u8mf2(v4730, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4732 = __riscv_vzext_vf2_u16m1(v4731, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4733 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4732, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4734 = __riscv_vsrl_vx_u8mf2(v4730, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4735 = __riscv_vzext_vf2_u16m1(v4734, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4736 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4735, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4737 = v21 + 211;
      const int8_t* v4738 = (const int8_t*) v4737;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4739 = *(const int8_t *)(v4738);
      const uint8_t* v4740 = v21 + 227;
      const int8_t* v4741 = (const int8_t*) v4740;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4742 = *(const int8_t *)(v4741);
      vint32m2_t v4743 = v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4744 = __riscv_vwmul_vx_i16m1(v4733, v4739, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4745 = __riscv_vwadd_wv_i32m2(v4743, v4744, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4746 = __riscv_vwmul_vx_i16m1(v4736, v4742, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4747 = __riscv_vwadd_wv_i32m2(v4745, v4746, 8);
      v4124 = v4747;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4748 = v19 + 1912;
      const uint8_t* v4749 = (const uint8_t*) v4748;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4750 = __riscv_vle8_v_u8mf2(v4749, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4751 = __riscv_vand_vx_u8mf2(v4750, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4752 = __riscv_vzext_vf2_u16m1(v4751, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4753 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4752, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4754 = __riscv_vsrl_vx_u8mf2(v4750, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4755 = __riscv_vzext_vf2_u16m1(v4754, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4756 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4755, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4757 = v21 + 211;
      const int8_t* v4758 = (const int8_t*) v4757;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4759 = *(const int8_t *)(v4758);
      const uint8_t* v4760 = v21 + 227;
      const int8_t* v4761 = (const int8_t*) v4760;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4762 = *(const int8_t *)(v4761);
      vint32m2_t v4763 = v4126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4764 = __riscv_vwmul_vx_i16m1(v4753, v4759, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4765 = __riscv_vwadd_wv_i32m2(v4763, v4764, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4766 = __riscv_vwmul_vx_i16m1(v4756, v4762, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4767 = __riscv_vwadd_wv_i32m2(v4765, v4766, 8);
      v4126 = v4767;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_scale_fold
      vint32m2_t v4768 = v4124;
      vint32m2_t v4769 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v4770 = __riscv_vmacc_vv_i32m2(v4769, v4109, v4768, 8);
      v24 = v4770;
      vint32m2_t v4771 = v4126;
      vint32m2_t v4772 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v4773 = __riscv_vmacc_vv_i32m2(v4772, v4123, v4771, 8);
      v26 = v4773;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_signed_scale
      const uint8_t* v4774 = v19 + 112;
      const uint8_t* v4775 = (const uint8_t*) v4774;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4776 = __riscv_vle8_v_u8mf2(v4775, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4777 = __riscv_vsrl_vx_u8mf2(v4776, 4, 8);
      const uint8_t* v4778 = v19 + 48;
      const uint8_t* v4779 = (const uint8_t*) v4778;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4780 = __riscv_vle8_v_u8mf2(v4779, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4781 = __riscv_vsrl_vx_u8mf2(v4780, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4782 = __riscv_vand_vx_u8mf2(v4781, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v4783 = __riscv_vsll_vx_u8mf2(v4782, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v4784 = __riscv_vor_vv_u8mf2(v4783, v4777, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4785 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4784);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v4786 = __riscv_vsub_vx_i8mf2(v4785, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v4787 = __riscv_vsext_vf4_i32m2(v4786, 8);
      const uint8_t* v4788 = v19 + 120;
      const uint8_t* v4789 = (const uint8_t*) v4788;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4790 = __riscv_vle8_v_u8mf2(v4789, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4791 = __riscv_vsrl_vx_u8mf2(v4790, 4, 8);
      const uint8_t* v4792 = v19 + 56;
      const uint8_t* v4793 = (const uint8_t*) v4792;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4794 = __riscv_vle8_v_u8mf2(v4793, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4795 = __riscv_vsrl_vx_u8mf2(v4794, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4796 = __riscv_vand_vx_u8mf2(v4795, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v4797 = __riscv_vsll_vx_u8mf2(v4796, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v4798 = __riscv_vor_vv_u8mf2(v4797, v4791, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4799 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4798);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
      vint8mf2_t v4800 = __riscv_vsub_vx_i8mf2(v4799, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_widen
      vint32m2_t v4801 = __riscv_vsext_vf4_i32m2(v4800, 8);
      vint32m2_t v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v4803 = __riscv_vmv_v_x_i32m2(0, 8);
      v4802 = v4803;
      vint32m2_t v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v4805 = __riscv_vmv_v_x_i32m2(0, 8);
      v4804 = v4805;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4806 = v19 + 1920;
      const uint8_t* v4807 = (const uint8_t*) v4806;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4808 = __riscv_vle8_v_u8mf2(v4807, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4809 = __riscv_vand_vx_u8mf2(v4808, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4810 = __riscv_vzext_vf2_u16m1(v4809, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4811 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4810, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4812 = __riscv_vsrl_vx_u8mf2(v4808, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4813 = __riscv_vzext_vf2_u16m1(v4812, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4814 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4813, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4815 = v21 + 228;
      const int8_t* v4816 = (const int8_t*) v4815;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4817 = *(const int8_t *)(v4816);
      const uint8_t* v4818 = v21 + 244;
      const int8_t* v4819 = (const int8_t*) v4818;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4820 = *(const int8_t *)(v4819);
      vint32m2_t v4821 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4822 = __riscv_vwmul_vx_i16m1(v4811, v4817, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4823 = __riscv_vwadd_wv_i32m2(v4821, v4822, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4824 = __riscv_vwmul_vx_i16m1(v4814, v4820, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4825 = __riscv_vwadd_wv_i32m2(v4823, v4824, 8);
      v4802 = v4825;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4826 = v19 + 1928;
      const uint8_t* v4827 = (const uint8_t*) v4826;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4828 = __riscv_vle8_v_u8mf2(v4827, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4829 = __riscv_vand_vx_u8mf2(v4828, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4830 = __riscv_vzext_vf2_u16m1(v4829, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4831 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4830, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4832 = __riscv_vsrl_vx_u8mf2(v4828, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4833 = __riscv_vzext_vf2_u16m1(v4832, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4834 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4833, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4835 = v21 + 228;
      const int8_t* v4836 = (const int8_t*) v4835;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4837 = *(const int8_t *)(v4836);
      const uint8_t* v4838 = v21 + 244;
      const int8_t* v4839 = (const int8_t*) v4838;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4840 = *(const int8_t *)(v4839);
      vint32m2_t v4841 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4842 = __riscv_vwmul_vx_i16m1(v4831, v4837, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4843 = __riscv_vwadd_wv_i32m2(v4841, v4842, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4844 = __riscv_vwmul_vx_i16m1(v4834, v4840, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4845 = __riscv_vwadd_wv_i32m2(v4843, v4844, 8);
      v4804 = v4845;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4846 = v19 + 1936;
      const uint8_t* v4847 = (const uint8_t*) v4846;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4848 = __riscv_vle8_v_u8mf2(v4847, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4849 = __riscv_vand_vx_u8mf2(v4848, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4850 = __riscv_vzext_vf2_u16m1(v4849, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4851 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4850, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4852 = __riscv_vsrl_vx_u8mf2(v4848, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4853 = __riscv_vzext_vf2_u16m1(v4852, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4854 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4853, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4855 = v21 + 229;
      const int8_t* v4856 = (const int8_t*) v4855;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4857 = *(const int8_t *)(v4856);
      const uint8_t* v4858 = v21 + 245;
      const int8_t* v4859 = (const int8_t*) v4858;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4860 = *(const int8_t *)(v4859);
      vint32m2_t v4861 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4862 = __riscv_vwmul_vx_i16m1(v4851, v4857, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4863 = __riscv_vwadd_wv_i32m2(v4861, v4862, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4864 = __riscv_vwmul_vx_i16m1(v4854, v4860, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4865 = __riscv_vwadd_wv_i32m2(v4863, v4864, 8);
      v4802 = v4865;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4866 = v19 + 1944;
      const uint8_t* v4867 = (const uint8_t*) v4866;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4868 = __riscv_vle8_v_u8mf2(v4867, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4869 = __riscv_vand_vx_u8mf2(v4868, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4870 = __riscv_vzext_vf2_u16m1(v4869, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4871 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4870, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4872 = __riscv_vsrl_vx_u8mf2(v4868, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4873 = __riscv_vzext_vf2_u16m1(v4872, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4874 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4873, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4875 = v21 + 229;
      const int8_t* v4876 = (const int8_t*) v4875;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4877 = *(const int8_t *)(v4876);
      const uint8_t* v4878 = v21 + 245;
      const int8_t* v4879 = (const int8_t*) v4878;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4880 = *(const int8_t *)(v4879);
      vint32m2_t v4881 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4882 = __riscv_vwmul_vx_i16m1(v4871, v4877, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4883 = __riscv_vwadd_wv_i32m2(v4881, v4882, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4884 = __riscv_vwmul_vx_i16m1(v4874, v4880, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4885 = __riscv_vwadd_wv_i32m2(v4883, v4884, 8);
      v4804 = v4885;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4886 = v19 + 1952;
      const uint8_t* v4887 = (const uint8_t*) v4886;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4888 = __riscv_vle8_v_u8mf2(v4887, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4889 = __riscv_vand_vx_u8mf2(v4888, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4890 = __riscv_vzext_vf2_u16m1(v4889, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4891 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4890, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4892 = __riscv_vsrl_vx_u8mf2(v4888, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4893 = __riscv_vzext_vf2_u16m1(v4892, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4894 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4893, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4895 = v21 + 230;
      const int8_t* v4896 = (const int8_t*) v4895;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4897 = *(const int8_t *)(v4896);
      const uint8_t* v4898 = v21 + 246;
      const int8_t* v4899 = (const int8_t*) v4898;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4900 = *(const int8_t *)(v4899);
      vint32m2_t v4901 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4902 = __riscv_vwmul_vx_i16m1(v4891, v4897, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4903 = __riscv_vwadd_wv_i32m2(v4901, v4902, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4904 = __riscv_vwmul_vx_i16m1(v4894, v4900, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4905 = __riscv_vwadd_wv_i32m2(v4903, v4904, 8);
      v4802 = v4905;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4906 = v19 + 1960;
      const uint8_t* v4907 = (const uint8_t*) v4906;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4908 = __riscv_vle8_v_u8mf2(v4907, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4909 = __riscv_vand_vx_u8mf2(v4908, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4910 = __riscv_vzext_vf2_u16m1(v4909, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4911 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4910, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4912 = __riscv_vsrl_vx_u8mf2(v4908, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4913 = __riscv_vzext_vf2_u16m1(v4912, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4914 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4913, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4915 = v21 + 230;
      const int8_t* v4916 = (const int8_t*) v4915;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4917 = *(const int8_t *)(v4916);
      const uint8_t* v4918 = v21 + 246;
      const int8_t* v4919 = (const int8_t*) v4918;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4920 = *(const int8_t *)(v4919);
      vint32m2_t v4921 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4922 = __riscv_vwmul_vx_i16m1(v4911, v4917, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4923 = __riscv_vwadd_wv_i32m2(v4921, v4922, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4924 = __riscv_vwmul_vx_i16m1(v4914, v4920, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4925 = __riscv_vwadd_wv_i32m2(v4923, v4924, 8);
      v4804 = v4925;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4926 = v19 + 1968;
      const uint8_t* v4927 = (const uint8_t*) v4926;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4928 = __riscv_vle8_v_u8mf2(v4927, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4929 = __riscv_vand_vx_u8mf2(v4928, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4930 = __riscv_vzext_vf2_u16m1(v4929, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4931 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4930, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4932 = __riscv_vsrl_vx_u8mf2(v4928, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4933 = __riscv_vzext_vf2_u16m1(v4932, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4934 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4933, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4935 = v21 + 231;
      const int8_t* v4936 = (const int8_t*) v4935;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4937 = *(const int8_t *)(v4936);
      const uint8_t* v4938 = v21 + 247;
      const int8_t* v4939 = (const int8_t*) v4938;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4940 = *(const int8_t *)(v4939);
      vint32m2_t v4941 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4942 = __riscv_vwmul_vx_i16m1(v4931, v4937, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4943 = __riscv_vwadd_wv_i32m2(v4941, v4942, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4944 = __riscv_vwmul_vx_i16m1(v4934, v4940, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4945 = __riscv_vwadd_wv_i32m2(v4943, v4944, 8);
      v4802 = v4945;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4946 = v19 + 1976;
      const uint8_t* v4947 = (const uint8_t*) v4946;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4948 = __riscv_vle8_v_u8mf2(v4947, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4949 = __riscv_vand_vx_u8mf2(v4948, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4950 = __riscv_vzext_vf2_u16m1(v4949, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4951 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4950, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4952 = __riscv_vsrl_vx_u8mf2(v4948, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4953 = __riscv_vzext_vf2_u16m1(v4952, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4954 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4953, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4955 = v21 + 231;
      const int8_t* v4956 = (const int8_t*) v4955;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4957 = *(const int8_t *)(v4956);
      const uint8_t* v4958 = v21 + 247;
      const int8_t* v4959 = (const int8_t*) v4958;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4960 = *(const int8_t *)(v4959);
      vint32m2_t v4961 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4962 = __riscv_vwmul_vx_i16m1(v4951, v4957, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4963 = __riscv_vwadd_wv_i32m2(v4961, v4962, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4964 = __riscv_vwmul_vx_i16m1(v4954, v4960, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4965 = __riscv_vwadd_wv_i32m2(v4963, v4964, 8);
      v4804 = v4965;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4966 = v19 + 1984;
      const uint8_t* v4967 = (const uint8_t*) v4966;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4968 = __riscv_vle8_v_u8mf2(v4967, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4969 = __riscv_vand_vx_u8mf2(v4968, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4970 = __riscv_vzext_vf2_u16m1(v4969, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4971 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4970, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4972 = __riscv_vsrl_vx_u8mf2(v4968, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4973 = __riscv_vzext_vf2_u16m1(v4972, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4974 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4973, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4975 = v21 + 232;
      const int8_t* v4976 = (const int8_t*) v4975;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4977 = *(const int8_t *)(v4976);
      const uint8_t* v4978 = v21 + 248;
      const int8_t* v4979 = (const int8_t*) v4978;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4980 = *(const int8_t *)(v4979);
      vint32m2_t v4981 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4982 = __riscv_vwmul_vx_i16m1(v4971, v4977, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4983 = __riscv_vwadd_wv_i32m2(v4981, v4982, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v4984 = __riscv_vwmul_vx_i16m1(v4974, v4980, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v4985 = __riscv_vwadd_wv_i32m2(v4983, v4984, 8);
      v4802 = v4985;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4986 = v19 + 1992;
      const uint8_t* v4987 = (const uint8_t*) v4986;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4988 = __riscv_vle8_v_u8mf2(v4987, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4989 = __riscv_vand_vx_u8mf2(v4988, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4990 = __riscv_vzext_vf2_u16m1(v4989, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4991 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4990, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4992 = __riscv_vsrl_vx_u8mf2(v4988, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v4993 = __riscv_vzext_vf2_u16m1(v4992, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v4994 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v4993, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4995 = v21 + 232;
      const int8_t* v4996 = (const int8_t*) v4995;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4997 = *(const int8_t *)(v4996);
      const uint8_t* v4998 = v21 + 248;
      const int8_t* v4999 = (const int8_t*) v4998;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5000 = *(const int8_t *)(v4999);
      vint32m2_t v5001 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5002 = __riscv_vwmul_vx_i16m1(v4991, v4997, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5003 = __riscv_vwadd_wv_i32m2(v5001, v5002, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5004 = __riscv_vwmul_vx_i16m1(v4994, v5000, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5005 = __riscv_vwadd_wv_i32m2(v5003, v5004, 8);
      v4804 = v5005;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5006 = v19 + 2000;
      const uint8_t* v5007 = (const uint8_t*) v5006;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5008 = __riscv_vle8_v_u8mf2(v5007, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5009 = __riscv_vand_vx_u8mf2(v5008, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5010 = __riscv_vzext_vf2_u16m1(v5009, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5011 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5010, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5012 = __riscv_vsrl_vx_u8mf2(v5008, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5013 = __riscv_vzext_vf2_u16m1(v5012, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5014 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5013, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5015 = v21 + 233;
      const int8_t* v5016 = (const int8_t*) v5015;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5017 = *(const int8_t *)(v5016);
      const uint8_t* v5018 = v21 + 249;
      const int8_t* v5019 = (const int8_t*) v5018;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5020 = *(const int8_t *)(v5019);
      vint32m2_t v5021 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5022 = __riscv_vwmul_vx_i16m1(v5011, v5017, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5023 = __riscv_vwadd_wv_i32m2(v5021, v5022, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5024 = __riscv_vwmul_vx_i16m1(v5014, v5020, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5025 = __riscv_vwadd_wv_i32m2(v5023, v5024, 8);
      v4802 = v5025;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5026 = v19 + 2008;
      const uint8_t* v5027 = (const uint8_t*) v5026;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5028 = __riscv_vle8_v_u8mf2(v5027, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5029 = __riscv_vand_vx_u8mf2(v5028, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5030 = __riscv_vzext_vf2_u16m1(v5029, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5031 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5030, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5032 = __riscv_vsrl_vx_u8mf2(v5028, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5033 = __riscv_vzext_vf2_u16m1(v5032, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5034 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5033, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5035 = v21 + 233;
      const int8_t* v5036 = (const int8_t*) v5035;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5037 = *(const int8_t *)(v5036);
      const uint8_t* v5038 = v21 + 249;
      const int8_t* v5039 = (const int8_t*) v5038;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5040 = *(const int8_t *)(v5039);
      vint32m2_t v5041 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5042 = __riscv_vwmul_vx_i16m1(v5031, v5037, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5043 = __riscv_vwadd_wv_i32m2(v5041, v5042, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5044 = __riscv_vwmul_vx_i16m1(v5034, v5040, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5045 = __riscv_vwadd_wv_i32m2(v5043, v5044, 8);
      v4804 = v5045;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5046 = v19 + 2016;
      const uint8_t* v5047 = (const uint8_t*) v5046;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5048 = __riscv_vle8_v_u8mf2(v5047, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5049 = __riscv_vand_vx_u8mf2(v5048, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5050 = __riscv_vzext_vf2_u16m1(v5049, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5051 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5050, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5052 = __riscv_vsrl_vx_u8mf2(v5048, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5053 = __riscv_vzext_vf2_u16m1(v5052, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5054 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5053, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5055 = v21 + 234;
      const int8_t* v5056 = (const int8_t*) v5055;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5057 = *(const int8_t *)(v5056);
      const uint8_t* v5058 = v21 + 250;
      const int8_t* v5059 = (const int8_t*) v5058;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5060 = *(const int8_t *)(v5059);
      vint32m2_t v5061 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5062 = __riscv_vwmul_vx_i16m1(v5051, v5057, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5063 = __riscv_vwadd_wv_i32m2(v5061, v5062, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5064 = __riscv_vwmul_vx_i16m1(v5054, v5060, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5065 = __riscv_vwadd_wv_i32m2(v5063, v5064, 8);
      v4802 = v5065;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5066 = v19 + 2024;
      const uint8_t* v5067 = (const uint8_t*) v5066;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5068 = __riscv_vle8_v_u8mf2(v5067, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5069 = __riscv_vand_vx_u8mf2(v5068, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5070 = __riscv_vzext_vf2_u16m1(v5069, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5071 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5070, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5072 = __riscv_vsrl_vx_u8mf2(v5068, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5073 = __riscv_vzext_vf2_u16m1(v5072, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5074 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5073, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5075 = v21 + 234;
      const int8_t* v5076 = (const int8_t*) v5075;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5077 = *(const int8_t *)(v5076);
      const uint8_t* v5078 = v21 + 250;
      const int8_t* v5079 = (const int8_t*) v5078;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5080 = *(const int8_t *)(v5079);
      vint32m2_t v5081 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5082 = __riscv_vwmul_vx_i16m1(v5071, v5077, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5083 = __riscv_vwadd_wv_i32m2(v5081, v5082, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5084 = __riscv_vwmul_vx_i16m1(v5074, v5080, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5085 = __riscv_vwadd_wv_i32m2(v5083, v5084, 8);
      v4804 = v5085;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5086 = v19 + 2032;
      const uint8_t* v5087 = (const uint8_t*) v5086;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5088 = __riscv_vle8_v_u8mf2(v5087, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5089 = __riscv_vand_vx_u8mf2(v5088, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5090 = __riscv_vzext_vf2_u16m1(v5089, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5091 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5090, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5092 = __riscv_vsrl_vx_u8mf2(v5088, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5093 = __riscv_vzext_vf2_u16m1(v5092, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5094 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5093, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5095 = v21 + 235;
      const int8_t* v5096 = (const int8_t*) v5095;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5097 = *(const int8_t *)(v5096);
      const uint8_t* v5098 = v21 + 251;
      const int8_t* v5099 = (const int8_t*) v5098;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5100 = *(const int8_t *)(v5099);
      vint32m2_t v5101 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5102 = __riscv_vwmul_vx_i16m1(v5091, v5097, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5103 = __riscv_vwadd_wv_i32m2(v5101, v5102, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5104 = __riscv_vwmul_vx_i16m1(v5094, v5100, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5105 = __riscv_vwadd_wv_i32m2(v5103, v5104, 8);
      v4802 = v5105;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5106 = v19 + 2040;
      const uint8_t* v5107 = (const uint8_t*) v5106;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5108 = __riscv_vle8_v_u8mf2(v5107, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5109 = __riscv_vand_vx_u8mf2(v5108, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5110 = __riscv_vzext_vf2_u16m1(v5109, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5111 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5110, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5112 = __riscv_vsrl_vx_u8mf2(v5108, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5113 = __riscv_vzext_vf2_u16m1(v5112, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5114 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5113, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5115 = v21 + 235;
      const int8_t* v5116 = (const int8_t*) v5115;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5117 = *(const int8_t *)(v5116);
      const uint8_t* v5118 = v21 + 251;
      const int8_t* v5119 = (const int8_t*) v5118;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5120 = *(const int8_t *)(v5119);
      vint32m2_t v5121 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5122 = __riscv_vwmul_vx_i16m1(v5111, v5117, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5123 = __riscv_vwadd_wv_i32m2(v5121, v5122, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5124 = __riscv_vwmul_vx_i16m1(v5114, v5120, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5125 = __riscv_vwadd_wv_i32m2(v5123, v5124, 8);
      v4804 = v5125;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5126 = v19 + 2048;
      const uint8_t* v5127 = (const uint8_t*) v5126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5128 = __riscv_vle8_v_u8mf2(v5127, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5129 = __riscv_vand_vx_u8mf2(v5128, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5130 = __riscv_vzext_vf2_u16m1(v5129, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5131 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5130, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5132 = __riscv_vsrl_vx_u8mf2(v5128, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5133 = __riscv_vzext_vf2_u16m1(v5132, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5134 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5133, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5135 = v21 + 236;
      const int8_t* v5136 = (const int8_t*) v5135;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5137 = *(const int8_t *)(v5136);
      const uint8_t* v5138 = v21 + 252;
      const int8_t* v5139 = (const int8_t*) v5138;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5140 = *(const int8_t *)(v5139);
      vint32m2_t v5141 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5142 = __riscv_vwmul_vx_i16m1(v5131, v5137, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5143 = __riscv_vwadd_wv_i32m2(v5141, v5142, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5144 = __riscv_vwmul_vx_i16m1(v5134, v5140, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5145 = __riscv_vwadd_wv_i32m2(v5143, v5144, 8);
      v4802 = v5145;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5146 = v19 + 2056;
      const uint8_t* v5147 = (const uint8_t*) v5146;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5148 = __riscv_vle8_v_u8mf2(v5147, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5149 = __riscv_vand_vx_u8mf2(v5148, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5150 = __riscv_vzext_vf2_u16m1(v5149, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5151 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5150, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5152 = __riscv_vsrl_vx_u8mf2(v5148, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5153 = __riscv_vzext_vf2_u16m1(v5152, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5154 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5153, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5155 = v21 + 236;
      const int8_t* v5156 = (const int8_t*) v5155;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5157 = *(const int8_t *)(v5156);
      const uint8_t* v5158 = v21 + 252;
      const int8_t* v5159 = (const int8_t*) v5158;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5160 = *(const int8_t *)(v5159);
      vint32m2_t v5161 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5162 = __riscv_vwmul_vx_i16m1(v5151, v5157, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5163 = __riscv_vwadd_wv_i32m2(v5161, v5162, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5164 = __riscv_vwmul_vx_i16m1(v5154, v5160, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5165 = __riscv_vwadd_wv_i32m2(v5163, v5164, 8);
      v4804 = v5165;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5166 = v19 + 2064;
      const uint8_t* v5167 = (const uint8_t*) v5166;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5168 = __riscv_vle8_v_u8mf2(v5167, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5169 = __riscv_vand_vx_u8mf2(v5168, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5170 = __riscv_vzext_vf2_u16m1(v5169, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5171 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5170, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5172 = __riscv_vsrl_vx_u8mf2(v5168, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5173 = __riscv_vzext_vf2_u16m1(v5172, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5174 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5173, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5175 = v21 + 237;
      const int8_t* v5176 = (const int8_t*) v5175;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5177 = *(const int8_t *)(v5176);
      const uint8_t* v5178 = v21 + 253;
      const int8_t* v5179 = (const int8_t*) v5178;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5180 = *(const int8_t *)(v5179);
      vint32m2_t v5181 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5182 = __riscv_vwmul_vx_i16m1(v5171, v5177, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5183 = __riscv_vwadd_wv_i32m2(v5181, v5182, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5184 = __riscv_vwmul_vx_i16m1(v5174, v5180, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5185 = __riscv_vwadd_wv_i32m2(v5183, v5184, 8);
      v4802 = v5185;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5186 = v19 + 2072;
      const uint8_t* v5187 = (const uint8_t*) v5186;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5188 = __riscv_vle8_v_u8mf2(v5187, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5189 = __riscv_vand_vx_u8mf2(v5188, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5190 = __riscv_vzext_vf2_u16m1(v5189, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5191 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5190, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5192 = __riscv_vsrl_vx_u8mf2(v5188, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5193 = __riscv_vzext_vf2_u16m1(v5192, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5194 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5193, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5195 = v21 + 237;
      const int8_t* v5196 = (const int8_t*) v5195;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5197 = *(const int8_t *)(v5196);
      const uint8_t* v5198 = v21 + 253;
      const int8_t* v5199 = (const int8_t*) v5198;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5200 = *(const int8_t *)(v5199);
      vint32m2_t v5201 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5202 = __riscv_vwmul_vx_i16m1(v5191, v5197, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5203 = __riscv_vwadd_wv_i32m2(v5201, v5202, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5204 = __riscv_vwmul_vx_i16m1(v5194, v5200, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5205 = __riscv_vwadd_wv_i32m2(v5203, v5204, 8);
      v4804 = v5205;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5206 = v19 + 2080;
      const uint8_t* v5207 = (const uint8_t*) v5206;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5208 = __riscv_vle8_v_u8mf2(v5207, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5209 = __riscv_vand_vx_u8mf2(v5208, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5210 = __riscv_vzext_vf2_u16m1(v5209, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5211 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5210, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5212 = __riscv_vsrl_vx_u8mf2(v5208, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5213 = __riscv_vzext_vf2_u16m1(v5212, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5214 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5213, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5215 = v21 + 238;
      const int8_t* v5216 = (const int8_t*) v5215;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5217 = *(const int8_t *)(v5216);
      const uint8_t* v5218 = v21 + 254;
      const int8_t* v5219 = (const int8_t*) v5218;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5220 = *(const int8_t *)(v5219);
      vint32m2_t v5221 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5222 = __riscv_vwmul_vx_i16m1(v5211, v5217, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5223 = __riscv_vwadd_wv_i32m2(v5221, v5222, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5224 = __riscv_vwmul_vx_i16m1(v5214, v5220, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5225 = __riscv_vwadd_wv_i32m2(v5223, v5224, 8);
      v4802 = v5225;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5226 = v19 + 2088;
      const uint8_t* v5227 = (const uint8_t*) v5226;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5228 = __riscv_vle8_v_u8mf2(v5227, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5229 = __riscv_vand_vx_u8mf2(v5228, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5230 = __riscv_vzext_vf2_u16m1(v5229, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5231 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5230, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5232 = __riscv_vsrl_vx_u8mf2(v5228, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5233 = __riscv_vzext_vf2_u16m1(v5232, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5234 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5233, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5235 = v21 + 238;
      const int8_t* v5236 = (const int8_t*) v5235;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5237 = *(const int8_t *)(v5236);
      const uint8_t* v5238 = v21 + 254;
      const int8_t* v5239 = (const int8_t*) v5238;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5240 = *(const int8_t *)(v5239);
      vint32m2_t v5241 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5242 = __riscv_vwmul_vx_i16m1(v5231, v5237, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5243 = __riscv_vwadd_wv_i32m2(v5241, v5242, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5244 = __riscv_vwmul_vx_i16m1(v5234, v5240, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5245 = __riscv_vwadd_wv_i32m2(v5243, v5244, 8);
      v4804 = v5245;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5246 = v19 + 2096;
      const uint8_t* v5247 = (const uint8_t*) v5246;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5248 = __riscv_vle8_v_u8mf2(v5247, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5249 = __riscv_vand_vx_u8mf2(v5248, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5250 = __riscv_vzext_vf2_u16m1(v5249, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5251 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5250, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5252 = __riscv_vsrl_vx_u8mf2(v5248, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5253 = __riscv_vzext_vf2_u16m1(v5252, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5254 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5253, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5255 = v21 + 239;
      const int8_t* v5256 = (const int8_t*) v5255;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5257 = *(const int8_t *)(v5256);
      const uint8_t* v5258 = v21 + 255;
      const int8_t* v5259 = (const int8_t*) v5258;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5260 = *(const int8_t *)(v5259);
      vint32m2_t v5261 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5262 = __riscv_vwmul_vx_i16m1(v5251, v5257, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5263 = __riscv_vwadd_wv_i32m2(v5261, v5262, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5264 = __riscv_vwmul_vx_i16m1(v5254, v5260, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5265 = __riscv_vwadd_wv_i32m2(v5263, v5264, 8);
      v4802 = v5265;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5266 = v19 + 2104;
      const uint8_t* v5267 = (const uint8_t*) v5266;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5268 = __riscv_vle8_v_u8mf2(v5267, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5269 = __riscv_vand_vx_u8mf2(v5268, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5270 = __riscv_vzext_vf2_u16m1(v5269, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5271 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5270, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5272 = __riscv_vsrl_vx_u8mf2(v5268, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5273 = __riscv_vzext_vf2_u16m1(v5272, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5274 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5273, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5275 = v21 + 239;
      const int8_t* v5276 = (const int8_t*) v5275;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5277 = *(const int8_t *)(v5276);
      const uint8_t* v5278 = v21 + 255;
      const int8_t* v5279 = (const int8_t*) v5278;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5280 = *(const int8_t *)(v5279);
      vint32m2_t v5281 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5282 = __riscv_vwmul_vx_i16m1(v5271, v5277, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5283 = __riscv_vwadd_wv_i32m2(v5281, v5282, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5284 = __riscv_vwmul_vx_i16m1(v5274, v5280, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5285 = __riscv_vwadd_wv_i32m2(v5283, v5284, 8);
      v4804 = v5285;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5286 = v19 + 2112;
      const uint8_t* v5287 = (const uint8_t*) v5286;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5288 = __riscv_vle8_v_u8mf2(v5287, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5289 = __riscv_vand_vx_u8mf2(v5288, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5290 = __riscv_vzext_vf2_u16m1(v5289, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5291 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5290, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5292 = __riscv_vsrl_vx_u8mf2(v5288, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5293 = __riscv_vzext_vf2_u16m1(v5292, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5294 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5293, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5295 = v21 + 240;
      const int8_t* v5296 = (const int8_t*) v5295;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5297 = *(const int8_t *)(v5296);
      const uint8_t* v5298 = v21 + 256;
      const int8_t* v5299 = (const int8_t*) v5298;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5300 = *(const int8_t *)(v5299);
      vint32m2_t v5301 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5302 = __riscv_vwmul_vx_i16m1(v5291, v5297, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5303 = __riscv_vwadd_wv_i32m2(v5301, v5302, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5304 = __riscv_vwmul_vx_i16m1(v5294, v5300, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5305 = __riscv_vwadd_wv_i32m2(v5303, v5304, 8);
      v4802 = v5305;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5306 = v19 + 2120;
      const uint8_t* v5307 = (const uint8_t*) v5306;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5308 = __riscv_vle8_v_u8mf2(v5307, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5309 = __riscv_vand_vx_u8mf2(v5308, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5310 = __riscv_vzext_vf2_u16m1(v5309, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5311 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5310, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5312 = __riscv_vsrl_vx_u8mf2(v5308, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5313 = __riscv_vzext_vf2_u16m1(v5312, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5314 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5313, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5315 = v21 + 240;
      const int8_t* v5316 = (const int8_t*) v5315;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5317 = *(const int8_t *)(v5316);
      const uint8_t* v5318 = v21 + 256;
      const int8_t* v5319 = (const int8_t*) v5318;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5320 = *(const int8_t *)(v5319);
      vint32m2_t v5321 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5322 = __riscv_vwmul_vx_i16m1(v5311, v5317, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5323 = __riscv_vwadd_wv_i32m2(v5321, v5322, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5324 = __riscv_vwmul_vx_i16m1(v5314, v5320, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5325 = __riscv_vwadd_wv_i32m2(v5323, v5324, 8);
      v4804 = v5325;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5326 = v19 + 2128;
      const uint8_t* v5327 = (const uint8_t*) v5326;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5328 = __riscv_vle8_v_u8mf2(v5327, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5329 = __riscv_vand_vx_u8mf2(v5328, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5330 = __riscv_vzext_vf2_u16m1(v5329, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5331 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5330, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5332 = __riscv_vsrl_vx_u8mf2(v5328, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5333 = __riscv_vzext_vf2_u16m1(v5332, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5334 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5333, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5335 = v21 + 241;
      const int8_t* v5336 = (const int8_t*) v5335;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5337 = *(const int8_t *)(v5336);
      const uint8_t* v5338 = v21 + 257;
      const int8_t* v5339 = (const int8_t*) v5338;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5340 = *(const int8_t *)(v5339);
      vint32m2_t v5341 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5342 = __riscv_vwmul_vx_i16m1(v5331, v5337, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5343 = __riscv_vwadd_wv_i32m2(v5341, v5342, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5344 = __riscv_vwmul_vx_i16m1(v5334, v5340, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5345 = __riscv_vwadd_wv_i32m2(v5343, v5344, 8);
      v4802 = v5345;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5346 = v19 + 2136;
      const uint8_t* v5347 = (const uint8_t*) v5346;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5348 = __riscv_vle8_v_u8mf2(v5347, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5349 = __riscv_vand_vx_u8mf2(v5348, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5350 = __riscv_vzext_vf2_u16m1(v5349, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5351 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5350, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5352 = __riscv_vsrl_vx_u8mf2(v5348, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5353 = __riscv_vzext_vf2_u16m1(v5352, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5354 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5353, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5355 = v21 + 241;
      const int8_t* v5356 = (const int8_t*) v5355;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5357 = *(const int8_t *)(v5356);
      const uint8_t* v5358 = v21 + 257;
      const int8_t* v5359 = (const int8_t*) v5358;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5360 = *(const int8_t *)(v5359);
      vint32m2_t v5361 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5362 = __riscv_vwmul_vx_i16m1(v5351, v5357, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5363 = __riscv_vwadd_wv_i32m2(v5361, v5362, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5364 = __riscv_vwmul_vx_i16m1(v5354, v5360, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5365 = __riscv_vwadd_wv_i32m2(v5363, v5364, 8);
      v4804 = v5365;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5366 = v19 + 2144;
      const uint8_t* v5367 = (const uint8_t*) v5366;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5368 = __riscv_vle8_v_u8mf2(v5367, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5369 = __riscv_vand_vx_u8mf2(v5368, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5370 = __riscv_vzext_vf2_u16m1(v5369, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5371 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5370, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5372 = __riscv_vsrl_vx_u8mf2(v5368, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5373 = __riscv_vzext_vf2_u16m1(v5372, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5374 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5373, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5375 = v21 + 242;
      const int8_t* v5376 = (const int8_t*) v5375;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5377 = *(const int8_t *)(v5376);
      const uint8_t* v5378 = v21 + 258;
      const int8_t* v5379 = (const int8_t*) v5378;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5380 = *(const int8_t *)(v5379);
      vint32m2_t v5381 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5382 = __riscv_vwmul_vx_i16m1(v5371, v5377, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5383 = __riscv_vwadd_wv_i32m2(v5381, v5382, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5384 = __riscv_vwmul_vx_i16m1(v5374, v5380, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5385 = __riscv_vwadd_wv_i32m2(v5383, v5384, 8);
      v4802 = v5385;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5386 = v19 + 2152;
      const uint8_t* v5387 = (const uint8_t*) v5386;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5388 = __riscv_vle8_v_u8mf2(v5387, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5389 = __riscv_vand_vx_u8mf2(v5388, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5390 = __riscv_vzext_vf2_u16m1(v5389, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5391 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5390, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5392 = __riscv_vsrl_vx_u8mf2(v5388, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5393 = __riscv_vzext_vf2_u16m1(v5392, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5394 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5393, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5395 = v21 + 242;
      const int8_t* v5396 = (const int8_t*) v5395;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5397 = *(const int8_t *)(v5396);
      const uint8_t* v5398 = v21 + 258;
      const int8_t* v5399 = (const int8_t*) v5398;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5400 = *(const int8_t *)(v5399);
      vint32m2_t v5401 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5402 = __riscv_vwmul_vx_i16m1(v5391, v5397, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5403 = __riscv_vwadd_wv_i32m2(v5401, v5402, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5404 = __riscv_vwmul_vx_i16m1(v5394, v5400, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5405 = __riscv_vwadd_wv_i32m2(v5403, v5404, 8);
      v4804 = v5405;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5406 = v19 + 2160;
      const uint8_t* v5407 = (const uint8_t*) v5406;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5408 = __riscv_vle8_v_u8mf2(v5407, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5409 = __riscv_vand_vx_u8mf2(v5408, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5410 = __riscv_vzext_vf2_u16m1(v5409, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5411 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5410, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5412 = __riscv_vsrl_vx_u8mf2(v5408, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5413 = __riscv_vzext_vf2_u16m1(v5412, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5414 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5413, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5415 = v21 + 243;
      const int8_t* v5416 = (const int8_t*) v5415;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5417 = *(const int8_t *)(v5416);
      const uint8_t* v5418 = v21 + 259;
      const int8_t* v5419 = (const int8_t*) v5418;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5420 = *(const int8_t *)(v5419);
      vint32m2_t v5421 = v4802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5422 = __riscv_vwmul_vx_i16m1(v5411, v5417, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5423 = __riscv_vwadd_wv_i32m2(v5421, v5422, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5424 = __riscv_vwmul_vx_i16m1(v5414, v5420, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5425 = __riscv_vwadd_wv_i32m2(v5423, v5424, 8);
      v4802 = v5425;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v5426 = v19 + 2168;
      const uint8_t* v5427 = (const uint8_t*) v5426;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5428 = __riscv_vle8_v_u8mf2(v5427, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5429 = __riscv_vand_vx_u8mf2(v5428, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5430 = __riscv_vzext_vf2_u16m1(v5429, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5431 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5430, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5432 = __riscv_vsrl_vx_u8mf2(v5428, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v5433 = __riscv_vzext_vf2_u16m1(v5432, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v5434 = __riscv_vluxei16_v_i8mf2(weft_iq4_xs_repack_kvalues, v5433, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5435 = v21 + 243;
      const int8_t* v5436 = (const int8_t*) v5435;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5437 = *(const int8_t *)(v5436);
      const uint8_t* v5438 = v21 + 259;
      const int8_t* v5439 = (const int8_t*) v5438;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5440 = *(const int8_t *)(v5439);
      vint32m2_t v5441 = v4804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5442 = __riscv_vwmul_vx_i16m1(v5431, v5437, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5443 = __riscv_vwadd_wv_i32m2(v5441, v5442, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v5444 = __riscv_vwmul_vx_i16m1(v5434, v5440, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v5445 = __riscv_vwadd_wv_i32m2(v5443, v5444, 8);
      v4804 = v5445;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=subblock_scale_fold
      vint32m2_t v5446 = v4802;
      vint32m2_t v5447 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v5448 = __riscv_vmacc_vv_i32m2(v5447, v4787, v5446, 8);
      v24 = v5448;
      vint32m2_t v5449 = v4804;
      vint32m2_t v5450 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmacc_vv_i32m2
      vint32m2_t v5451 = __riscv_vmacc_vv_i32m2(v5450, v4801, v5449, 8);
      v26 = v5451;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v5452 = (const _Float16*) v19;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v5453 = __riscv_vle16_v_f16m1(v5452, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v5454 = __riscv_vfwcvt_f_f_v_f32m2(v5453, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v5455 = __riscv_vfmul_vf_f32m2(v5454, v23, 8);
      vint32m2_t v5456 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v5457 = __riscv_vfcvt_f_x_v_f32m2(v5456, 8);
      vfloat32m2_t v5458 = v13;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v5459 = __riscv_vfmacc_vv_f32m2(v5458, v5457, v5455, 8);
      v13 = v5459;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v5460 = v19 + 16;
      const _Float16* v5461 = (const _Float16*) v5460;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v5462 = __riscv_vle16_v_f16m1(v5461, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v5463 = __riscv_vfwcvt_f_f_v_f32m2(v5462, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v5464 = __riscv_vfmul_vf_f32m2(v5463, v23, 8);
      vint32m2_t v5465 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v5466 = __riscv_vfcvt_f_x_v_f32m2(v5465, 8);
      vfloat32m2_t v5467 = v15;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v5468 = __riscv_vfmacc_vv_f32m2(v5467, v5466, v5464, 8);
      v15 = v5468;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v5469 = v9 * 16;
    float* v5470 = v2 + v5469;
    vfloat32m2_t v5471 = v13;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v5470, v5471, 8);
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v5472 = v9 * 16;
    size_t v5473 = v5472 + 8;
    float* v5474 = v2 + v5473;
    vfloat32m2_t v5475 = v15;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v5474, v5475, 8);
  }
  return;
}


