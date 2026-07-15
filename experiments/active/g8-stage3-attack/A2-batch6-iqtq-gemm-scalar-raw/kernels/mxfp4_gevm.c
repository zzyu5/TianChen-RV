#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemv_mxfp4_q8_0_kernel_ggml_repack_gemv_mxfp4_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_mxfp4_repack_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=col_group_count
  size_t v8 = v5 / 16;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_group_base
    size_t v10 = v9 * v7;
    size_t v11 = v10 * 272;
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
      size_t v18 = v17 * 272;
      const uint8_t* v19 = v12 + v18;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
      size_t v20 = v17 * 34;
      const uint8_t* v21 = v4 + v20;
      vint32m2_t v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v23 = __riscv_vmv_v_x_i32m2(0, 8);
      v22 = v23;
      vint32m2_t v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v25 = __riscv_vmv_v_x_i32m2(0, 8);
      v24 = v25;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v26 = v19 + 16;
      const uint8_t* v27 = (const uint8_t*) v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v28 = __riscv_vle8_v_u8mf2(v27, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v29 = __riscv_vand_vx_u8mf2(v28, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v30 = __riscv_vzext_vf2_u16m1(v29, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v31 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v32 = __riscv_vsrl_vx_u8mf2(v28, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v33 = __riscv_vzext_vf2_u16m1(v32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v34 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v33, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v35 = v21 + 2;
      const int8_t* v36 = (const int8_t*) v35;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v37 = *(const int8_t *)(v36);
      const uint8_t* v38 = v21 + 18;
      const int8_t* v39 = (const int8_t*) v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v40 = *(const int8_t *)(v39);
      vint32m2_t v41 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v42 = __riscv_vwmul_vx_i16m1(v31, v37, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v43 = __riscv_vwadd_wv_i32m2(v41, v42, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v44 = __riscv_vwmul_vx_i16m1(v34, v40, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v45 = __riscv_vwadd_wv_i32m2(v43, v44, 8);
      v22 = v45;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v46 = v19 + 24;
      const uint8_t* v47 = (const uint8_t*) v46;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v48 = __riscv_vle8_v_u8mf2(v47, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v49 = __riscv_vand_vx_u8mf2(v48, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v50 = __riscv_vzext_vf2_u16m1(v49, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v51 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v50, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v52 = __riscv_vsrl_vx_u8mf2(v48, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v53 = __riscv_vzext_vf2_u16m1(v52, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v54 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v53, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v55 = v21 + 2;
      const int8_t* v56 = (const int8_t*) v55;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v57 = *(const int8_t *)(v56);
      const uint8_t* v58 = v21 + 18;
      const int8_t* v59 = (const int8_t*) v58;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v60 = *(const int8_t *)(v59);
      vint32m2_t v61 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v62 = __riscv_vwmul_vx_i16m1(v51, v57, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v63 = __riscv_vwadd_wv_i32m2(v61, v62, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v64 = __riscv_vwmul_vx_i16m1(v54, v60, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v65 = __riscv_vwadd_wv_i32m2(v63, v64, 8);
      v24 = v65;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v66 = v19 + 32;
      const uint8_t* v67 = (const uint8_t*) v66;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v68 = __riscv_vle8_v_u8mf2(v67, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v69 = __riscv_vand_vx_u8mf2(v68, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v70 = __riscv_vzext_vf2_u16m1(v69, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v71 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v70, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v72 = __riscv_vsrl_vx_u8mf2(v68, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v73 = __riscv_vzext_vf2_u16m1(v72, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v74 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v73, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v75 = v21 + 3;
      const int8_t* v76 = (const int8_t*) v75;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v77 = *(const int8_t *)(v76);
      const uint8_t* v78 = v21 + 19;
      const int8_t* v79 = (const int8_t*) v78;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v80 = *(const int8_t *)(v79);
      vint32m2_t v81 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v82 = __riscv_vwmul_vx_i16m1(v71, v77, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v83 = __riscv_vwadd_wv_i32m2(v81, v82, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v84 = __riscv_vwmul_vx_i16m1(v74, v80, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v85 = __riscv_vwadd_wv_i32m2(v83, v84, 8);
      v22 = v85;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v86 = v19 + 40;
      const uint8_t* v87 = (const uint8_t*) v86;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v88 = __riscv_vle8_v_u8mf2(v87, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v89 = __riscv_vand_vx_u8mf2(v88, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v90 = __riscv_vzext_vf2_u16m1(v89, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v91 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v90, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v92 = __riscv_vsrl_vx_u8mf2(v88, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v93 = __riscv_vzext_vf2_u16m1(v92, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v94 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v93, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v95 = v21 + 3;
      const int8_t* v96 = (const int8_t*) v95;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v97 = *(const int8_t *)(v96);
      const uint8_t* v98 = v21 + 19;
      const int8_t* v99 = (const int8_t*) v98;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v100 = *(const int8_t *)(v99);
      vint32m2_t v101 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v102 = __riscv_vwmul_vx_i16m1(v91, v97, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v103 = __riscv_vwadd_wv_i32m2(v101, v102, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v104 = __riscv_vwmul_vx_i16m1(v94, v100, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v105 = __riscv_vwadd_wv_i32m2(v103, v104, 8);
      v24 = v105;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v106 = v19 + 48;
      const uint8_t* v107 = (const uint8_t*) v106;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v108 = __riscv_vle8_v_u8mf2(v107, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v109 = __riscv_vand_vx_u8mf2(v108, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v110 = __riscv_vzext_vf2_u16m1(v109, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v111 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v110, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v112 = __riscv_vsrl_vx_u8mf2(v108, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v113 = __riscv_vzext_vf2_u16m1(v112, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v114 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v113, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v115 = v21 + 4;
      const int8_t* v116 = (const int8_t*) v115;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v117 = *(const int8_t *)(v116);
      const uint8_t* v118 = v21 + 20;
      const int8_t* v119 = (const int8_t*) v118;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v120 = *(const int8_t *)(v119);
      vint32m2_t v121 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v122 = __riscv_vwmul_vx_i16m1(v111, v117, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v123 = __riscv_vwadd_wv_i32m2(v121, v122, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v124 = __riscv_vwmul_vx_i16m1(v114, v120, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v125 = __riscv_vwadd_wv_i32m2(v123, v124, 8);
      v22 = v125;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v126 = v19 + 56;
      const uint8_t* v127 = (const uint8_t*) v126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v128 = __riscv_vle8_v_u8mf2(v127, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v129 = __riscv_vand_vx_u8mf2(v128, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v130 = __riscv_vzext_vf2_u16m1(v129, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v131 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v130, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v132 = __riscv_vsrl_vx_u8mf2(v128, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v133 = __riscv_vzext_vf2_u16m1(v132, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v134 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v133, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v135 = v21 + 4;
      const int8_t* v136 = (const int8_t*) v135;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v137 = *(const int8_t *)(v136);
      const uint8_t* v138 = v21 + 20;
      const int8_t* v139 = (const int8_t*) v138;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v140 = *(const int8_t *)(v139);
      vint32m2_t v141 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v142 = __riscv_vwmul_vx_i16m1(v131, v137, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v143 = __riscv_vwadd_wv_i32m2(v141, v142, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v144 = __riscv_vwmul_vx_i16m1(v134, v140, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v145 = __riscv_vwadd_wv_i32m2(v143, v144, 8);
      v24 = v145;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v146 = v19 + 64;
      const uint8_t* v147 = (const uint8_t*) v146;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v148 = __riscv_vle8_v_u8mf2(v147, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v149 = __riscv_vand_vx_u8mf2(v148, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v150 = __riscv_vzext_vf2_u16m1(v149, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v151 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v150, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v152 = __riscv_vsrl_vx_u8mf2(v148, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v153 = __riscv_vzext_vf2_u16m1(v152, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v154 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v153, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v155 = v21 + 5;
      const int8_t* v156 = (const int8_t*) v155;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v157 = *(const int8_t *)(v156);
      const uint8_t* v158 = v21 + 21;
      const int8_t* v159 = (const int8_t*) v158;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v160 = *(const int8_t *)(v159);
      vint32m2_t v161 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v162 = __riscv_vwmul_vx_i16m1(v151, v157, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v163 = __riscv_vwadd_wv_i32m2(v161, v162, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v164 = __riscv_vwmul_vx_i16m1(v154, v160, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v165 = __riscv_vwadd_wv_i32m2(v163, v164, 8);
      v22 = v165;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v166 = v19 + 72;
      const uint8_t* v167 = (const uint8_t*) v166;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v168 = __riscv_vle8_v_u8mf2(v167, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v169 = __riscv_vand_vx_u8mf2(v168, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v170 = __riscv_vzext_vf2_u16m1(v169, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v171 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v170, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v172 = __riscv_vsrl_vx_u8mf2(v168, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v173 = __riscv_vzext_vf2_u16m1(v172, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v174 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v173, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v175 = v21 + 5;
      const int8_t* v176 = (const int8_t*) v175;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v177 = *(const int8_t *)(v176);
      const uint8_t* v178 = v21 + 21;
      const int8_t* v179 = (const int8_t*) v178;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v180 = *(const int8_t *)(v179);
      vint32m2_t v181 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v182 = __riscv_vwmul_vx_i16m1(v171, v177, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v183 = __riscv_vwadd_wv_i32m2(v181, v182, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v184 = __riscv_vwmul_vx_i16m1(v174, v180, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v185 = __riscv_vwadd_wv_i32m2(v183, v184, 8);
      v24 = v185;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v186 = v19 + 80;
      const uint8_t* v187 = (const uint8_t*) v186;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v188 = __riscv_vle8_v_u8mf2(v187, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v189 = __riscv_vand_vx_u8mf2(v188, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v190 = __riscv_vzext_vf2_u16m1(v189, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v191 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v190, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v192 = __riscv_vsrl_vx_u8mf2(v188, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v193 = __riscv_vzext_vf2_u16m1(v192, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v194 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v193, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v195 = v21 + 6;
      const int8_t* v196 = (const int8_t*) v195;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v197 = *(const int8_t *)(v196);
      const uint8_t* v198 = v21 + 22;
      const int8_t* v199 = (const int8_t*) v198;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v200 = *(const int8_t *)(v199);
      vint32m2_t v201 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v202 = __riscv_vwmul_vx_i16m1(v191, v197, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v203 = __riscv_vwadd_wv_i32m2(v201, v202, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v204 = __riscv_vwmul_vx_i16m1(v194, v200, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v205 = __riscv_vwadd_wv_i32m2(v203, v204, 8);
      v22 = v205;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v206 = v19 + 88;
      const uint8_t* v207 = (const uint8_t*) v206;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v208 = __riscv_vle8_v_u8mf2(v207, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v209 = __riscv_vand_vx_u8mf2(v208, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v210 = __riscv_vzext_vf2_u16m1(v209, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v211 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v210, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v212 = __riscv_vsrl_vx_u8mf2(v208, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v213 = __riscv_vzext_vf2_u16m1(v212, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v214 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v213, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v215 = v21 + 6;
      const int8_t* v216 = (const int8_t*) v215;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v217 = *(const int8_t *)(v216);
      const uint8_t* v218 = v21 + 22;
      const int8_t* v219 = (const int8_t*) v218;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v220 = *(const int8_t *)(v219);
      vint32m2_t v221 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v222 = __riscv_vwmul_vx_i16m1(v211, v217, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v223 = __riscv_vwadd_wv_i32m2(v221, v222, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v224 = __riscv_vwmul_vx_i16m1(v214, v220, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v225 = __riscv_vwadd_wv_i32m2(v223, v224, 8);
      v24 = v225;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v226 = v19 + 96;
      const uint8_t* v227 = (const uint8_t*) v226;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v228 = __riscv_vle8_v_u8mf2(v227, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v229 = __riscv_vand_vx_u8mf2(v228, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v230 = __riscv_vzext_vf2_u16m1(v229, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v231 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v230, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v232 = __riscv_vsrl_vx_u8mf2(v228, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v233 = __riscv_vzext_vf2_u16m1(v232, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v234 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v233, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v235 = v21 + 7;
      const int8_t* v236 = (const int8_t*) v235;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v237 = *(const int8_t *)(v236);
      const uint8_t* v238 = v21 + 23;
      const int8_t* v239 = (const int8_t*) v238;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v240 = *(const int8_t *)(v239);
      vint32m2_t v241 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v242 = __riscv_vwmul_vx_i16m1(v231, v237, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v243 = __riscv_vwadd_wv_i32m2(v241, v242, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v244 = __riscv_vwmul_vx_i16m1(v234, v240, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v245 = __riscv_vwadd_wv_i32m2(v243, v244, 8);
      v22 = v245;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v246 = v19 + 104;
      const uint8_t* v247 = (const uint8_t*) v246;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v248 = __riscv_vle8_v_u8mf2(v247, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v249 = __riscv_vand_vx_u8mf2(v248, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v250 = __riscv_vzext_vf2_u16m1(v249, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v251 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v250, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v252 = __riscv_vsrl_vx_u8mf2(v248, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v253 = __riscv_vzext_vf2_u16m1(v252, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v254 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v253, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v255 = v21 + 7;
      const int8_t* v256 = (const int8_t*) v255;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v257 = *(const int8_t *)(v256);
      const uint8_t* v258 = v21 + 23;
      const int8_t* v259 = (const int8_t*) v258;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v260 = *(const int8_t *)(v259);
      vint32m2_t v261 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v262 = __riscv_vwmul_vx_i16m1(v251, v257, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v263 = __riscv_vwadd_wv_i32m2(v261, v262, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v264 = __riscv_vwmul_vx_i16m1(v254, v260, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v265 = __riscv_vwadd_wv_i32m2(v263, v264, 8);
      v24 = v265;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v266 = v19 + 112;
      const uint8_t* v267 = (const uint8_t*) v266;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v268 = __riscv_vle8_v_u8mf2(v267, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v269 = __riscv_vand_vx_u8mf2(v268, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v270 = __riscv_vzext_vf2_u16m1(v269, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v271 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v270, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v272 = __riscv_vsrl_vx_u8mf2(v268, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v273 = __riscv_vzext_vf2_u16m1(v272, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v274 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v273, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v275 = v21 + 8;
      const int8_t* v276 = (const int8_t*) v275;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v277 = *(const int8_t *)(v276);
      const uint8_t* v278 = v21 + 24;
      const int8_t* v279 = (const int8_t*) v278;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v280 = *(const int8_t *)(v279);
      vint32m2_t v281 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v282 = __riscv_vwmul_vx_i16m1(v271, v277, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v283 = __riscv_vwadd_wv_i32m2(v281, v282, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v284 = __riscv_vwmul_vx_i16m1(v274, v280, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v285 = __riscv_vwadd_wv_i32m2(v283, v284, 8);
      v22 = v285;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v286 = v19 + 120;
      const uint8_t* v287 = (const uint8_t*) v286;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v288 = __riscv_vle8_v_u8mf2(v287, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v289 = __riscv_vand_vx_u8mf2(v288, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v290 = __riscv_vzext_vf2_u16m1(v289, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v291 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v290, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v292 = __riscv_vsrl_vx_u8mf2(v288, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v293 = __riscv_vzext_vf2_u16m1(v292, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v294 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v293, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v295 = v21 + 8;
      const int8_t* v296 = (const int8_t*) v295;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v297 = *(const int8_t *)(v296);
      const uint8_t* v298 = v21 + 24;
      const int8_t* v299 = (const int8_t*) v298;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v300 = *(const int8_t *)(v299);
      vint32m2_t v301 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v302 = __riscv_vwmul_vx_i16m1(v291, v297, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v303 = __riscv_vwadd_wv_i32m2(v301, v302, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v304 = __riscv_vwmul_vx_i16m1(v294, v300, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v305 = __riscv_vwadd_wv_i32m2(v303, v304, 8);
      v24 = v305;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v306 = v19 + 128;
      const uint8_t* v307 = (const uint8_t*) v306;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v308 = __riscv_vle8_v_u8mf2(v307, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v309 = __riscv_vand_vx_u8mf2(v308, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v310 = __riscv_vzext_vf2_u16m1(v309, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v311 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v310, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v312 = __riscv_vsrl_vx_u8mf2(v308, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v313 = __riscv_vzext_vf2_u16m1(v312, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v314 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v313, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v315 = v21 + 9;
      const int8_t* v316 = (const int8_t*) v315;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v317 = *(const int8_t *)(v316);
      const uint8_t* v318 = v21 + 25;
      const int8_t* v319 = (const int8_t*) v318;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v320 = *(const int8_t *)(v319);
      vint32m2_t v321 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v322 = __riscv_vwmul_vx_i16m1(v311, v317, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v323 = __riscv_vwadd_wv_i32m2(v321, v322, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v324 = __riscv_vwmul_vx_i16m1(v314, v320, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v325 = __riscv_vwadd_wv_i32m2(v323, v324, 8);
      v22 = v325;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v326 = v19 + 136;
      const uint8_t* v327 = (const uint8_t*) v326;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v328 = __riscv_vle8_v_u8mf2(v327, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v329 = __riscv_vand_vx_u8mf2(v328, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v330 = __riscv_vzext_vf2_u16m1(v329, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v331 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v330, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v332 = __riscv_vsrl_vx_u8mf2(v328, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v333 = __riscv_vzext_vf2_u16m1(v332, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v334 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v333, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v335 = v21 + 9;
      const int8_t* v336 = (const int8_t*) v335;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v337 = *(const int8_t *)(v336);
      const uint8_t* v338 = v21 + 25;
      const int8_t* v339 = (const int8_t*) v338;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v340 = *(const int8_t *)(v339);
      vint32m2_t v341 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v342 = __riscv_vwmul_vx_i16m1(v331, v337, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v343 = __riscv_vwadd_wv_i32m2(v341, v342, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v344 = __riscv_vwmul_vx_i16m1(v334, v340, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v345 = __riscv_vwadd_wv_i32m2(v343, v344, 8);
      v24 = v345;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v346 = v19 + 144;
      const uint8_t* v347 = (const uint8_t*) v346;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v348 = __riscv_vle8_v_u8mf2(v347, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v349 = __riscv_vand_vx_u8mf2(v348, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v350 = __riscv_vzext_vf2_u16m1(v349, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v351 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v350, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v352 = __riscv_vsrl_vx_u8mf2(v348, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v353 = __riscv_vzext_vf2_u16m1(v352, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v354 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v353, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v355 = v21 + 10;
      const int8_t* v356 = (const int8_t*) v355;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v357 = *(const int8_t *)(v356);
      const uint8_t* v358 = v21 + 26;
      const int8_t* v359 = (const int8_t*) v358;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v360 = *(const int8_t *)(v359);
      vint32m2_t v361 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v362 = __riscv_vwmul_vx_i16m1(v351, v357, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v363 = __riscv_vwadd_wv_i32m2(v361, v362, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v364 = __riscv_vwmul_vx_i16m1(v354, v360, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v365 = __riscv_vwadd_wv_i32m2(v363, v364, 8);
      v22 = v365;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v366 = v19 + 152;
      const uint8_t* v367 = (const uint8_t*) v366;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v368 = __riscv_vle8_v_u8mf2(v367, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v369 = __riscv_vand_vx_u8mf2(v368, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v370 = __riscv_vzext_vf2_u16m1(v369, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v371 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v370, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v372 = __riscv_vsrl_vx_u8mf2(v368, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v373 = __riscv_vzext_vf2_u16m1(v372, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v374 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v373, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v375 = v21 + 10;
      const int8_t* v376 = (const int8_t*) v375;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v377 = *(const int8_t *)(v376);
      const uint8_t* v378 = v21 + 26;
      const int8_t* v379 = (const int8_t*) v378;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v380 = *(const int8_t *)(v379);
      vint32m2_t v381 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v382 = __riscv_vwmul_vx_i16m1(v371, v377, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v383 = __riscv_vwadd_wv_i32m2(v381, v382, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v384 = __riscv_vwmul_vx_i16m1(v374, v380, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v385 = __riscv_vwadd_wv_i32m2(v383, v384, 8);
      v24 = v385;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v386 = v19 + 160;
      const uint8_t* v387 = (const uint8_t*) v386;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v388 = __riscv_vle8_v_u8mf2(v387, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v389 = __riscv_vand_vx_u8mf2(v388, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v390 = __riscv_vzext_vf2_u16m1(v389, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v391 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v390, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v392 = __riscv_vsrl_vx_u8mf2(v388, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v393 = __riscv_vzext_vf2_u16m1(v392, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v394 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v393, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v395 = v21 + 11;
      const int8_t* v396 = (const int8_t*) v395;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v397 = *(const int8_t *)(v396);
      const uint8_t* v398 = v21 + 27;
      const int8_t* v399 = (const int8_t*) v398;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v400 = *(const int8_t *)(v399);
      vint32m2_t v401 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v402 = __riscv_vwmul_vx_i16m1(v391, v397, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v403 = __riscv_vwadd_wv_i32m2(v401, v402, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v404 = __riscv_vwmul_vx_i16m1(v394, v400, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v405 = __riscv_vwadd_wv_i32m2(v403, v404, 8);
      v22 = v405;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v406 = v19 + 168;
      const uint8_t* v407 = (const uint8_t*) v406;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v408 = __riscv_vle8_v_u8mf2(v407, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v409 = __riscv_vand_vx_u8mf2(v408, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v410 = __riscv_vzext_vf2_u16m1(v409, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v411 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v410, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v412 = __riscv_vsrl_vx_u8mf2(v408, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v413 = __riscv_vzext_vf2_u16m1(v412, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v414 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v413, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v415 = v21 + 11;
      const int8_t* v416 = (const int8_t*) v415;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v417 = *(const int8_t *)(v416);
      const uint8_t* v418 = v21 + 27;
      const int8_t* v419 = (const int8_t*) v418;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v420 = *(const int8_t *)(v419);
      vint32m2_t v421 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v422 = __riscv_vwmul_vx_i16m1(v411, v417, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v423 = __riscv_vwadd_wv_i32m2(v421, v422, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v424 = __riscv_vwmul_vx_i16m1(v414, v420, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v425 = __riscv_vwadd_wv_i32m2(v423, v424, 8);
      v24 = v425;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v426 = v19 + 176;
      const uint8_t* v427 = (const uint8_t*) v426;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v428 = __riscv_vle8_v_u8mf2(v427, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v429 = __riscv_vand_vx_u8mf2(v428, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v430 = __riscv_vzext_vf2_u16m1(v429, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v431 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v430, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v432 = __riscv_vsrl_vx_u8mf2(v428, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v433 = __riscv_vzext_vf2_u16m1(v432, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v434 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v433, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v435 = v21 + 12;
      const int8_t* v436 = (const int8_t*) v435;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v437 = *(const int8_t *)(v436);
      const uint8_t* v438 = v21 + 28;
      const int8_t* v439 = (const int8_t*) v438;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v440 = *(const int8_t *)(v439);
      vint32m2_t v441 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v442 = __riscv_vwmul_vx_i16m1(v431, v437, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v443 = __riscv_vwadd_wv_i32m2(v441, v442, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v444 = __riscv_vwmul_vx_i16m1(v434, v440, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v445 = __riscv_vwadd_wv_i32m2(v443, v444, 8);
      v22 = v445;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v446 = v19 + 184;
      const uint8_t* v447 = (const uint8_t*) v446;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v448 = __riscv_vle8_v_u8mf2(v447, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v449 = __riscv_vand_vx_u8mf2(v448, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v450 = __riscv_vzext_vf2_u16m1(v449, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v451 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v450, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v452 = __riscv_vsrl_vx_u8mf2(v448, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v453 = __riscv_vzext_vf2_u16m1(v452, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v454 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v453, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v455 = v21 + 12;
      const int8_t* v456 = (const int8_t*) v455;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v457 = *(const int8_t *)(v456);
      const uint8_t* v458 = v21 + 28;
      const int8_t* v459 = (const int8_t*) v458;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v460 = *(const int8_t *)(v459);
      vint32m2_t v461 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v462 = __riscv_vwmul_vx_i16m1(v451, v457, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v463 = __riscv_vwadd_wv_i32m2(v461, v462, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v464 = __riscv_vwmul_vx_i16m1(v454, v460, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v465 = __riscv_vwadd_wv_i32m2(v463, v464, 8);
      v24 = v465;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v466 = v19 + 192;
      const uint8_t* v467 = (const uint8_t*) v466;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v468 = __riscv_vle8_v_u8mf2(v467, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v469 = __riscv_vand_vx_u8mf2(v468, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v470 = __riscv_vzext_vf2_u16m1(v469, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v471 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v470, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v472 = __riscv_vsrl_vx_u8mf2(v468, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v473 = __riscv_vzext_vf2_u16m1(v472, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v474 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v473, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v475 = v21 + 13;
      const int8_t* v476 = (const int8_t*) v475;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v477 = *(const int8_t *)(v476);
      const uint8_t* v478 = v21 + 29;
      const int8_t* v479 = (const int8_t*) v478;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v480 = *(const int8_t *)(v479);
      vint32m2_t v481 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v482 = __riscv_vwmul_vx_i16m1(v471, v477, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v483 = __riscv_vwadd_wv_i32m2(v481, v482, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v484 = __riscv_vwmul_vx_i16m1(v474, v480, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v485 = __riscv_vwadd_wv_i32m2(v483, v484, 8);
      v22 = v485;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v486 = v19 + 200;
      const uint8_t* v487 = (const uint8_t*) v486;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v488 = __riscv_vle8_v_u8mf2(v487, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v489 = __riscv_vand_vx_u8mf2(v488, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v490 = __riscv_vzext_vf2_u16m1(v489, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v491 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v490, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v492 = __riscv_vsrl_vx_u8mf2(v488, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v493 = __riscv_vzext_vf2_u16m1(v492, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v494 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v493, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v495 = v21 + 13;
      const int8_t* v496 = (const int8_t*) v495;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v497 = *(const int8_t *)(v496);
      const uint8_t* v498 = v21 + 29;
      const int8_t* v499 = (const int8_t*) v498;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v500 = *(const int8_t *)(v499);
      vint32m2_t v501 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v502 = __riscv_vwmul_vx_i16m1(v491, v497, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v503 = __riscv_vwadd_wv_i32m2(v501, v502, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v504 = __riscv_vwmul_vx_i16m1(v494, v500, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v505 = __riscv_vwadd_wv_i32m2(v503, v504, 8);
      v24 = v505;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v506 = v19 + 208;
      const uint8_t* v507 = (const uint8_t*) v506;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v508 = __riscv_vle8_v_u8mf2(v507, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v509 = __riscv_vand_vx_u8mf2(v508, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v510 = __riscv_vzext_vf2_u16m1(v509, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v511 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v510, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v512 = __riscv_vsrl_vx_u8mf2(v508, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v513 = __riscv_vzext_vf2_u16m1(v512, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v514 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v513, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v515 = v21 + 14;
      const int8_t* v516 = (const int8_t*) v515;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v517 = *(const int8_t *)(v516);
      const uint8_t* v518 = v21 + 30;
      const int8_t* v519 = (const int8_t*) v518;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v520 = *(const int8_t *)(v519);
      vint32m2_t v521 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v522 = __riscv_vwmul_vx_i16m1(v511, v517, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v523 = __riscv_vwadd_wv_i32m2(v521, v522, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v524 = __riscv_vwmul_vx_i16m1(v514, v520, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v525 = __riscv_vwadd_wv_i32m2(v523, v524, 8);
      v22 = v525;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v526 = v19 + 216;
      const uint8_t* v527 = (const uint8_t*) v526;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v528 = __riscv_vle8_v_u8mf2(v527, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v529 = __riscv_vand_vx_u8mf2(v528, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v530 = __riscv_vzext_vf2_u16m1(v529, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v531 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v530, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v532 = __riscv_vsrl_vx_u8mf2(v528, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v533 = __riscv_vzext_vf2_u16m1(v532, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v534 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v533, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v535 = v21 + 14;
      const int8_t* v536 = (const int8_t*) v535;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v537 = *(const int8_t *)(v536);
      const uint8_t* v538 = v21 + 30;
      const int8_t* v539 = (const int8_t*) v538;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v540 = *(const int8_t *)(v539);
      vint32m2_t v541 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v542 = __riscv_vwmul_vx_i16m1(v531, v537, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v543 = __riscv_vwadd_wv_i32m2(v541, v542, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v544 = __riscv_vwmul_vx_i16m1(v534, v540, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v545 = __riscv_vwadd_wv_i32m2(v543, v544, 8);
      v24 = v545;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v546 = v19 + 224;
      const uint8_t* v547 = (const uint8_t*) v546;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v548 = __riscv_vle8_v_u8mf2(v547, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v549 = __riscv_vand_vx_u8mf2(v548, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v550 = __riscv_vzext_vf2_u16m1(v549, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v551 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v550, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v552 = __riscv_vsrl_vx_u8mf2(v548, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v553 = __riscv_vzext_vf2_u16m1(v552, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v554 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v553, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v555 = v21 + 15;
      const int8_t* v556 = (const int8_t*) v555;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v557 = *(const int8_t *)(v556);
      const uint8_t* v558 = v21 + 31;
      const int8_t* v559 = (const int8_t*) v558;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v560 = *(const int8_t *)(v559);
      vint32m2_t v561 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v562 = __riscv_vwmul_vx_i16m1(v551, v557, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v563 = __riscv_vwadd_wv_i32m2(v561, v562, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v564 = __riscv_vwmul_vx_i16m1(v554, v560, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v565 = __riscv_vwadd_wv_i32m2(v563, v564, 8);
      v22 = v565;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v566 = v19 + 232;
      const uint8_t* v567 = (const uint8_t*) v566;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v568 = __riscv_vle8_v_u8mf2(v567, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v569 = __riscv_vand_vx_u8mf2(v568, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v570 = __riscv_vzext_vf2_u16m1(v569, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v571 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v570, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v572 = __riscv_vsrl_vx_u8mf2(v568, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v573 = __riscv_vzext_vf2_u16m1(v572, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v574 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v573, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v575 = v21 + 15;
      const int8_t* v576 = (const int8_t*) v575;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v577 = *(const int8_t *)(v576);
      const uint8_t* v578 = v21 + 31;
      const int8_t* v579 = (const int8_t*) v578;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v580 = *(const int8_t *)(v579);
      vint32m2_t v581 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v582 = __riscv_vwmul_vx_i16m1(v571, v577, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v583 = __riscv_vwadd_wv_i32m2(v581, v582, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v584 = __riscv_vwmul_vx_i16m1(v574, v580, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v585 = __riscv_vwadd_wv_i32m2(v583, v584, 8);
      v24 = v585;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v586 = v19 + 240;
      const uint8_t* v587 = (const uint8_t*) v586;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v588 = __riscv_vle8_v_u8mf2(v587, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v589 = __riscv_vand_vx_u8mf2(v588, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v590 = __riscv_vzext_vf2_u16m1(v589, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v591 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v590, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v592 = __riscv_vsrl_vx_u8mf2(v588, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v593 = __riscv_vzext_vf2_u16m1(v592, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v594 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v593, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v595 = v21 + 16;
      const int8_t* v596 = (const int8_t*) v595;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v597 = *(const int8_t *)(v596);
      const uint8_t* v598 = v21 + 32;
      const int8_t* v599 = (const int8_t*) v598;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v600 = *(const int8_t *)(v599);
      vint32m2_t v601 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v602 = __riscv_vwmul_vx_i16m1(v591, v597, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v603 = __riscv_vwadd_wv_i32m2(v601, v602, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v604 = __riscv_vwmul_vx_i16m1(v594, v600, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v605 = __riscv_vwadd_wv_i32m2(v603, v604, 8);
      v22 = v605;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v606 = v19 + 248;
      const uint8_t* v607 = (const uint8_t*) v606;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v608 = __riscv_vle8_v_u8mf2(v607, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v609 = __riscv_vand_vx_u8mf2(v608, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v610 = __riscv_vzext_vf2_u16m1(v609, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v611 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v610, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v612 = __riscv_vsrl_vx_u8mf2(v608, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v613 = __riscv_vzext_vf2_u16m1(v612, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v614 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v613, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v615 = v21 + 16;
      const int8_t* v616 = (const int8_t*) v615;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v617 = *(const int8_t *)(v616);
      const uint8_t* v618 = v21 + 32;
      const int8_t* v619 = (const int8_t*) v618;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v620 = *(const int8_t *)(v619);
      vint32m2_t v621 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v622 = __riscv_vwmul_vx_i16m1(v611, v617, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v623 = __riscv_vwadd_wv_i32m2(v621, v622, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v624 = __riscv_vwmul_vx_i16m1(v614, v620, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v625 = __riscv_vwadd_wv_i32m2(v623, v624, 8);
      v24 = v625;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v626 = v19 + 256;
      const uint8_t* v627 = (const uint8_t*) v626;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v628 = __riscv_vle8_v_u8mf2(v627, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v629 = __riscv_vand_vx_u8mf2(v628, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v630 = __riscv_vzext_vf2_u16m1(v629, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v631 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v630, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v632 = __riscv_vsrl_vx_u8mf2(v628, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v633 = __riscv_vzext_vf2_u16m1(v632, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v634 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v633, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v635 = v21 + 17;
      const int8_t* v636 = (const int8_t*) v635;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v637 = *(const int8_t *)(v636);
      const uint8_t* v638 = v21 + 33;
      const int8_t* v639 = (const int8_t*) v638;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v640 = *(const int8_t *)(v639);
      vint32m2_t v641 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v642 = __riscv_vwmul_vx_i16m1(v631, v637, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v643 = __riscv_vwadd_wv_i32m2(v641, v642, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v644 = __riscv_vwmul_vx_i16m1(v634, v640, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v645 = __riscv_vwadd_wv_i32m2(v643, v644, 8);
      v22 = v645;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v646 = v19 + 264;
      const uint8_t* v647 = (const uint8_t*) v646;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v648 = __riscv_vle8_v_u8mf2(v647, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v649 = __riscv_vand_vx_u8mf2(v648, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v650 = __riscv_vzext_vf2_u16m1(v649, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v651 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v650, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v652 = __riscv_vsrl_vx_u8mf2(v648, 0x04, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_index_offset
      vuint16m1_t v653 = __riscv_vzext_vf2_u16m1(v652, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_gather
      vint8mf2_t v654 = __riscv_vluxei16_v_i8mf2(weft_mxfp4_repack_kvalues, v653, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v655 = v21 + 17;
      const int8_t* v656 = (const int8_t*) v655;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v657 = *(const int8_t *)(v656);
      const uint8_t* v658 = v21 + 33;
      const int8_t* v659 = (const int8_t*) v658;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v660 = *(const int8_t *)(v659);
      vint32m2_t v661 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v662 = __riscv_vwmul_vx_i16m1(v651, v657, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v663 = __riscv_vwadd_wv_i32m2(v661, v662, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
      vint16m1_t v664 = __riscv_vwmul_vx_i16m1(v654, v660, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
      vint32m2_t v665 = __riscv_vwadd_wv_i32m2(v663, v664, 8);
      v24 = v665;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v666 = (const _Float16*) v21;
      _Float16 v667 = *(const _Float16 *)(v666);
      float v668 = (float) v667;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_e8m0_scale_addr
      const uint8_t* v669 = (const uint8_t*) v19;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_exponent_load
      vuint8mf2_t v670 = __riscv_vle8_v_u8mf2(v669, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_widen_u32
      vuint32m2_t v671 = __riscv_vzext_vf4_u32m2(v670, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_to_fp32_half_bits
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u32m2
      vuint32m2_t v672 = __riscv_vand_vx_u32m2(v671, 0x1F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u32m2
      vuint32m2_t v673 = __riscv_vmv_v_x_u32m2(0x00200000, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vv_u32m2
      vuint32m2_t v674 = __riscv_vsll_vv_u32m2(v673, v672, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_u32m2
      vuint32m2_t v675 = __riscv_vsub_vx_u32m2(v671, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u32m2
      vuint32m2_t v676 = __riscv_vsll_vx_u32m2(v675, 23, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsltu_vx_u32m2_b16
      vbool16_t v677 = __riscv_vmsltu_vx_u32m2_b16(v671, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_u32m2
      vuint32m2_t v678 = __riscv_vmerge_vvm_u32m2(v676, v674, v677, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_reinterpret_f32
      vfloat32m2_t v679 = __riscv_vreinterpret_v_u32m2_f32m2(v678);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v680 = __riscv_vfmul_vf_f32m2(v679, v668, 8);
      vint32m2_t v681 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v682 = __riscv_vfcvt_f_x_v_f32m2(v681, 8);
      vfloat32m2_t v683 = v13;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v684 = __riscv_vfmacc_vv_f32m2(v683, v682, v680, 8);
      v13 = v684;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_e8m0_scale_addr
      const uint8_t* v685 = v19 + 8;
      const uint8_t* v686 = (const uint8_t*) v685;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_exponent_load
      vuint8mf2_t v687 = __riscv_vle8_v_u8mf2(v686, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_widen_u32
      vuint32m2_t v688 = __riscv_vzext_vf4_u32m2(v687, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_to_fp32_half_bits
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u32m2
      vuint32m2_t v689 = __riscv_vand_vx_u32m2(v688, 0x1F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u32m2
      vuint32m2_t v690 = __riscv_vmv_v_x_u32m2(0x00200000, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vv_u32m2
      vuint32m2_t v691 = __riscv_vsll_vv_u32m2(v690, v689, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_u32m2
      vuint32m2_t v692 = __riscv_vsub_vx_u32m2(v688, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u32m2
      vuint32m2_t v693 = __riscv_vsll_vx_u32m2(v692, 23, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsltu_vx_u32m2_b16
      vbool16_t v694 = __riscv_vmsltu_vx_u32m2_b16(v688, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_u32m2
      vuint32m2_t v695 = __riscv_vmerge_vvm_u32m2(v693, v691, v694, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_reinterpret_f32
      vfloat32m2_t v696 = __riscv_vreinterpret_v_u32m2_f32m2(v695);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v697 = __riscv_vfmul_vf_f32m2(v696, v668, 8);
      vint32m2_t v698 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v699 = __riscv_vfcvt_f_x_v_f32m2(v698, 8);
      vfloat32m2_t v700 = v15;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v701 = __riscv_vfmacc_vv_f32m2(v700, v699, v697, 8);
      v15 = v701;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v702 = v9 * 16;
    float* v703 = v2 + v702;
    vfloat32m2_t v704 = v13;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v703, v704, 8);
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v705 = v9 * 16;
    size_t v706 = v705 + 8;
    float* v707 = v2 + v706;
    vfloat32m2_t v708 = v15;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v707, v708, 8);
  }
  return;
}


