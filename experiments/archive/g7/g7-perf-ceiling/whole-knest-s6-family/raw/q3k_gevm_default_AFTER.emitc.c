#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemv_q3_K_q8_K_kernel_ggml_repack_gemv_q3_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=col_group_count
  size_t v8 = v5 / 16;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_group_base
    size_t v10 = v9 * v7;
    size_t v11 = v10 * 1824;
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
      size_t v18 = v17 * 1824;
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
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
      const uint8_t* v28 = v19 + 32;
      const int8_t* v29 = (const int8_t*) v28;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v30 = __riscv_vle8_v_i8mf2(v29, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v31 = __riscv_vsext_vf2_i16m1(v30, 8);
      const uint8_t* v32 = v19 + 64;
      const int8_t* v33 = (const int8_t*) v32;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v34 = __riscv_vle8_v_i8mf2(v33, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v35 = __riscv_vsext_vf2_i16m1(v34, 8);
      const uint8_t* v36 = v19 + 96;
      const int8_t* v37 = (const int8_t*) v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v38 = __riscv_vle8_v_i8mf2(v37, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v39 = __riscv_vsext_vf2_i16m1(v38, 8);
      const uint8_t* v40 = v19 + 128;
      const int8_t* v41 = (const int8_t*) v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v42 = __riscv_vle8_v_i8mf2(v41, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v43 = __riscv_vsext_vf2_i16m1(v42, 8);
      const uint8_t* v44 = v19 + 40;
      const int8_t* v45 = (const int8_t*) v44;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v46 = __riscv_vle8_v_i8mf2(v45, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v47 = __riscv_vsext_vf2_i16m1(v46, 8);
      const uint8_t* v48 = v19 + 72;
      const int8_t* v49 = (const int8_t*) v48;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v50 = __riscv_vle8_v_i8mf2(v49, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v51 = __riscv_vsext_vf2_i16m1(v50, 8);
      const uint8_t* v52 = v19 + 104;
      const int8_t* v53 = (const int8_t*) v52;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v54 = __riscv_vle8_v_i8mf2(v53, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v55 = __riscv_vsext_vf2_i16m1(v54, 8);
      const uint8_t* v56 = v19 + 136;
      const int8_t* v57 = (const int8_t*) v56;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v58 = __riscv_vle8_v_i8mf2(v57, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v59 = __riscv_vsext_vf2_i16m1(v58, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v60 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v61 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v62 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v63 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v64 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v65 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v66 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v67 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v68 = v21 + 4;
      const int8_t* v69 = (const int8_t*) v68;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v70 = *(const int8_t *)(v69);
      const uint8_t* v71 = v21 + 36;
      const int8_t* v72 = (const int8_t*) v71;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v73 = *(const int8_t *)(v72);
      const uint8_t* v74 = v21 + 68;
      const int8_t* v75 = (const int8_t*) v74;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v76 = *(const int8_t *)(v75);
      const uint8_t* v77 = v21 + 100;
      const int8_t* v78 = (const int8_t*) v77;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v79 = *(const int8_t *)(v78);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v80 = v19 + 800;
      const uint8_t* v81 = (const uint8_t*) v80;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v82 = __riscv_vle8_v_u8mf2(v81, 8);
      const uint8_t* v83 = v19 + 288;
      const uint8_t* v84 = (const uint8_t*) v83;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v85 = __riscv_vle8_v_u8mf2(v84, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v86 = __riscv_vand_vx_u8mf2(v82, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v87 = __riscv_vreinterpret_v_u8mf2_i8mf2(v86);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v88 = __riscv_vand_vx_u8mf2(v85, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v89 = __riscv_vmseq_vx_u8mf2_b16(v88, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v90 = __riscv_vadd_vx_i8mf2_mu(v89, v87, v87, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v91 = __riscv_vwmacc_vx_i16m1(v60, v70, v90, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v92 = __riscv_vsrl_vx_u8mf2(v82, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v93 = __riscv_vand_vx_u8mf2(v92, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v94 = __riscv_vreinterpret_v_u8mf2_i8mf2(v93);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v95 = __riscv_vand_vx_u8mf2(v85, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v96 = __riscv_vmseq_vx_u8mf2_b16(v95, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v97 = __riscv_vadd_vx_i8mf2_mu(v96, v94, v94, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v98 = __riscv_vwmacc_vx_i16m1(v61, v73, v97, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v99 = __riscv_vsrl_vx_u8mf2(v82, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v100 = __riscv_vand_vx_u8mf2(v99, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v101 = __riscv_vreinterpret_v_u8mf2_i8mf2(v100);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v102 = __riscv_vand_vx_u8mf2(v85, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v103 = __riscv_vmseq_vx_u8mf2_b16(v102, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v104 = __riscv_vadd_vx_i8mf2_mu(v103, v101, v101, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v105 = __riscv_vwmacc_vx_i16m1(v62, v76, v104, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v106 = __riscv_vsrl_vx_u8mf2(v82, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v107 = __riscv_vand_vx_u8mf2(v106, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v108 = __riscv_vreinterpret_v_u8mf2_i8mf2(v107);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v109 = __riscv_vand_vx_u8mf2(v85, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v110 = __riscv_vmseq_vx_u8mf2_b16(v109, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v111 = __riscv_vadd_vx_i8mf2_mu(v110, v108, v108, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v112 = __riscv_vwmacc_vx_i16m1(v63, v79, v111, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v113 = v19 + 808;
      const uint8_t* v114 = (const uint8_t*) v113;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v115 = __riscv_vle8_v_u8mf2(v114, 8);
      const uint8_t* v116 = v19 + 296;
      const uint8_t* v117 = (const uint8_t*) v116;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v118 = __riscv_vle8_v_u8mf2(v117, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v119 = __riscv_vand_vx_u8mf2(v115, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v120 = __riscv_vreinterpret_v_u8mf2_i8mf2(v119);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v121 = __riscv_vand_vx_u8mf2(v118, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v122 = __riscv_vmseq_vx_u8mf2_b16(v121, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v123 = __riscv_vadd_vx_i8mf2_mu(v122, v120, v120, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v124 = __riscv_vwmacc_vx_i16m1(v64, v70, v123, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v125 = __riscv_vsrl_vx_u8mf2(v115, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v126 = __riscv_vand_vx_u8mf2(v125, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v127 = __riscv_vreinterpret_v_u8mf2_i8mf2(v126);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v128 = __riscv_vand_vx_u8mf2(v118, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v129 = __riscv_vmseq_vx_u8mf2_b16(v128, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v130 = __riscv_vadd_vx_i8mf2_mu(v129, v127, v127, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v131 = __riscv_vwmacc_vx_i16m1(v65, v73, v130, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v132 = __riscv_vsrl_vx_u8mf2(v115, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v133 = __riscv_vand_vx_u8mf2(v132, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v134 = __riscv_vreinterpret_v_u8mf2_i8mf2(v133);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v135 = __riscv_vand_vx_u8mf2(v118, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v136 = __riscv_vmseq_vx_u8mf2_b16(v135, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v137 = __riscv_vadd_vx_i8mf2_mu(v136, v134, v134, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v138 = __riscv_vwmacc_vx_i16m1(v66, v76, v137, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v139 = __riscv_vsrl_vx_u8mf2(v115, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v140 = __riscv_vand_vx_u8mf2(v139, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v141 = __riscv_vreinterpret_v_u8mf2_i8mf2(v140);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v142 = __riscv_vand_vx_u8mf2(v118, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v143 = __riscv_vmseq_vx_u8mf2_b16(v142, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v144 = __riscv_vadd_vx_i8mf2_mu(v143, v141, v141, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v145 = __riscv_vwmacc_vx_i16m1(v67, v79, v144, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v146 = v21 + 5;
      const int8_t* v147 = (const int8_t*) v146;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v148 = *(const int8_t *)(v147);
      const uint8_t* v149 = v21 + 37;
      const int8_t* v150 = (const int8_t*) v149;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v151 = *(const int8_t *)(v150);
      const uint8_t* v152 = v21 + 69;
      const int8_t* v153 = (const int8_t*) v152;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v154 = *(const int8_t *)(v153);
      const uint8_t* v155 = v21 + 101;
      const int8_t* v156 = (const int8_t*) v155;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v157 = *(const int8_t *)(v156);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v158 = v19 + 816;
      const uint8_t* v159 = (const uint8_t*) v158;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v160 = __riscv_vle8_v_u8mf2(v159, 8);
      const uint8_t* v161 = v19 + 304;
      const uint8_t* v162 = (const uint8_t*) v161;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v163 = __riscv_vle8_v_u8mf2(v162, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v164 = __riscv_vand_vx_u8mf2(v160, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v165 = __riscv_vreinterpret_v_u8mf2_i8mf2(v164);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v166 = __riscv_vand_vx_u8mf2(v163, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v167 = __riscv_vmseq_vx_u8mf2_b16(v166, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v168 = __riscv_vadd_vx_i8mf2_mu(v167, v165, v165, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v169 = __riscv_vwmacc_vx_i16m1(v91, v148, v168, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v170 = __riscv_vsrl_vx_u8mf2(v160, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v171 = __riscv_vand_vx_u8mf2(v170, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v172 = __riscv_vreinterpret_v_u8mf2_i8mf2(v171);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v173 = __riscv_vand_vx_u8mf2(v163, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v174 = __riscv_vmseq_vx_u8mf2_b16(v173, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v175 = __riscv_vadd_vx_i8mf2_mu(v174, v172, v172, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v176 = __riscv_vwmacc_vx_i16m1(v98, v151, v175, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v177 = __riscv_vsrl_vx_u8mf2(v160, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v178 = __riscv_vand_vx_u8mf2(v177, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v179 = __riscv_vreinterpret_v_u8mf2_i8mf2(v178);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v180 = __riscv_vand_vx_u8mf2(v163, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v181 = __riscv_vmseq_vx_u8mf2_b16(v180, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v182 = __riscv_vadd_vx_i8mf2_mu(v181, v179, v179, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v183 = __riscv_vwmacc_vx_i16m1(v105, v154, v182, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v184 = __riscv_vsrl_vx_u8mf2(v160, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v185 = __riscv_vand_vx_u8mf2(v184, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v186 = __riscv_vreinterpret_v_u8mf2_i8mf2(v185);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v187 = __riscv_vand_vx_u8mf2(v163, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v188 = __riscv_vmseq_vx_u8mf2_b16(v187, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v189 = __riscv_vadd_vx_i8mf2_mu(v188, v186, v186, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v190 = __riscv_vwmacc_vx_i16m1(v112, v157, v189, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v191 = v19 + 824;
      const uint8_t* v192 = (const uint8_t*) v191;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v193 = __riscv_vle8_v_u8mf2(v192, 8);
      const uint8_t* v194 = v19 + 312;
      const uint8_t* v195 = (const uint8_t*) v194;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v196 = __riscv_vle8_v_u8mf2(v195, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v197 = __riscv_vand_vx_u8mf2(v193, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v198 = __riscv_vreinterpret_v_u8mf2_i8mf2(v197);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v199 = __riscv_vand_vx_u8mf2(v196, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v200 = __riscv_vmseq_vx_u8mf2_b16(v199, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v201 = __riscv_vadd_vx_i8mf2_mu(v200, v198, v198, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v202 = __riscv_vwmacc_vx_i16m1(v124, v148, v201, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v203 = __riscv_vsrl_vx_u8mf2(v193, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v204 = __riscv_vand_vx_u8mf2(v203, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v205 = __riscv_vreinterpret_v_u8mf2_i8mf2(v204);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v206 = __riscv_vand_vx_u8mf2(v196, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v207 = __riscv_vmseq_vx_u8mf2_b16(v206, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v208 = __riscv_vadd_vx_i8mf2_mu(v207, v205, v205, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v209 = __riscv_vwmacc_vx_i16m1(v131, v151, v208, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v210 = __riscv_vsrl_vx_u8mf2(v193, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v211 = __riscv_vand_vx_u8mf2(v210, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v212 = __riscv_vreinterpret_v_u8mf2_i8mf2(v211);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v213 = __riscv_vand_vx_u8mf2(v196, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v214 = __riscv_vmseq_vx_u8mf2_b16(v213, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v215 = __riscv_vadd_vx_i8mf2_mu(v214, v212, v212, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v216 = __riscv_vwmacc_vx_i16m1(v138, v154, v215, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v217 = __riscv_vsrl_vx_u8mf2(v193, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v218 = __riscv_vand_vx_u8mf2(v217, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v219 = __riscv_vreinterpret_v_u8mf2_i8mf2(v218);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v220 = __riscv_vand_vx_u8mf2(v196, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v221 = __riscv_vmseq_vx_u8mf2_b16(v220, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v222 = __riscv_vadd_vx_i8mf2_mu(v221, v219, v219, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v223 = __riscv_vwmacc_vx_i16m1(v145, v157, v222, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v224 = v21 + 6;
      const int8_t* v225 = (const int8_t*) v224;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v226 = *(const int8_t *)(v225);
      const uint8_t* v227 = v21 + 38;
      const int8_t* v228 = (const int8_t*) v227;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v229 = *(const int8_t *)(v228);
      const uint8_t* v230 = v21 + 70;
      const int8_t* v231 = (const int8_t*) v230;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v232 = *(const int8_t *)(v231);
      const uint8_t* v233 = v21 + 102;
      const int8_t* v234 = (const int8_t*) v233;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v235 = *(const int8_t *)(v234);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v236 = v19 + 832;
      const uint8_t* v237 = (const uint8_t*) v236;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v238 = __riscv_vle8_v_u8mf2(v237, 8);
      const uint8_t* v239 = v19 + 320;
      const uint8_t* v240 = (const uint8_t*) v239;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v241 = __riscv_vle8_v_u8mf2(v240, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v242 = __riscv_vand_vx_u8mf2(v238, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v243 = __riscv_vreinterpret_v_u8mf2_i8mf2(v242);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v244 = __riscv_vand_vx_u8mf2(v241, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v245 = __riscv_vmseq_vx_u8mf2_b16(v244, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v246 = __riscv_vadd_vx_i8mf2_mu(v245, v243, v243, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v247 = __riscv_vwmacc_vx_i16m1(v169, v226, v246, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v248 = __riscv_vsrl_vx_u8mf2(v238, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v249 = __riscv_vand_vx_u8mf2(v248, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v250 = __riscv_vreinterpret_v_u8mf2_i8mf2(v249);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v251 = __riscv_vand_vx_u8mf2(v241, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v252 = __riscv_vmseq_vx_u8mf2_b16(v251, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v253 = __riscv_vadd_vx_i8mf2_mu(v252, v250, v250, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v254 = __riscv_vwmacc_vx_i16m1(v176, v229, v253, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v255 = __riscv_vsrl_vx_u8mf2(v238, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v256 = __riscv_vand_vx_u8mf2(v255, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v257 = __riscv_vreinterpret_v_u8mf2_i8mf2(v256);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v258 = __riscv_vand_vx_u8mf2(v241, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v259 = __riscv_vmseq_vx_u8mf2_b16(v258, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v260 = __riscv_vadd_vx_i8mf2_mu(v259, v257, v257, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v261 = __riscv_vwmacc_vx_i16m1(v183, v232, v260, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v262 = __riscv_vsrl_vx_u8mf2(v238, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v263 = __riscv_vand_vx_u8mf2(v262, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v264 = __riscv_vreinterpret_v_u8mf2_i8mf2(v263);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v265 = __riscv_vand_vx_u8mf2(v241, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v266 = __riscv_vmseq_vx_u8mf2_b16(v265, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v267 = __riscv_vadd_vx_i8mf2_mu(v266, v264, v264, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v268 = __riscv_vwmacc_vx_i16m1(v190, v235, v267, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v269 = v19 + 840;
      const uint8_t* v270 = (const uint8_t*) v269;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v271 = __riscv_vle8_v_u8mf2(v270, 8);
      const uint8_t* v272 = v19 + 328;
      const uint8_t* v273 = (const uint8_t*) v272;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v274 = __riscv_vle8_v_u8mf2(v273, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v275 = __riscv_vand_vx_u8mf2(v271, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v276 = __riscv_vreinterpret_v_u8mf2_i8mf2(v275);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v277 = __riscv_vand_vx_u8mf2(v274, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v278 = __riscv_vmseq_vx_u8mf2_b16(v277, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v279 = __riscv_vadd_vx_i8mf2_mu(v278, v276, v276, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v280 = __riscv_vwmacc_vx_i16m1(v202, v226, v279, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v281 = __riscv_vsrl_vx_u8mf2(v271, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v282 = __riscv_vand_vx_u8mf2(v281, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v283 = __riscv_vreinterpret_v_u8mf2_i8mf2(v282);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v284 = __riscv_vand_vx_u8mf2(v274, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v285 = __riscv_vmseq_vx_u8mf2_b16(v284, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v286 = __riscv_vadd_vx_i8mf2_mu(v285, v283, v283, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v287 = __riscv_vwmacc_vx_i16m1(v209, v229, v286, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v288 = __riscv_vsrl_vx_u8mf2(v271, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v289 = __riscv_vand_vx_u8mf2(v288, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v290 = __riscv_vreinterpret_v_u8mf2_i8mf2(v289);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v291 = __riscv_vand_vx_u8mf2(v274, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v292 = __riscv_vmseq_vx_u8mf2_b16(v291, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v293 = __riscv_vadd_vx_i8mf2_mu(v292, v290, v290, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v294 = __riscv_vwmacc_vx_i16m1(v216, v232, v293, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v295 = __riscv_vsrl_vx_u8mf2(v271, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v296 = __riscv_vand_vx_u8mf2(v295, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v297 = __riscv_vreinterpret_v_u8mf2_i8mf2(v296);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v298 = __riscv_vand_vx_u8mf2(v274, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v299 = __riscv_vmseq_vx_u8mf2_b16(v298, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v300 = __riscv_vadd_vx_i8mf2_mu(v299, v297, v297, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v301 = __riscv_vwmacc_vx_i16m1(v223, v235, v300, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v302 = v21 + 7;
      const int8_t* v303 = (const int8_t*) v302;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v304 = *(const int8_t *)(v303);
      const uint8_t* v305 = v21 + 39;
      const int8_t* v306 = (const int8_t*) v305;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v307 = *(const int8_t *)(v306);
      const uint8_t* v308 = v21 + 71;
      const int8_t* v309 = (const int8_t*) v308;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v310 = *(const int8_t *)(v309);
      const uint8_t* v311 = v21 + 103;
      const int8_t* v312 = (const int8_t*) v311;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v313 = *(const int8_t *)(v312);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v314 = v19 + 848;
      const uint8_t* v315 = (const uint8_t*) v314;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v316 = __riscv_vle8_v_u8mf2(v315, 8);
      const uint8_t* v317 = v19 + 336;
      const uint8_t* v318 = (const uint8_t*) v317;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v319 = __riscv_vle8_v_u8mf2(v318, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v320 = __riscv_vand_vx_u8mf2(v316, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v321 = __riscv_vreinterpret_v_u8mf2_i8mf2(v320);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v322 = __riscv_vand_vx_u8mf2(v319, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v323 = __riscv_vmseq_vx_u8mf2_b16(v322, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v324 = __riscv_vadd_vx_i8mf2_mu(v323, v321, v321, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v325 = __riscv_vwmacc_vx_i16m1(v247, v304, v324, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v326 = __riscv_vsrl_vx_u8mf2(v316, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v327 = __riscv_vand_vx_u8mf2(v326, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v328 = __riscv_vreinterpret_v_u8mf2_i8mf2(v327);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v329 = __riscv_vand_vx_u8mf2(v319, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v330 = __riscv_vmseq_vx_u8mf2_b16(v329, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v331 = __riscv_vadd_vx_i8mf2_mu(v330, v328, v328, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v332 = __riscv_vwmacc_vx_i16m1(v254, v307, v331, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v333 = __riscv_vsrl_vx_u8mf2(v316, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v334 = __riscv_vand_vx_u8mf2(v333, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v335 = __riscv_vreinterpret_v_u8mf2_i8mf2(v334);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v336 = __riscv_vand_vx_u8mf2(v319, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v337 = __riscv_vmseq_vx_u8mf2_b16(v336, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v338 = __riscv_vadd_vx_i8mf2_mu(v337, v335, v335, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v339 = __riscv_vwmacc_vx_i16m1(v261, v310, v338, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v340 = __riscv_vsrl_vx_u8mf2(v316, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v341 = __riscv_vand_vx_u8mf2(v340, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v342 = __riscv_vreinterpret_v_u8mf2_i8mf2(v341);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v343 = __riscv_vand_vx_u8mf2(v319, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v344 = __riscv_vmseq_vx_u8mf2_b16(v343, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v345 = __riscv_vadd_vx_i8mf2_mu(v344, v342, v342, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v346 = __riscv_vwmacc_vx_i16m1(v268, v313, v345, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v347 = v19 + 856;
      const uint8_t* v348 = (const uint8_t*) v347;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v349 = __riscv_vle8_v_u8mf2(v348, 8);
      const uint8_t* v350 = v19 + 344;
      const uint8_t* v351 = (const uint8_t*) v350;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v352 = __riscv_vle8_v_u8mf2(v351, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v353 = __riscv_vand_vx_u8mf2(v349, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v354 = __riscv_vreinterpret_v_u8mf2_i8mf2(v353);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v355 = __riscv_vand_vx_u8mf2(v352, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v356 = __riscv_vmseq_vx_u8mf2_b16(v355, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v357 = __riscv_vadd_vx_i8mf2_mu(v356, v354, v354, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v358 = __riscv_vwmacc_vx_i16m1(v280, v304, v357, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v359 = __riscv_vsrl_vx_u8mf2(v349, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v360 = __riscv_vand_vx_u8mf2(v359, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v361 = __riscv_vreinterpret_v_u8mf2_i8mf2(v360);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v362 = __riscv_vand_vx_u8mf2(v352, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v363 = __riscv_vmseq_vx_u8mf2_b16(v362, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v364 = __riscv_vadd_vx_i8mf2_mu(v363, v361, v361, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v365 = __riscv_vwmacc_vx_i16m1(v287, v307, v364, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v366 = __riscv_vsrl_vx_u8mf2(v349, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v367 = __riscv_vand_vx_u8mf2(v366, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v368 = __riscv_vreinterpret_v_u8mf2_i8mf2(v367);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v369 = __riscv_vand_vx_u8mf2(v352, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v370 = __riscv_vmseq_vx_u8mf2_b16(v369, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v371 = __riscv_vadd_vx_i8mf2_mu(v370, v368, v368, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v372 = __riscv_vwmacc_vx_i16m1(v294, v310, v371, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v373 = __riscv_vsrl_vx_u8mf2(v349, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v374 = __riscv_vand_vx_u8mf2(v373, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v375 = __riscv_vreinterpret_v_u8mf2_i8mf2(v374);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v376 = __riscv_vand_vx_u8mf2(v352, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v377 = __riscv_vmseq_vx_u8mf2_b16(v376, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v378 = __riscv_vadd_vx_i8mf2_mu(v377, v375, v375, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v379 = __riscv_vwmacc_vx_i16m1(v301, v313, v378, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v380 = v21 + 8;
      const int8_t* v381 = (const int8_t*) v380;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v382 = *(const int8_t *)(v381);
      const uint8_t* v383 = v21 + 40;
      const int8_t* v384 = (const int8_t*) v383;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v385 = *(const int8_t *)(v384);
      const uint8_t* v386 = v21 + 72;
      const int8_t* v387 = (const int8_t*) v386;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v388 = *(const int8_t *)(v387);
      const uint8_t* v389 = v21 + 104;
      const int8_t* v390 = (const int8_t*) v389;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v391 = *(const int8_t *)(v390);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v392 = v19 + 864;
      const uint8_t* v393 = (const uint8_t*) v392;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v394 = __riscv_vle8_v_u8mf2(v393, 8);
      const uint8_t* v395 = v19 + 352;
      const uint8_t* v396 = (const uint8_t*) v395;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v397 = __riscv_vle8_v_u8mf2(v396, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v398 = __riscv_vand_vx_u8mf2(v394, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v399 = __riscv_vreinterpret_v_u8mf2_i8mf2(v398);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v400 = __riscv_vand_vx_u8mf2(v397, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v401 = __riscv_vmseq_vx_u8mf2_b16(v400, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v402 = __riscv_vadd_vx_i8mf2_mu(v401, v399, v399, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v403 = __riscv_vwmacc_vx_i16m1(v325, v382, v402, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v404 = __riscv_vsrl_vx_u8mf2(v394, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v405 = __riscv_vand_vx_u8mf2(v404, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v406 = __riscv_vreinterpret_v_u8mf2_i8mf2(v405);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v407 = __riscv_vand_vx_u8mf2(v397, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v408 = __riscv_vmseq_vx_u8mf2_b16(v407, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v409 = __riscv_vadd_vx_i8mf2_mu(v408, v406, v406, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v410 = __riscv_vwmacc_vx_i16m1(v332, v385, v409, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v411 = __riscv_vsrl_vx_u8mf2(v394, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v412 = __riscv_vand_vx_u8mf2(v411, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v413 = __riscv_vreinterpret_v_u8mf2_i8mf2(v412);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v414 = __riscv_vand_vx_u8mf2(v397, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v415 = __riscv_vmseq_vx_u8mf2_b16(v414, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v416 = __riscv_vadd_vx_i8mf2_mu(v415, v413, v413, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v417 = __riscv_vwmacc_vx_i16m1(v339, v388, v416, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v418 = __riscv_vsrl_vx_u8mf2(v394, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v419 = __riscv_vand_vx_u8mf2(v418, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v420 = __riscv_vreinterpret_v_u8mf2_i8mf2(v419);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v421 = __riscv_vand_vx_u8mf2(v397, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v422 = __riscv_vmseq_vx_u8mf2_b16(v421, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v423 = __riscv_vadd_vx_i8mf2_mu(v422, v420, v420, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v424 = __riscv_vwmacc_vx_i16m1(v346, v391, v423, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v425 = v19 + 872;
      const uint8_t* v426 = (const uint8_t*) v425;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v427 = __riscv_vle8_v_u8mf2(v426, 8);
      const uint8_t* v428 = v19 + 360;
      const uint8_t* v429 = (const uint8_t*) v428;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v430 = __riscv_vle8_v_u8mf2(v429, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v431 = __riscv_vand_vx_u8mf2(v427, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v432 = __riscv_vreinterpret_v_u8mf2_i8mf2(v431);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v433 = __riscv_vand_vx_u8mf2(v430, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v434 = __riscv_vmseq_vx_u8mf2_b16(v433, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v435 = __riscv_vadd_vx_i8mf2_mu(v434, v432, v432, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v436 = __riscv_vwmacc_vx_i16m1(v358, v382, v435, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v437 = __riscv_vsrl_vx_u8mf2(v427, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v438 = __riscv_vand_vx_u8mf2(v437, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v439 = __riscv_vreinterpret_v_u8mf2_i8mf2(v438);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v440 = __riscv_vand_vx_u8mf2(v430, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v441 = __riscv_vmseq_vx_u8mf2_b16(v440, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v442 = __riscv_vadd_vx_i8mf2_mu(v441, v439, v439, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v443 = __riscv_vwmacc_vx_i16m1(v365, v385, v442, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v444 = __riscv_vsrl_vx_u8mf2(v427, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v445 = __riscv_vand_vx_u8mf2(v444, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v446 = __riscv_vreinterpret_v_u8mf2_i8mf2(v445);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v447 = __riscv_vand_vx_u8mf2(v430, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v448 = __riscv_vmseq_vx_u8mf2_b16(v447, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v449 = __riscv_vadd_vx_i8mf2_mu(v448, v446, v446, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v450 = __riscv_vwmacc_vx_i16m1(v372, v388, v449, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v451 = __riscv_vsrl_vx_u8mf2(v427, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v452 = __riscv_vand_vx_u8mf2(v451, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v453 = __riscv_vreinterpret_v_u8mf2_i8mf2(v452);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v454 = __riscv_vand_vx_u8mf2(v430, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v455 = __riscv_vmseq_vx_u8mf2_b16(v454, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v456 = __riscv_vadd_vx_i8mf2_mu(v455, v453, v453, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v457 = __riscv_vwmacc_vx_i16m1(v379, v391, v456, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v458 = v21 + 9;
      const int8_t* v459 = (const int8_t*) v458;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v460 = *(const int8_t *)(v459);
      const uint8_t* v461 = v21 + 41;
      const int8_t* v462 = (const int8_t*) v461;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v463 = *(const int8_t *)(v462);
      const uint8_t* v464 = v21 + 73;
      const int8_t* v465 = (const int8_t*) v464;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v466 = *(const int8_t *)(v465);
      const uint8_t* v467 = v21 + 105;
      const int8_t* v468 = (const int8_t*) v467;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v469 = *(const int8_t *)(v468);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v470 = v19 + 880;
      const uint8_t* v471 = (const uint8_t*) v470;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v472 = __riscv_vle8_v_u8mf2(v471, 8);
      const uint8_t* v473 = v19 + 368;
      const uint8_t* v474 = (const uint8_t*) v473;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v475 = __riscv_vle8_v_u8mf2(v474, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v476 = __riscv_vand_vx_u8mf2(v472, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v477 = __riscv_vreinterpret_v_u8mf2_i8mf2(v476);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v478 = __riscv_vand_vx_u8mf2(v475, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v479 = __riscv_vmseq_vx_u8mf2_b16(v478, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v480 = __riscv_vadd_vx_i8mf2_mu(v479, v477, v477, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v481 = __riscv_vwmacc_vx_i16m1(v403, v460, v480, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v482 = __riscv_vsrl_vx_u8mf2(v472, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v483 = __riscv_vand_vx_u8mf2(v482, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v484 = __riscv_vreinterpret_v_u8mf2_i8mf2(v483);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v485 = __riscv_vand_vx_u8mf2(v475, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v486 = __riscv_vmseq_vx_u8mf2_b16(v485, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v487 = __riscv_vadd_vx_i8mf2_mu(v486, v484, v484, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v488 = __riscv_vwmacc_vx_i16m1(v410, v463, v487, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v489 = __riscv_vsrl_vx_u8mf2(v472, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v490 = __riscv_vand_vx_u8mf2(v489, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v491 = __riscv_vreinterpret_v_u8mf2_i8mf2(v490);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v492 = __riscv_vand_vx_u8mf2(v475, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v493 = __riscv_vmseq_vx_u8mf2_b16(v492, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v494 = __riscv_vadd_vx_i8mf2_mu(v493, v491, v491, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v495 = __riscv_vwmacc_vx_i16m1(v417, v466, v494, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v496 = __riscv_vsrl_vx_u8mf2(v472, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v497 = __riscv_vand_vx_u8mf2(v496, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v498 = __riscv_vreinterpret_v_u8mf2_i8mf2(v497);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v499 = __riscv_vand_vx_u8mf2(v475, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v500 = __riscv_vmseq_vx_u8mf2_b16(v499, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v501 = __riscv_vadd_vx_i8mf2_mu(v500, v498, v498, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v502 = __riscv_vwmacc_vx_i16m1(v424, v469, v501, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v503 = v19 + 888;
      const uint8_t* v504 = (const uint8_t*) v503;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v505 = __riscv_vle8_v_u8mf2(v504, 8);
      const uint8_t* v506 = v19 + 376;
      const uint8_t* v507 = (const uint8_t*) v506;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v508 = __riscv_vle8_v_u8mf2(v507, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v509 = __riscv_vand_vx_u8mf2(v505, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v510 = __riscv_vreinterpret_v_u8mf2_i8mf2(v509);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v511 = __riscv_vand_vx_u8mf2(v508, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v512 = __riscv_vmseq_vx_u8mf2_b16(v511, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v513 = __riscv_vadd_vx_i8mf2_mu(v512, v510, v510, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v514 = __riscv_vwmacc_vx_i16m1(v436, v460, v513, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v515 = __riscv_vsrl_vx_u8mf2(v505, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v516 = __riscv_vand_vx_u8mf2(v515, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v517 = __riscv_vreinterpret_v_u8mf2_i8mf2(v516);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v518 = __riscv_vand_vx_u8mf2(v508, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v519 = __riscv_vmseq_vx_u8mf2_b16(v518, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v520 = __riscv_vadd_vx_i8mf2_mu(v519, v517, v517, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v521 = __riscv_vwmacc_vx_i16m1(v443, v463, v520, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v522 = __riscv_vsrl_vx_u8mf2(v505, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v523 = __riscv_vand_vx_u8mf2(v522, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v524 = __riscv_vreinterpret_v_u8mf2_i8mf2(v523);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v525 = __riscv_vand_vx_u8mf2(v508, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v526 = __riscv_vmseq_vx_u8mf2_b16(v525, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v527 = __riscv_vadd_vx_i8mf2_mu(v526, v524, v524, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v528 = __riscv_vwmacc_vx_i16m1(v450, v466, v527, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v529 = __riscv_vsrl_vx_u8mf2(v505, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v530 = __riscv_vand_vx_u8mf2(v529, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v531 = __riscv_vreinterpret_v_u8mf2_i8mf2(v530);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v532 = __riscv_vand_vx_u8mf2(v508, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v533 = __riscv_vmseq_vx_u8mf2_b16(v532, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v534 = __riscv_vadd_vx_i8mf2_mu(v533, v531, v531, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v535 = __riscv_vwmacc_vx_i16m1(v457, v469, v534, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v536 = v21 + 10;
      const int8_t* v537 = (const int8_t*) v536;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v538 = *(const int8_t *)(v537);
      const uint8_t* v539 = v21 + 42;
      const int8_t* v540 = (const int8_t*) v539;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v541 = *(const int8_t *)(v540);
      const uint8_t* v542 = v21 + 74;
      const int8_t* v543 = (const int8_t*) v542;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v544 = *(const int8_t *)(v543);
      const uint8_t* v545 = v21 + 106;
      const int8_t* v546 = (const int8_t*) v545;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v547 = *(const int8_t *)(v546);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v548 = v19 + 896;
      const uint8_t* v549 = (const uint8_t*) v548;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v550 = __riscv_vle8_v_u8mf2(v549, 8);
      const uint8_t* v551 = v19 + 384;
      const uint8_t* v552 = (const uint8_t*) v551;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v553 = __riscv_vle8_v_u8mf2(v552, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v554 = __riscv_vand_vx_u8mf2(v550, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v555 = __riscv_vreinterpret_v_u8mf2_i8mf2(v554);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v556 = __riscv_vand_vx_u8mf2(v553, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v557 = __riscv_vmseq_vx_u8mf2_b16(v556, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v558 = __riscv_vadd_vx_i8mf2_mu(v557, v555, v555, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v559 = __riscv_vwmacc_vx_i16m1(v481, v538, v558, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v560 = __riscv_vsrl_vx_u8mf2(v550, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v561 = __riscv_vand_vx_u8mf2(v560, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v562 = __riscv_vreinterpret_v_u8mf2_i8mf2(v561);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v563 = __riscv_vand_vx_u8mf2(v553, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v564 = __riscv_vmseq_vx_u8mf2_b16(v563, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v565 = __riscv_vadd_vx_i8mf2_mu(v564, v562, v562, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v566 = __riscv_vwmacc_vx_i16m1(v488, v541, v565, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v567 = __riscv_vsrl_vx_u8mf2(v550, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v568 = __riscv_vand_vx_u8mf2(v567, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v569 = __riscv_vreinterpret_v_u8mf2_i8mf2(v568);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v570 = __riscv_vand_vx_u8mf2(v553, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v571 = __riscv_vmseq_vx_u8mf2_b16(v570, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v572 = __riscv_vadd_vx_i8mf2_mu(v571, v569, v569, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v573 = __riscv_vwmacc_vx_i16m1(v495, v544, v572, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v574 = __riscv_vsrl_vx_u8mf2(v550, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v575 = __riscv_vand_vx_u8mf2(v574, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v576 = __riscv_vreinterpret_v_u8mf2_i8mf2(v575);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v577 = __riscv_vand_vx_u8mf2(v553, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v578 = __riscv_vmseq_vx_u8mf2_b16(v577, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v579 = __riscv_vadd_vx_i8mf2_mu(v578, v576, v576, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v580 = __riscv_vwmacc_vx_i16m1(v502, v547, v579, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v581 = v19 + 904;
      const uint8_t* v582 = (const uint8_t*) v581;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v583 = __riscv_vle8_v_u8mf2(v582, 8);
      const uint8_t* v584 = v19 + 392;
      const uint8_t* v585 = (const uint8_t*) v584;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v586 = __riscv_vle8_v_u8mf2(v585, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v587 = __riscv_vand_vx_u8mf2(v583, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v588 = __riscv_vreinterpret_v_u8mf2_i8mf2(v587);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v589 = __riscv_vand_vx_u8mf2(v586, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v590 = __riscv_vmseq_vx_u8mf2_b16(v589, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v591 = __riscv_vadd_vx_i8mf2_mu(v590, v588, v588, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v592 = __riscv_vwmacc_vx_i16m1(v514, v538, v591, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v593 = __riscv_vsrl_vx_u8mf2(v583, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v594 = __riscv_vand_vx_u8mf2(v593, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v595 = __riscv_vreinterpret_v_u8mf2_i8mf2(v594);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v596 = __riscv_vand_vx_u8mf2(v586, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v597 = __riscv_vmseq_vx_u8mf2_b16(v596, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v598 = __riscv_vadd_vx_i8mf2_mu(v597, v595, v595, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v599 = __riscv_vwmacc_vx_i16m1(v521, v541, v598, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v600 = __riscv_vsrl_vx_u8mf2(v583, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v601 = __riscv_vand_vx_u8mf2(v600, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v602 = __riscv_vreinterpret_v_u8mf2_i8mf2(v601);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v603 = __riscv_vand_vx_u8mf2(v586, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v604 = __riscv_vmseq_vx_u8mf2_b16(v603, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v605 = __riscv_vadd_vx_i8mf2_mu(v604, v602, v602, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v606 = __riscv_vwmacc_vx_i16m1(v528, v544, v605, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v607 = __riscv_vsrl_vx_u8mf2(v583, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v608 = __riscv_vand_vx_u8mf2(v607, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v609 = __riscv_vreinterpret_v_u8mf2_i8mf2(v608);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v610 = __riscv_vand_vx_u8mf2(v586, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v611 = __riscv_vmseq_vx_u8mf2_b16(v610, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v612 = __riscv_vadd_vx_i8mf2_mu(v611, v609, v609, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v613 = __riscv_vwmacc_vx_i16m1(v535, v547, v612, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v614 = v21 + 11;
      const int8_t* v615 = (const int8_t*) v614;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v616 = *(const int8_t *)(v615);
      const uint8_t* v617 = v21 + 43;
      const int8_t* v618 = (const int8_t*) v617;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v619 = *(const int8_t *)(v618);
      const uint8_t* v620 = v21 + 75;
      const int8_t* v621 = (const int8_t*) v620;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v622 = *(const int8_t *)(v621);
      const uint8_t* v623 = v21 + 107;
      const int8_t* v624 = (const int8_t*) v623;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v625 = *(const int8_t *)(v624);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v626 = v19 + 912;
      const uint8_t* v627 = (const uint8_t*) v626;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v628 = __riscv_vle8_v_u8mf2(v627, 8);
      const uint8_t* v629 = v19 + 400;
      const uint8_t* v630 = (const uint8_t*) v629;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v631 = __riscv_vle8_v_u8mf2(v630, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v632 = __riscv_vand_vx_u8mf2(v628, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v633 = __riscv_vreinterpret_v_u8mf2_i8mf2(v632);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v634 = __riscv_vand_vx_u8mf2(v631, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v635 = __riscv_vmseq_vx_u8mf2_b16(v634, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v636 = __riscv_vadd_vx_i8mf2_mu(v635, v633, v633, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v637 = __riscv_vwmacc_vx_i16m1(v559, v616, v636, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v638 = __riscv_vsrl_vx_u8mf2(v628, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v639 = __riscv_vand_vx_u8mf2(v638, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v640 = __riscv_vreinterpret_v_u8mf2_i8mf2(v639);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v641 = __riscv_vand_vx_u8mf2(v631, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v642 = __riscv_vmseq_vx_u8mf2_b16(v641, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v643 = __riscv_vadd_vx_i8mf2_mu(v642, v640, v640, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v644 = __riscv_vwmacc_vx_i16m1(v566, v619, v643, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v645 = __riscv_vsrl_vx_u8mf2(v628, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v646 = __riscv_vand_vx_u8mf2(v645, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v647 = __riscv_vreinterpret_v_u8mf2_i8mf2(v646);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v648 = __riscv_vand_vx_u8mf2(v631, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v649 = __riscv_vmseq_vx_u8mf2_b16(v648, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v650 = __riscv_vadd_vx_i8mf2_mu(v649, v647, v647, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v651 = __riscv_vwmacc_vx_i16m1(v573, v622, v650, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v652 = __riscv_vsrl_vx_u8mf2(v628, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v653 = __riscv_vand_vx_u8mf2(v652, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v654 = __riscv_vreinterpret_v_u8mf2_i8mf2(v653);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v655 = __riscv_vand_vx_u8mf2(v631, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v656 = __riscv_vmseq_vx_u8mf2_b16(v655, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v657 = __riscv_vadd_vx_i8mf2_mu(v656, v654, v654, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v658 = __riscv_vwmacc_vx_i16m1(v580, v625, v657, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v659 = v19 + 920;
      const uint8_t* v660 = (const uint8_t*) v659;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v661 = __riscv_vle8_v_u8mf2(v660, 8);
      const uint8_t* v662 = v19 + 408;
      const uint8_t* v663 = (const uint8_t*) v662;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v664 = __riscv_vle8_v_u8mf2(v663, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v665 = __riscv_vand_vx_u8mf2(v661, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v666 = __riscv_vreinterpret_v_u8mf2_i8mf2(v665);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v667 = __riscv_vand_vx_u8mf2(v664, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v668 = __riscv_vmseq_vx_u8mf2_b16(v667, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v669 = __riscv_vadd_vx_i8mf2_mu(v668, v666, v666, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v670 = __riscv_vwmacc_vx_i16m1(v592, v616, v669, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v671 = __riscv_vsrl_vx_u8mf2(v661, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v672 = __riscv_vand_vx_u8mf2(v671, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v673 = __riscv_vreinterpret_v_u8mf2_i8mf2(v672);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v674 = __riscv_vand_vx_u8mf2(v664, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v675 = __riscv_vmseq_vx_u8mf2_b16(v674, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v676 = __riscv_vadd_vx_i8mf2_mu(v675, v673, v673, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v677 = __riscv_vwmacc_vx_i16m1(v599, v619, v676, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v678 = __riscv_vsrl_vx_u8mf2(v661, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v679 = __riscv_vand_vx_u8mf2(v678, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v680 = __riscv_vreinterpret_v_u8mf2_i8mf2(v679);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v681 = __riscv_vand_vx_u8mf2(v664, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v682 = __riscv_vmseq_vx_u8mf2_b16(v681, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v683 = __riscv_vadd_vx_i8mf2_mu(v682, v680, v680, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v684 = __riscv_vwmacc_vx_i16m1(v606, v622, v683, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v685 = __riscv_vsrl_vx_u8mf2(v661, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v686 = __riscv_vand_vx_u8mf2(v685, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v687 = __riscv_vreinterpret_v_u8mf2_i8mf2(v686);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v688 = __riscv_vand_vx_u8mf2(v664, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v689 = __riscv_vmseq_vx_u8mf2_b16(v688, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v690 = __riscv_vadd_vx_i8mf2_mu(v689, v687, v687, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v691 = __riscv_vwmacc_vx_i16m1(v613, v625, v690, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v692 = v21 + 12;
      const int8_t* v693 = (const int8_t*) v692;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v694 = *(const int8_t *)(v693);
      const uint8_t* v695 = v21 + 44;
      const int8_t* v696 = (const int8_t*) v695;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v697 = *(const int8_t *)(v696);
      const uint8_t* v698 = v21 + 76;
      const int8_t* v699 = (const int8_t*) v698;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v700 = *(const int8_t *)(v699);
      const uint8_t* v701 = v21 + 108;
      const int8_t* v702 = (const int8_t*) v701;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v703 = *(const int8_t *)(v702);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v704 = v19 + 928;
      const uint8_t* v705 = (const uint8_t*) v704;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v706 = __riscv_vle8_v_u8mf2(v705, 8);
      const uint8_t* v707 = v19 + 416;
      const uint8_t* v708 = (const uint8_t*) v707;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v709 = __riscv_vle8_v_u8mf2(v708, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v710 = __riscv_vand_vx_u8mf2(v706, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v711 = __riscv_vreinterpret_v_u8mf2_i8mf2(v710);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v712 = __riscv_vand_vx_u8mf2(v709, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v713 = __riscv_vmseq_vx_u8mf2_b16(v712, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v714 = __riscv_vadd_vx_i8mf2_mu(v713, v711, v711, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v715 = __riscv_vwmacc_vx_i16m1(v637, v694, v714, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v716 = __riscv_vsrl_vx_u8mf2(v706, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v717 = __riscv_vand_vx_u8mf2(v716, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v718 = __riscv_vreinterpret_v_u8mf2_i8mf2(v717);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v719 = __riscv_vand_vx_u8mf2(v709, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v720 = __riscv_vmseq_vx_u8mf2_b16(v719, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v721 = __riscv_vadd_vx_i8mf2_mu(v720, v718, v718, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v722 = __riscv_vwmacc_vx_i16m1(v644, v697, v721, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v723 = __riscv_vsrl_vx_u8mf2(v706, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v724 = __riscv_vand_vx_u8mf2(v723, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v725 = __riscv_vreinterpret_v_u8mf2_i8mf2(v724);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v726 = __riscv_vand_vx_u8mf2(v709, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v727 = __riscv_vmseq_vx_u8mf2_b16(v726, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v728 = __riscv_vadd_vx_i8mf2_mu(v727, v725, v725, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v729 = __riscv_vwmacc_vx_i16m1(v651, v700, v728, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v730 = __riscv_vsrl_vx_u8mf2(v706, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v731 = __riscv_vand_vx_u8mf2(v730, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v732 = __riscv_vreinterpret_v_u8mf2_i8mf2(v731);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v733 = __riscv_vand_vx_u8mf2(v709, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v734 = __riscv_vmseq_vx_u8mf2_b16(v733, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v735 = __riscv_vadd_vx_i8mf2_mu(v734, v732, v732, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v736 = __riscv_vwmacc_vx_i16m1(v658, v703, v735, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v737 = v19 + 936;
      const uint8_t* v738 = (const uint8_t*) v737;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v739 = __riscv_vle8_v_u8mf2(v738, 8);
      const uint8_t* v740 = v19 + 424;
      const uint8_t* v741 = (const uint8_t*) v740;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v742 = __riscv_vle8_v_u8mf2(v741, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v743 = __riscv_vand_vx_u8mf2(v739, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v744 = __riscv_vreinterpret_v_u8mf2_i8mf2(v743);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v745 = __riscv_vand_vx_u8mf2(v742, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v746 = __riscv_vmseq_vx_u8mf2_b16(v745, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v747 = __riscv_vadd_vx_i8mf2_mu(v746, v744, v744, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v748 = __riscv_vwmacc_vx_i16m1(v670, v694, v747, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v749 = __riscv_vsrl_vx_u8mf2(v739, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v750 = __riscv_vand_vx_u8mf2(v749, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v751 = __riscv_vreinterpret_v_u8mf2_i8mf2(v750);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v752 = __riscv_vand_vx_u8mf2(v742, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v753 = __riscv_vmseq_vx_u8mf2_b16(v752, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v754 = __riscv_vadd_vx_i8mf2_mu(v753, v751, v751, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v755 = __riscv_vwmacc_vx_i16m1(v677, v697, v754, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v756 = __riscv_vsrl_vx_u8mf2(v739, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v757 = __riscv_vand_vx_u8mf2(v756, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v758 = __riscv_vreinterpret_v_u8mf2_i8mf2(v757);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v759 = __riscv_vand_vx_u8mf2(v742, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v760 = __riscv_vmseq_vx_u8mf2_b16(v759, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v761 = __riscv_vadd_vx_i8mf2_mu(v760, v758, v758, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v762 = __riscv_vwmacc_vx_i16m1(v684, v700, v761, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v763 = __riscv_vsrl_vx_u8mf2(v739, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v764 = __riscv_vand_vx_u8mf2(v763, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v765 = __riscv_vreinterpret_v_u8mf2_i8mf2(v764);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v766 = __riscv_vand_vx_u8mf2(v742, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v767 = __riscv_vmseq_vx_u8mf2_b16(v766, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v768 = __riscv_vadd_vx_i8mf2_mu(v767, v765, v765, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v769 = __riscv_vwmacc_vx_i16m1(v691, v703, v768, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v770 = v21 + 13;
      const int8_t* v771 = (const int8_t*) v770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v772 = *(const int8_t *)(v771);
      const uint8_t* v773 = v21 + 45;
      const int8_t* v774 = (const int8_t*) v773;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v775 = *(const int8_t *)(v774);
      const uint8_t* v776 = v21 + 77;
      const int8_t* v777 = (const int8_t*) v776;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v778 = *(const int8_t *)(v777);
      const uint8_t* v779 = v21 + 109;
      const int8_t* v780 = (const int8_t*) v779;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v781 = *(const int8_t *)(v780);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v782 = v19 + 944;
      const uint8_t* v783 = (const uint8_t*) v782;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v784 = __riscv_vle8_v_u8mf2(v783, 8);
      const uint8_t* v785 = v19 + 432;
      const uint8_t* v786 = (const uint8_t*) v785;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v787 = __riscv_vle8_v_u8mf2(v786, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v788 = __riscv_vand_vx_u8mf2(v784, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v789 = __riscv_vreinterpret_v_u8mf2_i8mf2(v788);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v790 = __riscv_vand_vx_u8mf2(v787, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v791 = __riscv_vmseq_vx_u8mf2_b16(v790, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v792 = __riscv_vadd_vx_i8mf2_mu(v791, v789, v789, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v793 = __riscv_vwmacc_vx_i16m1(v715, v772, v792, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v794 = __riscv_vsrl_vx_u8mf2(v784, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v795 = __riscv_vand_vx_u8mf2(v794, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v796 = __riscv_vreinterpret_v_u8mf2_i8mf2(v795);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v797 = __riscv_vand_vx_u8mf2(v787, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v798 = __riscv_vmseq_vx_u8mf2_b16(v797, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v799 = __riscv_vadd_vx_i8mf2_mu(v798, v796, v796, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v800 = __riscv_vwmacc_vx_i16m1(v722, v775, v799, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v801 = __riscv_vsrl_vx_u8mf2(v784, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v802 = __riscv_vand_vx_u8mf2(v801, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v803 = __riscv_vreinterpret_v_u8mf2_i8mf2(v802);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v804 = __riscv_vand_vx_u8mf2(v787, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v805 = __riscv_vmseq_vx_u8mf2_b16(v804, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v806 = __riscv_vadd_vx_i8mf2_mu(v805, v803, v803, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v807 = __riscv_vwmacc_vx_i16m1(v729, v778, v806, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v808 = __riscv_vsrl_vx_u8mf2(v784, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v809 = __riscv_vand_vx_u8mf2(v808, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v810 = __riscv_vreinterpret_v_u8mf2_i8mf2(v809);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v811 = __riscv_vand_vx_u8mf2(v787, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v812 = __riscv_vmseq_vx_u8mf2_b16(v811, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v813 = __riscv_vadd_vx_i8mf2_mu(v812, v810, v810, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v814 = __riscv_vwmacc_vx_i16m1(v736, v781, v813, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v815 = v19 + 952;
      const uint8_t* v816 = (const uint8_t*) v815;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v817 = __riscv_vle8_v_u8mf2(v816, 8);
      const uint8_t* v818 = v19 + 440;
      const uint8_t* v819 = (const uint8_t*) v818;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v820 = __riscv_vle8_v_u8mf2(v819, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v821 = __riscv_vand_vx_u8mf2(v817, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v822 = __riscv_vreinterpret_v_u8mf2_i8mf2(v821);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v823 = __riscv_vand_vx_u8mf2(v820, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v824 = __riscv_vmseq_vx_u8mf2_b16(v823, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v825 = __riscv_vadd_vx_i8mf2_mu(v824, v822, v822, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v826 = __riscv_vwmacc_vx_i16m1(v748, v772, v825, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v827 = __riscv_vsrl_vx_u8mf2(v817, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v828 = __riscv_vand_vx_u8mf2(v827, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v829 = __riscv_vreinterpret_v_u8mf2_i8mf2(v828);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v830 = __riscv_vand_vx_u8mf2(v820, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v831 = __riscv_vmseq_vx_u8mf2_b16(v830, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v832 = __riscv_vadd_vx_i8mf2_mu(v831, v829, v829, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v833 = __riscv_vwmacc_vx_i16m1(v755, v775, v832, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v834 = __riscv_vsrl_vx_u8mf2(v817, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v835 = __riscv_vand_vx_u8mf2(v834, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v836 = __riscv_vreinterpret_v_u8mf2_i8mf2(v835);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v837 = __riscv_vand_vx_u8mf2(v820, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v838 = __riscv_vmseq_vx_u8mf2_b16(v837, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v839 = __riscv_vadd_vx_i8mf2_mu(v838, v836, v836, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v840 = __riscv_vwmacc_vx_i16m1(v762, v778, v839, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v841 = __riscv_vsrl_vx_u8mf2(v817, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v842 = __riscv_vand_vx_u8mf2(v841, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v843 = __riscv_vreinterpret_v_u8mf2_i8mf2(v842);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v844 = __riscv_vand_vx_u8mf2(v820, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v845 = __riscv_vmseq_vx_u8mf2_b16(v844, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v846 = __riscv_vadd_vx_i8mf2_mu(v845, v843, v843, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v847 = __riscv_vwmacc_vx_i16m1(v769, v781, v846, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v848 = v21 + 14;
      const int8_t* v849 = (const int8_t*) v848;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v850 = *(const int8_t *)(v849);
      const uint8_t* v851 = v21 + 46;
      const int8_t* v852 = (const int8_t*) v851;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v853 = *(const int8_t *)(v852);
      const uint8_t* v854 = v21 + 78;
      const int8_t* v855 = (const int8_t*) v854;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v856 = *(const int8_t *)(v855);
      const uint8_t* v857 = v21 + 110;
      const int8_t* v858 = (const int8_t*) v857;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v859 = *(const int8_t *)(v858);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v860 = v19 + 960;
      const uint8_t* v861 = (const uint8_t*) v860;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v862 = __riscv_vle8_v_u8mf2(v861, 8);
      const uint8_t* v863 = v19 + 448;
      const uint8_t* v864 = (const uint8_t*) v863;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v865 = __riscv_vle8_v_u8mf2(v864, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v866 = __riscv_vand_vx_u8mf2(v862, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v867 = __riscv_vreinterpret_v_u8mf2_i8mf2(v866);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v868 = __riscv_vand_vx_u8mf2(v865, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v869 = __riscv_vmseq_vx_u8mf2_b16(v868, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v870 = __riscv_vadd_vx_i8mf2_mu(v869, v867, v867, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v871 = __riscv_vwmacc_vx_i16m1(v793, v850, v870, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v872 = __riscv_vsrl_vx_u8mf2(v862, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v873 = __riscv_vand_vx_u8mf2(v872, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v874 = __riscv_vreinterpret_v_u8mf2_i8mf2(v873);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v875 = __riscv_vand_vx_u8mf2(v865, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v876 = __riscv_vmseq_vx_u8mf2_b16(v875, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v877 = __riscv_vadd_vx_i8mf2_mu(v876, v874, v874, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v878 = __riscv_vwmacc_vx_i16m1(v800, v853, v877, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v879 = __riscv_vsrl_vx_u8mf2(v862, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v880 = __riscv_vand_vx_u8mf2(v879, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v881 = __riscv_vreinterpret_v_u8mf2_i8mf2(v880);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v882 = __riscv_vand_vx_u8mf2(v865, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v883 = __riscv_vmseq_vx_u8mf2_b16(v882, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v884 = __riscv_vadd_vx_i8mf2_mu(v883, v881, v881, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v885 = __riscv_vwmacc_vx_i16m1(v807, v856, v884, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v886 = __riscv_vsrl_vx_u8mf2(v862, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v887 = __riscv_vand_vx_u8mf2(v886, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v888 = __riscv_vreinterpret_v_u8mf2_i8mf2(v887);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v889 = __riscv_vand_vx_u8mf2(v865, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v890 = __riscv_vmseq_vx_u8mf2_b16(v889, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v891 = __riscv_vadd_vx_i8mf2_mu(v890, v888, v888, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v892 = __riscv_vwmacc_vx_i16m1(v814, v859, v891, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v893 = v19 + 968;
      const uint8_t* v894 = (const uint8_t*) v893;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v895 = __riscv_vle8_v_u8mf2(v894, 8);
      const uint8_t* v896 = v19 + 456;
      const uint8_t* v897 = (const uint8_t*) v896;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v898 = __riscv_vle8_v_u8mf2(v897, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v899 = __riscv_vand_vx_u8mf2(v895, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v900 = __riscv_vreinterpret_v_u8mf2_i8mf2(v899);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v901 = __riscv_vand_vx_u8mf2(v898, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v902 = __riscv_vmseq_vx_u8mf2_b16(v901, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v903 = __riscv_vadd_vx_i8mf2_mu(v902, v900, v900, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v904 = __riscv_vwmacc_vx_i16m1(v826, v850, v903, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v905 = __riscv_vsrl_vx_u8mf2(v895, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v906 = __riscv_vand_vx_u8mf2(v905, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v907 = __riscv_vreinterpret_v_u8mf2_i8mf2(v906);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v908 = __riscv_vand_vx_u8mf2(v898, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v909 = __riscv_vmseq_vx_u8mf2_b16(v908, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v910 = __riscv_vadd_vx_i8mf2_mu(v909, v907, v907, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v911 = __riscv_vwmacc_vx_i16m1(v833, v853, v910, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v912 = __riscv_vsrl_vx_u8mf2(v895, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v913 = __riscv_vand_vx_u8mf2(v912, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v914 = __riscv_vreinterpret_v_u8mf2_i8mf2(v913);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v915 = __riscv_vand_vx_u8mf2(v898, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v916 = __riscv_vmseq_vx_u8mf2_b16(v915, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v917 = __riscv_vadd_vx_i8mf2_mu(v916, v914, v914, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v918 = __riscv_vwmacc_vx_i16m1(v840, v856, v917, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v919 = __riscv_vsrl_vx_u8mf2(v895, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v920 = __riscv_vand_vx_u8mf2(v919, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v921 = __riscv_vreinterpret_v_u8mf2_i8mf2(v920);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v922 = __riscv_vand_vx_u8mf2(v898, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v923 = __riscv_vmseq_vx_u8mf2_b16(v922, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v924 = __riscv_vadd_vx_i8mf2_mu(v923, v921, v921, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v925 = __riscv_vwmacc_vx_i16m1(v847, v859, v924, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v926 = v21 + 15;
      const int8_t* v927 = (const int8_t*) v926;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v928 = *(const int8_t *)(v927);
      const uint8_t* v929 = v21 + 47;
      const int8_t* v930 = (const int8_t*) v929;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v931 = *(const int8_t *)(v930);
      const uint8_t* v932 = v21 + 79;
      const int8_t* v933 = (const int8_t*) v932;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v934 = *(const int8_t *)(v933);
      const uint8_t* v935 = v21 + 111;
      const int8_t* v936 = (const int8_t*) v935;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v937 = *(const int8_t *)(v936);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v938 = v19 + 976;
      const uint8_t* v939 = (const uint8_t*) v938;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v940 = __riscv_vle8_v_u8mf2(v939, 8);
      const uint8_t* v941 = v19 + 464;
      const uint8_t* v942 = (const uint8_t*) v941;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v943 = __riscv_vle8_v_u8mf2(v942, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v944 = __riscv_vand_vx_u8mf2(v940, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v945 = __riscv_vreinterpret_v_u8mf2_i8mf2(v944);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v946 = __riscv_vand_vx_u8mf2(v943, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v947 = __riscv_vmseq_vx_u8mf2_b16(v946, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v948 = __riscv_vadd_vx_i8mf2_mu(v947, v945, v945, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v949 = __riscv_vwmacc_vx_i16m1(v871, v928, v948, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v950 = __riscv_vsrl_vx_u8mf2(v940, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v951 = __riscv_vand_vx_u8mf2(v950, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v952 = __riscv_vreinterpret_v_u8mf2_i8mf2(v951);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v953 = __riscv_vand_vx_u8mf2(v943, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v954 = __riscv_vmseq_vx_u8mf2_b16(v953, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v955 = __riscv_vadd_vx_i8mf2_mu(v954, v952, v952, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v956 = __riscv_vwmacc_vx_i16m1(v878, v931, v955, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v957 = __riscv_vsrl_vx_u8mf2(v940, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v958 = __riscv_vand_vx_u8mf2(v957, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v959 = __riscv_vreinterpret_v_u8mf2_i8mf2(v958);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v960 = __riscv_vand_vx_u8mf2(v943, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v961 = __riscv_vmseq_vx_u8mf2_b16(v960, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v962 = __riscv_vadd_vx_i8mf2_mu(v961, v959, v959, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v963 = __riscv_vwmacc_vx_i16m1(v885, v934, v962, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v964 = __riscv_vsrl_vx_u8mf2(v940, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v965 = __riscv_vand_vx_u8mf2(v964, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v966 = __riscv_vreinterpret_v_u8mf2_i8mf2(v965);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v967 = __riscv_vand_vx_u8mf2(v943, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v968 = __riscv_vmseq_vx_u8mf2_b16(v967, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v969 = __riscv_vadd_vx_i8mf2_mu(v968, v966, v966, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v970 = __riscv_vwmacc_vx_i16m1(v892, v937, v969, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v971 = v19 + 984;
      const uint8_t* v972 = (const uint8_t*) v971;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v973 = __riscv_vle8_v_u8mf2(v972, 8);
      const uint8_t* v974 = v19 + 472;
      const uint8_t* v975 = (const uint8_t*) v974;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v976 = __riscv_vle8_v_u8mf2(v975, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v977 = __riscv_vand_vx_u8mf2(v973, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v978 = __riscv_vreinterpret_v_u8mf2_i8mf2(v977);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v979 = __riscv_vand_vx_u8mf2(v976, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v980 = __riscv_vmseq_vx_u8mf2_b16(v979, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v981 = __riscv_vadd_vx_i8mf2_mu(v980, v978, v978, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v982 = __riscv_vwmacc_vx_i16m1(v904, v928, v981, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v983 = __riscv_vsrl_vx_u8mf2(v973, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v984 = __riscv_vand_vx_u8mf2(v983, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v985 = __riscv_vreinterpret_v_u8mf2_i8mf2(v984);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v986 = __riscv_vand_vx_u8mf2(v976, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v987 = __riscv_vmseq_vx_u8mf2_b16(v986, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v988 = __riscv_vadd_vx_i8mf2_mu(v987, v985, v985, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v989 = __riscv_vwmacc_vx_i16m1(v911, v931, v988, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v990 = __riscv_vsrl_vx_u8mf2(v973, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v991 = __riscv_vand_vx_u8mf2(v990, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v992 = __riscv_vreinterpret_v_u8mf2_i8mf2(v991);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v993 = __riscv_vand_vx_u8mf2(v976, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v994 = __riscv_vmseq_vx_u8mf2_b16(v993, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v995 = __riscv_vadd_vx_i8mf2_mu(v994, v992, v992, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v996 = __riscv_vwmacc_vx_i16m1(v918, v934, v995, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v997 = __riscv_vsrl_vx_u8mf2(v973, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v998 = __riscv_vand_vx_u8mf2(v997, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v999 = __riscv_vreinterpret_v_u8mf2_i8mf2(v998);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1000 = __riscv_vand_vx_u8mf2(v976, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1001 = __riscv_vmseq_vx_u8mf2_b16(v1000, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1002 = __riscv_vadd_vx_i8mf2_mu(v1001, v999, v999, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1003 = __riscv_vwmacc_vx_i16m1(v925, v937, v1002, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1004 = v21 + 16;
      const int8_t* v1005 = (const int8_t*) v1004;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1006 = *(const int8_t *)(v1005);
      const uint8_t* v1007 = v21 + 48;
      const int8_t* v1008 = (const int8_t*) v1007;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1009 = *(const int8_t *)(v1008);
      const uint8_t* v1010 = v21 + 80;
      const int8_t* v1011 = (const int8_t*) v1010;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1012 = *(const int8_t *)(v1011);
      const uint8_t* v1013 = v21 + 112;
      const int8_t* v1014 = (const int8_t*) v1013;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1015 = *(const int8_t *)(v1014);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1016 = v19 + 992;
      const uint8_t* v1017 = (const uint8_t*) v1016;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1018 = __riscv_vle8_v_u8mf2(v1017, 8);
      const uint8_t* v1019 = v19 + 480;
      const uint8_t* v1020 = (const uint8_t*) v1019;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1021 = __riscv_vle8_v_u8mf2(v1020, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1022 = __riscv_vand_vx_u8mf2(v1018, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1023 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1022);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1024 = __riscv_vand_vx_u8mf2(v1021, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1025 = __riscv_vmseq_vx_u8mf2_b16(v1024, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1026 = __riscv_vadd_vx_i8mf2_mu(v1025, v1023, v1023, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1027 = __riscv_vwmacc_vx_i16m1(v949, v1006, v1026, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1028 = __riscv_vsrl_vx_u8mf2(v1018, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1029 = __riscv_vand_vx_u8mf2(v1028, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1030 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1029);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1031 = __riscv_vand_vx_u8mf2(v1021, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1032 = __riscv_vmseq_vx_u8mf2_b16(v1031, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1033 = __riscv_vadd_vx_i8mf2_mu(v1032, v1030, v1030, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1034 = __riscv_vwmacc_vx_i16m1(v956, v1009, v1033, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1035 = __riscv_vsrl_vx_u8mf2(v1018, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1036 = __riscv_vand_vx_u8mf2(v1035, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1037 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1036);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1038 = __riscv_vand_vx_u8mf2(v1021, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1039 = __riscv_vmseq_vx_u8mf2_b16(v1038, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1040 = __riscv_vadd_vx_i8mf2_mu(v1039, v1037, v1037, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1041 = __riscv_vwmacc_vx_i16m1(v963, v1012, v1040, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1042 = __riscv_vsrl_vx_u8mf2(v1018, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1043 = __riscv_vand_vx_u8mf2(v1042, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1044 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1043);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1045 = __riscv_vand_vx_u8mf2(v1021, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1046 = __riscv_vmseq_vx_u8mf2_b16(v1045, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1047 = __riscv_vadd_vx_i8mf2_mu(v1046, v1044, v1044, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1048 = __riscv_vwmacc_vx_i16m1(v970, v1015, v1047, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1049 = v19 + 1000;
      const uint8_t* v1050 = (const uint8_t*) v1049;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1051 = __riscv_vle8_v_u8mf2(v1050, 8);
      const uint8_t* v1052 = v19 + 488;
      const uint8_t* v1053 = (const uint8_t*) v1052;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1054 = __riscv_vle8_v_u8mf2(v1053, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1055 = __riscv_vand_vx_u8mf2(v1051, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1056 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1055);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1057 = __riscv_vand_vx_u8mf2(v1054, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1058 = __riscv_vmseq_vx_u8mf2_b16(v1057, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1059 = __riscv_vadd_vx_i8mf2_mu(v1058, v1056, v1056, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1060 = __riscv_vwmacc_vx_i16m1(v982, v1006, v1059, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1061 = __riscv_vsrl_vx_u8mf2(v1051, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1062 = __riscv_vand_vx_u8mf2(v1061, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1063 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1062);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1064 = __riscv_vand_vx_u8mf2(v1054, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1065 = __riscv_vmseq_vx_u8mf2_b16(v1064, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1066 = __riscv_vadd_vx_i8mf2_mu(v1065, v1063, v1063, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1067 = __riscv_vwmacc_vx_i16m1(v989, v1009, v1066, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1068 = __riscv_vsrl_vx_u8mf2(v1051, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1069 = __riscv_vand_vx_u8mf2(v1068, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1070 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1069);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1071 = __riscv_vand_vx_u8mf2(v1054, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1072 = __riscv_vmseq_vx_u8mf2_b16(v1071, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1073 = __riscv_vadd_vx_i8mf2_mu(v1072, v1070, v1070, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1074 = __riscv_vwmacc_vx_i16m1(v996, v1012, v1073, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1075 = __riscv_vsrl_vx_u8mf2(v1051, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1076 = __riscv_vand_vx_u8mf2(v1075, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1077 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1076);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1078 = __riscv_vand_vx_u8mf2(v1054, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1079 = __riscv_vmseq_vx_u8mf2_b16(v1078, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1080 = __riscv_vadd_vx_i8mf2_mu(v1079, v1077, v1077, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1081 = __riscv_vwmacc_vx_i16m1(v1003, v1015, v1080, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1082 = v21 + 17;
      const int8_t* v1083 = (const int8_t*) v1082;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1084 = *(const int8_t *)(v1083);
      const uint8_t* v1085 = v21 + 49;
      const int8_t* v1086 = (const int8_t*) v1085;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1087 = *(const int8_t *)(v1086);
      const uint8_t* v1088 = v21 + 81;
      const int8_t* v1089 = (const int8_t*) v1088;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1090 = *(const int8_t *)(v1089);
      const uint8_t* v1091 = v21 + 113;
      const int8_t* v1092 = (const int8_t*) v1091;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1093 = *(const int8_t *)(v1092);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1094 = v19 + 1008;
      const uint8_t* v1095 = (const uint8_t*) v1094;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1096 = __riscv_vle8_v_u8mf2(v1095, 8);
      const uint8_t* v1097 = v19 + 496;
      const uint8_t* v1098 = (const uint8_t*) v1097;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1099 = __riscv_vle8_v_u8mf2(v1098, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1100 = __riscv_vand_vx_u8mf2(v1096, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1101 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1100);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1102 = __riscv_vand_vx_u8mf2(v1099, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1103 = __riscv_vmseq_vx_u8mf2_b16(v1102, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1104 = __riscv_vadd_vx_i8mf2_mu(v1103, v1101, v1101, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1105 = __riscv_vwmacc_vx_i16m1(v1027, v1084, v1104, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1106 = __riscv_vsrl_vx_u8mf2(v1096, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1107 = __riscv_vand_vx_u8mf2(v1106, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1108 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1107);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1109 = __riscv_vand_vx_u8mf2(v1099, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1110 = __riscv_vmseq_vx_u8mf2_b16(v1109, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1111 = __riscv_vadd_vx_i8mf2_mu(v1110, v1108, v1108, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1112 = __riscv_vwmacc_vx_i16m1(v1034, v1087, v1111, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1113 = __riscv_vsrl_vx_u8mf2(v1096, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1114 = __riscv_vand_vx_u8mf2(v1113, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1115 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1114);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1116 = __riscv_vand_vx_u8mf2(v1099, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1117 = __riscv_vmseq_vx_u8mf2_b16(v1116, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1118 = __riscv_vadd_vx_i8mf2_mu(v1117, v1115, v1115, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1119 = __riscv_vwmacc_vx_i16m1(v1041, v1090, v1118, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1120 = __riscv_vsrl_vx_u8mf2(v1096, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1121 = __riscv_vand_vx_u8mf2(v1120, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1122 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1121);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1123 = __riscv_vand_vx_u8mf2(v1099, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1124 = __riscv_vmseq_vx_u8mf2_b16(v1123, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1125 = __riscv_vadd_vx_i8mf2_mu(v1124, v1122, v1122, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1126 = __riscv_vwmacc_vx_i16m1(v1048, v1093, v1125, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1127 = v19 + 1016;
      const uint8_t* v1128 = (const uint8_t*) v1127;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1129 = __riscv_vle8_v_u8mf2(v1128, 8);
      const uint8_t* v1130 = v19 + 504;
      const uint8_t* v1131 = (const uint8_t*) v1130;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1132 = __riscv_vle8_v_u8mf2(v1131, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1133 = __riscv_vand_vx_u8mf2(v1129, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1134 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1133);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1135 = __riscv_vand_vx_u8mf2(v1132, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1136 = __riscv_vmseq_vx_u8mf2_b16(v1135, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1137 = __riscv_vadd_vx_i8mf2_mu(v1136, v1134, v1134, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1138 = __riscv_vwmacc_vx_i16m1(v1060, v1084, v1137, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1139 = __riscv_vsrl_vx_u8mf2(v1129, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1140 = __riscv_vand_vx_u8mf2(v1139, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1141 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1140);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1142 = __riscv_vand_vx_u8mf2(v1132, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1143 = __riscv_vmseq_vx_u8mf2_b16(v1142, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1144 = __riscv_vadd_vx_i8mf2_mu(v1143, v1141, v1141, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1145 = __riscv_vwmacc_vx_i16m1(v1067, v1087, v1144, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1146 = __riscv_vsrl_vx_u8mf2(v1129, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1147 = __riscv_vand_vx_u8mf2(v1146, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1148 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1147);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1149 = __riscv_vand_vx_u8mf2(v1132, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1150 = __riscv_vmseq_vx_u8mf2_b16(v1149, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1151 = __riscv_vadd_vx_i8mf2_mu(v1150, v1148, v1148, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1152 = __riscv_vwmacc_vx_i16m1(v1074, v1090, v1151, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1153 = __riscv_vsrl_vx_u8mf2(v1129, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1154 = __riscv_vand_vx_u8mf2(v1153, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1155 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1154);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1156 = __riscv_vand_vx_u8mf2(v1132, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1157 = __riscv_vmseq_vx_u8mf2_b16(v1156, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1158 = __riscv_vadd_vx_i8mf2_mu(v1157, v1155, v1155, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1159 = __riscv_vwmacc_vx_i16m1(v1081, v1093, v1158, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1160 = v21 + 18;
      const int8_t* v1161 = (const int8_t*) v1160;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1162 = *(const int8_t *)(v1161);
      const uint8_t* v1163 = v21 + 50;
      const int8_t* v1164 = (const int8_t*) v1163;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1165 = *(const int8_t *)(v1164);
      const uint8_t* v1166 = v21 + 82;
      const int8_t* v1167 = (const int8_t*) v1166;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1168 = *(const int8_t *)(v1167);
      const uint8_t* v1169 = v21 + 114;
      const int8_t* v1170 = (const int8_t*) v1169;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1171 = *(const int8_t *)(v1170);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1172 = v19 + 1024;
      const uint8_t* v1173 = (const uint8_t*) v1172;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1174 = __riscv_vle8_v_u8mf2(v1173, 8);
      const uint8_t* v1175 = v19 + 512;
      const uint8_t* v1176 = (const uint8_t*) v1175;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1177 = __riscv_vle8_v_u8mf2(v1176, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1178 = __riscv_vand_vx_u8mf2(v1174, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1179 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1178);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1180 = __riscv_vand_vx_u8mf2(v1177, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1181 = __riscv_vmseq_vx_u8mf2_b16(v1180, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1182 = __riscv_vadd_vx_i8mf2_mu(v1181, v1179, v1179, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1183 = __riscv_vwmacc_vx_i16m1(v1105, v1162, v1182, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1184 = __riscv_vsrl_vx_u8mf2(v1174, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1185 = __riscv_vand_vx_u8mf2(v1184, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1186 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1185);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1187 = __riscv_vand_vx_u8mf2(v1177, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1188 = __riscv_vmseq_vx_u8mf2_b16(v1187, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1189 = __riscv_vadd_vx_i8mf2_mu(v1188, v1186, v1186, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1190 = __riscv_vwmacc_vx_i16m1(v1112, v1165, v1189, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1191 = __riscv_vsrl_vx_u8mf2(v1174, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1192 = __riscv_vand_vx_u8mf2(v1191, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1193 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1192);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1194 = __riscv_vand_vx_u8mf2(v1177, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1195 = __riscv_vmseq_vx_u8mf2_b16(v1194, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1196 = __riscv_vadd_vx_i8mf2_mu(v1195, v1193, v1193, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1197 = __riscv_vwmacc_vx_i16m1(v1119, v1168, v1196, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1198 = __riscv_vsrl_vx_u8mf2(v1174, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1199 = __riscv_vand_vx_u8mf2(v1198, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1200 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1199);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1201 = __riscv_vand_vx_u8mf2(v1177, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1202 = __riscv_vmseq_vx_u8mf2_b16(v1201, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1203 = __riscv_vadd_vx_i8mf2_mu(v1202, v1200, v1200, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1204 = __riscv_vwmacc_vx_i16m1(v1126, v1171, v1203, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1205 = v19 + 1032;
      const uint8_t* v1206 = (const uint8_t*) v1205;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1207 = __riscv_vle8_v_u8mf2(v1206, 8);
      const uint8_t* v1208 = v19 + 520;
      const uint8_t* v1209 = (const uint8_t*) v1208;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1210 = __riscv_vle8_v_u8mf2(v1209, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1211 = __riscv_vand_vx_u8mf2(v1207, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1212 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1211);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1213 = __riscv_vand_vx_u8mf2(v1210, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1214 = __riscv_vmseq_vx_u8mf2_b16(v1213, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1215 = __riscv_vadd_vx_i8mf2_mu(v1214, v1212, v1212, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1216 = __riscv_vwmacc_vx_i16m1(v1138, v1162, v1215, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1217 = __riscv_vsrl_vx_u8mf2(v1207, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1218 = __riscv_vand_vx_u8mf2(v1217, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1219 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1218);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1220 = __riscv_vand_vx_u8mf2(v1210, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1221 = __riscv_vmseq_vx_u8mf2_b16(v1220, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1222 = __riscv_vadd_vx_i8mf2_mu(v1221, v1219, v1219, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1223 = __riscv_vwmacc_vx_i16m1(v1145, v1165, v1222, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1224 = __riscv_vsrl_vx_u8mf2(v1207, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1225 = __riscv_vand_vx_u8mf2(v1224, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1226 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1225);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1227 = __riscv_vand_vx_u8mf2(v1210, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1228 = __riscv_vmseq_vx_u8mf2_b16(v1227, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1229 = __riscv_vadd_vx_i8mf2_mu(v1228, v1226, v1226, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1230 = __riscv_vwmacc_vx_i16m1(v1152, v1168, v1229, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1231 = __riscv_vsrl_vx_u8mf2(v1207, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1232 = __riscv_vand_vx_u8mf2(v1231, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1233 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1232);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1234 = __riscv_vand_vx_u8mf2(v1210, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1235 = __riscv_vmseq_vx_u8mf2_b16(v1234, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1236 = __riscv_vadd_vx_i8mf2_mu(v1235, v1233, v1233, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1237 = __riscv_vwmacc_vx_i16m1(v1159, v1171, v1236, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1238 = v21 + 19;
      const int8_t* v1239 = (const int8_t*) v1238;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1240 = *(const int8_t *)(v1239);
      const uint8_t* v1241 = v21 + 51;
      const int8_t* v1242 = (const int8_t*) v1241;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1243 = *(const int8_t *)(v1242);
      const uint8_t* v1244 = v21 + 83;
      const int8_t* v1245 = (const int8_t*) v1244;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1246 = *(const int8_t *)(v1245);
      const uint8_t* v1247 = v21 + 115;
      const int8_t* v1248 = (const int8_t*) v1247;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1249 = *(const int8_t *)(v1248);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1250 = v19 + 1040;
      const uint8_t* v1251 = (const uint8_t*) v1250;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1252 = __riscv_vle8_v_u8mf2(v1251, 8);
      const uint8_t* v1253 = v19 + 528;
      const uint8_t* v1254 = (const uint8_t*) v1253;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1255 = __riscv_vle8_v_u8mf2(v1254, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1256 = __riscv_vand_vx_u8mf2(v1252, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1257 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1256);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1258 = __riscv_vand_vx_u8mf2(v1255, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1259 = __riscv_vmseq_vx_u8mf2_b16(v1258, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1260 = __riscv_vadd_vx_i8mf2_mu(v1259, v1257, v1257, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1261 = __riscv_vwmacc_vx_i16m1(v1183, v1240, v1260, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1262 = __riscv_vsrl_vx_u8mf2(v1252, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1263 = __riscv_vand_vx_u8mf2(v1262, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1264 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1263);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1265 = __riscv_vand_vx_u8mf2(v1255, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1266 = __riscv_vmseq_vx_u8mf2_b16(v1265, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1267 = __riscv_vadd_vx_i8mf2_mu(v1266, v1264, v1264, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1268 = __riscv_vwmacc_vx_i16m1(v1190, v1243, v1267, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1269 = __riscv_vsrl_vx_u8mf2(v1252, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1270 = __riscv_vand_vx_u8mf2(v1269, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1271 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1270);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1272 = __riscv_vand_vx_u8mf2(v1255, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1273 = __riscv_vmseq_vx_u8mf2_b16(v1272, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1274 = __riscv_vadd_vx_i8mf2_mu(v1273, v1271, v1271, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1275 = __riscv_vwmacc_vx_i16m1(v1197, v1246, v1274, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1276 = __riscv_vsrl_vx_u8mf2(v1252, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1277 = __riscv_vand_vx_u8mf2(v1276, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1278 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1277);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1279 = __riscv_vand_vx_u8mf2(v1255, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1280 = __riscv_vmseq_vx_u8mf2_b16(v1279, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1281 = __riscv_vadd_vx_i8mf2_mu(v1280, v1278, v1278, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1282 = __riscv_vwmacc_vx_i16m1(v1204, v1249, v1281, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1283 = v19 + 1048;
      const uint8_t* v1284 = (const uint8_t*) v1283;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1285 = __riscv_vle8_v_u8mf2(v1284, 8);
      const uint8_t* v1286 = v19 + 536;
      const uint8_t* v1287 = (const uint8_t*) v1286;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1288 = __riscv_vle8_v_u8mf2(v1287, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1289 = __riscv_vand_vx_u8mf2(v1285, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1290 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1289);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1291 = __riscv_vand_vx_u8mf2(v1288, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1292 = __riscv_vmseq_vx_u8mf2_b16(v1291, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1293 = __riscv_vadd_vx_i8mf2_mu(v1292, v1290, v1290, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1294 = __riscv_vwmacc_vx_i16m1(v1216, v1240, v1293, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1295 = __riscv_vsrl_vx_u8mf2(v1285, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1296 = __riscv_vand_vx_u8mf2(v1295, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1297 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1296);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1298 = __riscv_vand_vx_u8mf2(v1288, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1299 = __riscv_vmseq_vx_u8mf2_b16(v1298, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1300 = __riscv_vadd_vx_i8mf2_mu(v1299, v1297, v1297, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1301 = __riscv_vwmacc_vx_i16m1(v1223, v1243, v1300, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1302 = __riscv_vsrl_vx_u8mf2(v1285, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1303 = __riscv_vand_vx_u8mf2(v1302, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1304 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1303);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1305 = __riscv_vand_vx_u8mf2(v1288, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1306 = __riscv_vmseq_vx_u8mf2_b16(v1305, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1307 = __riscv_vadd_vx_i8mf2_mu(v1306, v1304, v1304, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1308 = __riscv_vwmacc_vx_i16m1(v1230, v1246, v1307, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1309 = __riscv_vsrl_vx_u8mf2(v1285, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1310 = __riscv_vand_vx_u8mf2(v1309, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1311 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1310);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1312 = __riscv_vand_vx_u8mf2(v1288, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1313 = __riscv_vmseq_vx_u8mf2_b16(v1312, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1314 = __riscv_vadd_vx_i8mf2_mu(v1313, v1311, v1311, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1315 = __riscv_vwmacc_vx_i16m1(v1237, v1249, v1314, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v1316 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1317 = __riscv_vwmacc_vv_i32m2(v1316, v31, v1261, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1318 = __riscv_vwmacc_vv_i32m2(v1317, v35, v1268, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1319 = __riscv_vwmacc_vv_i32m2(v1318, v39, v1275, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1320 = __riscv_vwmacc_vv_i32m2(v1319, v43, v1282, 8);
      v24 = v1320;
      vint32m2_t v1321 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1322 = __riscv_vwmacc_vv_i32m2(v1321, v47, v1294, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1323 = __riscv_vwmacc_vv_i32m2(v1322, v51, v1301, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1324 = __riscv_vwmacc_vv_i32m2(v1323, v55, v1308, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1325 = __riscv_vwmacc_vv_i32m2(v1324, v59, v1315, 8);
      v26 = v1325;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
      const uint8_t* v1326 = v19 + 48;
      const int8_t* v1327 = (const int8_t*) v1326;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v1328 = __riscv_vle8_v_i8mf2(v1327, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v1329 = __riscv_vsext_vf2_i16m1(v1328, 8);
      const uint8_t* v1330 = v19 + 80;
      const int8_t* v1331 = (const int8_t*) v1330;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v1332 = __riscv_vle8_v_i8mf2(v1331, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v1333 = __riscv_vsext_vf2_i16m1(v1332, 8);
      const uint8_t* v1334 = v19 + 112;
      const int8_t* v1335 = (const int8_t*) v1334;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v1336 = __riscv_vle8_v_i8mf2(v1335, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v1337 = __riscv_vsext_vf2_i16m1(v1336, 8);
      const uint8_t* v1338 = v19 + 144;
      const int8_t* v1339 = (const int8_t*) v1338;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v1340 = __riscv_vle8_v_i8mf2(v1339, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v1341 = __riscv_vsext_vf2_i16m1(v1340, 8);
      const uint8_t* v1342 = v19 + 56;
      const int8_t* v1343 = (const int8_t*) v1342;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v1344 = __riscv_vle8_v_i8mf2(v1343, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v1345 = __riscv_vsext_vf2_i16m1(v1344, 8);
      const uint8_t* v1346 = v19 + 88;
      const int8_t* v1347 = (const int8_t*) v1346;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v1348 = __riscv_vle8_v_i8mf2(v1347, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v1349 = __riscv_vsext_vf2_i16m1(v1348, 8);
      const uint8_t* v1350 = v19 + 120;
      const int8_t* v1351 = (const int8_t*) v1350;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v1352 = __riscv_vle8_v_i8mf2(v1351, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v1353 = __riscv_vsext_vf2_i16m1(v1352, 8);
      const uint8_t* v1354 = v19 + 152;
      const int8_t* v1355 = (const int8_t*) v1354;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v1356 = __riscv_vle8_v_i8mf2(v1355, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v1357 = __riscv_vsext_vf2_i16m1(v1356, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1358 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1359 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1360 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1361 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1362 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1363 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1364 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1365 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1366 = v21 + 20;
      const int8_t* v1367 = (const int8_t*) v1366;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1368 = *(const int8_t *)(v1367);
      const uint8_t* v1369 = v21 + 52;
      const int8_t* v1370 = (const int8_t*) v1369;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1371 = *(const int8_t *)(v1370);
      const uint8_t* v1372 = v21 + 84;
      const int8_t* v1373 = (const int8_t*) v1372;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1374 = *(const int8_t *)(v1373);
      const uint8_t* v1375 = v21 + 116;
      const int8_t* v1376 = (const int8_t*) v1375;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1377 = *(const int8_t *)(v1376);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1378 = v19 + 1056;
      const uint8_t* v1379 = (const uint8_t*) v1378;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1380 = __riscv_vle8_v_u8mf2(v1379, 8);
      const uint8_t* v1381 = v19 + 544;
      const uint8_t* v1382 = (const uint8_t*) v1381;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1383 = __riscv_vle8_v_u8mf2(v1382, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1384 = __riscv_vand_vx_u8mf2(v1380, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1385 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1384);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1386 = __riscv_vand_vx_u8mf2(v1383, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1387 = __riscv_vmseq_vx_u8mf2_b16(v1386, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1388 = __riscv_vadd_vx_i8mf2_mu(v1387, v1385, v1385, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1389 = __riscv_vwmacc_vx_i16m1(v1358, v1368, v1388, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1390 = __riscv_vsrl_vx_u8mf2(v1380, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1391 = __riscv_vand_vx_u8mf2(v1390, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1392 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1391);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1393 = __riscv_vand_vx_u8mf2(v1383, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1394 = __riscv_vmseq_vx_u8mf2_b16(v1393, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1395 = __riscv_vadd_vx_i8mf2_mu(v1394, v1392, v1392, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1396 = __riscv_vwmacc_vx_i16m1(v1359, v1371, v1395, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1397 = __riscv_vsrl_vx_u8mf2(v1380, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1398 = __riscv_vand_vx_u8mf2(v1397, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1399 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1398);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1400 = __riscv_vand_vx_u8mf2(v1383, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1401 = __riscv_vmseq_vx_u8mf2_b16(v1400, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1402 = __riscv_vadd_vx_i8mf2_mu(v1401, v1399, v1399, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1403 = __riscv_vwmacc_vx_i16m1(v1360, v1374, v1402, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1404 = __riscv_vsrl_vx_u8mf2(v1380, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1405 = __riscv_vand_vx_u8mf2(v1404, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1406 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1405);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1407 = __riscv_vand_vx_u8mf2(v1383, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1408 = __riscv_vmseq_vx_u8mf2_b16(v1407, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1409 = __riscv_vadd_vx_i8mf2_mu(v1408, v1406, v1406, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1410 = __riscv_vwmacc_vx_i16m1(v1361, v1377, v1409, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1411 = v19 + 1064;
      const uint8_t* v1412 = (const uint8_t*) v1411;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1413 = __riscv_vle8_v_u8mf2(v1412, 8);
      const uint8_t* v1414 = v19 + 552;
      const uint8_t* v1415 = (const uint8_t*) v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1416 = __riscv_vle8_v_u8mf2(v1415, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1417 = __riscv_vand_vx_u8mf2(v1413, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1418 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1417);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1419 = __riscv_vand_vx_u8mf2(v1416, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1420 = __riscv_vmseq_vx_u8mf2_b16(v1419, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1421 = __riscv_vadd_vx_i8mf2_mu(v1420, v1418, v1418, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1422 = __riscv_vwmacc_vx_i16m1(v1362, v1368, v1421, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1423 = __riscv_vsrl_vx_u8mf2(v1413, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1424 = __riscv_vand_vx_u8mf2(v1423, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1425 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1424);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1426 = __riscv_vand_vx_u8mf2(v1416, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1427 = __riscv_vmseq_vx_u8mf2_b16(v1426, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1428 = __riscv_vadd_vx_i8mf2_mu(v1427, v1425, v1425, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1429 = __riscv_vwmacc_vx_i16m1(v1363, v1371, v1428, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1430 = __riscv_vsrl_vx_u8mf2(v1413, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1431 = __riscv_vand_vx_u8mf2(v1430, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1432 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1431);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1433 = __riscv_vand_vx_u8mf2(v1416, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1434 = __riscv_vmseq_vx_u8mf2_b16(v1433, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1435 = __riscv_vadd_vx_i8mf2_mu(v1434, v1432, v1432, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1436 = __riscv_vwmacc_vx_i16m1(v1364, v1374, v1435, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1437 = __riscv_vsrl_vx_u8mf2(v1413, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1438 = __riscv_vand_vx_u8mf2(v1437, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1439 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1438);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1440 = __riscv_vand_vx_u8mf2(v1416, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1441 = __riscv_vmseq_vx_u8mf2_b16(v1440, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1442 = __riscv_vadd_vx_i8mf2_mu(v1441, v1439, v1439, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1443 = __riscv_vwmacc_vx_i16m1(v1365, v1377, v1442, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1444 = v21 + 21;
      const int8_t* v1445 = (const int8_t*) v1444;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1446 = *(const int8_t *)(v1445);
      const uint8_t* v1447 = v21 + 53;
      const int8_t* v1448 = (const int8_t*) v1447;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1449 = *(const int8_t *)(v1448);
      const uint8_t* v1450 = v21 + 85;
      const int8_t* v1451 = (const int8_t*) v1450;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1452 = *(const int8_t *)(v1451);
      const uint8_t* v1453 = v21 + 117;
      const int8_t* v1454 = (const int8_t*) v1453;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1455 = *(const int8_t *)(v1454);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1456 = v19 + 1072;
      const uint8_t* v1457 = (const uint8_t*) v1456;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1458 = __riscv_vle8_v_u8mf2(v1457, 8);
      const uint8_t* v1459 = v19 + 560;
      const uint8_t* v1460 = (const uint8_t*) v1459;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1461 = __riscv_vle8_v_u8mf2(v1460, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1462 = __riscv_vand_vx_u8mf2(v1458, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1463 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1462);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1464 = __riscv_vand_vx_u8mf2(v1461, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1465 = __riscv_vmseq_vx_u8mf2_b16(v1464, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1466 = __riscv_vadd_vx_i8mf2_mu(v1465, v1463, v1463, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1467 = __riscv_vwmacc_vx_i16m1(v1389, v1446, v1466, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1468 = __riscv_vsrl_vx_u8mf2(v1458, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1469 = __riscv_vand_vx_u8mf2(v1468, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1470 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1469);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1471 = __riscv_vand_vx_u8mf2(v1461, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1472 = __riscv_vmseq_vx_u8mf2_b16(v1471, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1473 = __riscv_vadd_vx_i8mf2_mu(v1472, v1470, v1470, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1474 = __riscv_vwmacc_vx_i16m1(v1396, v1449, v1473, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1475 = __riscv_vsrl_vx_u8mf2(v1458, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1476 = __riscv_vand_vx_u8mf2(v1475, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1477 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1476);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1478 = __riscv_vand_vx_u8mf2(v1461, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1479 = __riscv_vmseq_vx_u8mf2_b16(v1478, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1480 = __riscv_vadd_vx_i8mf2_mu(v1479, v1477, v1477, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1481 = __riscv_vwmacc_vx_i16m1(v1403, v1452, v1480, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1482 = __riscv_vsrl_vx_u8mf2(v1458, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1483 = __riscv_vand_vx_u8mf2(v1482, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1484 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1483);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1485 = __riscv_vand_vx_u8mf2(v1461, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1486 = __riscv_vmseq_vx_u8mf2_b16(v1485, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1487 = __riscv_vadd_vx_i8mf2_mu(v1486, v1484, v1484, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1488 = __riscv_vwmacc_vx_i16m1(v1410, v1455, v1487, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1489 = v19 + 1080;
      const uint8_t* v1490 = (const uint8_t*) v1489;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1491 = __riscv_vle8_v_u8mf2(v1490, 8);
      const uint8_t* v1492 = v19 + 568;
      const uint8_t* v1493 = (const uint8_t*) v1492;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1494 = __riscv_vle8_v_u8mf2(v1493, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1495 = __riscv_vand_vx_u8mf2(v1491, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1496 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1495);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1497 = __riscv_vand_vx_u8mf2(v1494, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1498 = __riscv_vmseq_vx_u8mf2_b16(v1497, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1499 = __riscv_vadd_vx_i8mf2_mu(v1498, v1496, v1496, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1500 = __riscv_vwmacc_vx_i16m1(v1422, v1446, v1499, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1501 = __riscv_vsrl_vx_u8mf2(v1491, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1502 = __riscv_vand_vx_u8mf2(v1501, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1503 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1502);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1504 = __riscv_vand_vx_u8mf2(v1494, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1505 = __riscv_vmseq_vx_u8mf2_b16(v1504, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1506 = __riscv_vadd_vx_i8mf2_mu(v1505, v1503, v1503, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1507 = __riscv_vwmacc_vx_i16m1(v1429, v1449, v1506, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1508 = __riscv_vsrl_vx_u8mf2(v1491, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1509 = __riscv_vand_vx_u8mf2(v1508, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1510 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1509);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1511 = __riscv_vand_vx_u8mf2(v1494, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1512 = __riscv_vmseq_vx_u8mf2_b16(v1511, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1513 = __riscv_vadd_vx_i8mf2_mu(v1512, v1510, v1510, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1514 = __riscv_vwmacc_vx_i16m1(v1436, v1452, v1513, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1515 = __riscv_vsrl_vx_u8mf2(v1491, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1516 = __riscv_vand_vx_u8mf2(v1515, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1517 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1516);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1518 = __riscv_vand_vx_u8mf2(v1494, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1519 = __riscv_vmseq_vx_u8mf2_b16(v1518, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1520 = __riscv_vadd_vx_i8mf2_mu(v1519, v1517, v1517, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1521 = __riscv_vwmacc_vx_i16m1(v1443, v1455, v1520, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1522 = v21 + 22;
      const int8_t* v1523 = (const int8_t*) v1522;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1524 = *(const int8_t *)(v1523);
      const uint8_t* v1525 = v21 + 54;
      const int8_t* v1526 = (const int8_t*) v1525;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1527 = *(const int8_t *)(v1526);
      const uint8_t* v1528 = v21 + 86;
      const int8_t* v1529 = (const int8_t*) v1528;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1530 = *(const int8_t *)(v1529);
      const uint8_t* v1531 = v21 + 118;
      const int8_t* v1532 = (const int8_t*) v1531;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1533 = *(const int8_t *)(v1532);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1534 = v19 + 1088;
      const uint8_t* v1535 = (const uint8_t*) v1534;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1536 = __riscv_vle8_v_u8mf2(v1535, 8);
      const uint8_t* v1537 = v19 + 576;
      const uint8_t* v1538 = (const uint8_t*) v1537;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1539 = __riscv_vle8_v_u8mf2(v1538, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1540 = __riscv_vand_vx_u8mf2(v1536, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1541 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1540);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1542 = __riscv_vand_vx_u8mf2(v1539, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1543 = __riscv_vmseq_vx_u8mf2_b16(v1542, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1544 = __riscv_vadd_vx_i8mf2_mu(v1543, v1541, v1541, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1545 = __riscv_vwmacc_vx_i16m1(v1467, v1524, v1544, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1546 = __riscv_vsrl_vx_u8mf2(v1536, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1547 = __riscv_vand_vx_u8mf2(v1546, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1548 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1547);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1549 = __riscv_vand_vx_u8mf2(v1539, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1550 = __riscv_vmseq_vx_u8mf2_b16(v1549, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1551 = __riscv_vadd_vx_i8mf2_mu(v1550, v1548, v1548, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1552 = __riscv_vwmacc_vx_i16m1(v1474, v1527, v1551, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1553 = __riscv_vsrl_vx_u8mf2(v1536, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1554 = __riscv_vand_vx_u8mf2(v1553, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1555 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1554);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1556 = __riscv_vand_vx_u8mf2(v1539, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1557 = __riscv_vmseq_vx_u8mf2_b16(v1556, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1558 = __riscv_vadd_vx_i8mf2_mu(v1557, v1555, v1555, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1559 = __riscv_vwmacc_vx_i16m1(v1481, v1530, v1558, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1560 = __riscv_vsrl_vx_u8mf2(v1536, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1561 = __riscv_vand_vx_u8mf2(v1560, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1562 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1561);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1563 = __riscv_vand_vx_u8mf2(v1539, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1564 = __riscv_vmseq_vx_u8mf2_b16(v1563, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1565 = __riscv_vadd_vx_i8mf2_mu(v1564, v1562, v1562, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1566 = __riscv_vwmacc_vx_i16m1(v1488, v1533, v1565, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1567 = v19 + 1096;
      const uint8_t* v1568 = (const uint8_t*) v1567;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1569 = __riscv_vle8_v_u8mf2(v1568, 8);
      const uint8_t* v1570 = v19 + 584;
      const uint8_t* v1571 = (const uint8_t*) v1570;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1572 = __riscv_vle8_v_u8mf2(v1571, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1573 = __riscv_vand_vx_u8mf2(v1569, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1574 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1573);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1575 = __riscv_vand_vx_u8mf2(v1572, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1576 = __riscv_vmseq_vx_u8mf2_b16(v1575, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1577 = __riscv_vadd_vx_i8mf2_mu(v1576, v1574, v1574, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1578 = __riscv_vwmacc_vx_i16m1(v1500, v1524, v1577, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1579 = __riscv_vsrl_vx_u8mf2(v1569, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1580 = __riscv_vand_vx_u8mf2(v1579, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1581 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1580);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1582 = __riscv_vand_vx_u8mf2(v1572, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1583 = __riscv_vmseq_vx_u8mf2_b16(v1582, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1584 = __riscv_vadd_vx_i8mf2_mu(v1583, v1581, v1581, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1585 = __riscv_vwmacc_vx_i16m1(v1507, v1527, v1584, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1586 = __riscv_vsrl_vx_u8mf2(v1569, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1587 = __riscv_vand_vx_u8mf2(v1586, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1588 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1587);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1589 = __riscv_vand_vx_u8mf2(v1572, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1590 = __riscv_vmseq_vx_u8mf2_b16(v1589, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1591 = __riscv_vadd_vx_i8mf2_mu(v1590, v1588, v1588, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1592 = __riscv_vwmacc_vx_i16m1(v1514, v1530, v1591, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1593 = __riscv_vsrl_vx_u8mf2(v1569, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1594 = __riscv_vand_vx_u8mf2(v1593, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1595 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1594);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1596 = __riscv_vand_vx_u8mf2(v1572, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1597 = __riscv_vmseq_vx_u8mf2_b16(v1596, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1598 = __riscv_vadd_vx_i8mf2_mu(v1597, v1595, v1595, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1599 = __riscv_vwmacc_vx_i16m1(v1521, v1533, v1598, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1600 = v21 + 23;
      const int8_t* v1601 = (const int8_t*) v1600;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1602 = *(const int8_t *)(v1601);
      const uint8_t* v1603 = v21 + 55;
      const int8_t* v1604 = (const int8_t*) v1603;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1605 = *(const int8_t *)(v1604);
      const uint8_t* v1606 = v21 + 87;
      const int8_t* v1607 = (const int8_t*) v1606;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1608 = *(const int8_t *)(v1607);
      const uint8_t* v1609 = v21 + 119;
      const int8_t* v1610 = (const int8_t*) v1609;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1611 = *(const int8_t *)(v1610);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1612 = v19 + 1104;
      const uint8_t* v1613 = (const uint8_t*) v1612;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1614 = __riscv_vle8_v_u8mf2(v1613, 8);
      const uint8_t* v1615 = v19 + 592;
      const uint8_t* v1616 = (const uint8_t*) v1615;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1617 = __riscv_vle8_v_u8mf2(v1616, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1618 = __riscv_vand_vx_u8mf2(v1614, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1619 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1618);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1620 = __riscv_vand_vx_u8mf2(v1617, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1621 = __riscv_vmseq_vx_u8mf2_b16(v1620, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1622 = __riscv_vadd_vx_i8mf2_mu(v1621, v1619, v1619, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1623 = __riscv_vwmacc_vx_i16m1(v1545, v1602, v1622, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1624 = __riscv_vsrl_vx_u8mf2(v1614, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1625 = __riscv_vand_vx_u8mf2(v1624, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1626 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1625);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1627 = __riscv_vand_vx_u8mf2(v1617, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1628 = __riscv_vmseq_vx_u8mf2_b16(v1627, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1629 = __riscv_vadd_vx_i8mf2_mu(v1628, v1626, v1626, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1630 = __riscv_vwmacc_vx_i16m1(v1552, v1605, v1629, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1631 = __riscv_vsrl_vx_u8mf2(v1614, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1632 = __riscv_vand_vx_u8mf2(v1631, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1633 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1632);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1634 = __riscv_vand_vx_u8mf2(v1617, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1635 = __riscv_vmseq_vx_u8mf2_b16(v1634, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1636 = __riscv_vadd_vx_i8mf2_mu(v1635, v1633, v1633, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1637 = __riscv_vwmacc_vx_i16m1(v1559, v1608, v1636, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1638 = __riscv_vsrl_vx_u8mf2(v1614, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1639 = __riscv_vand_vx_u8mf2(v1638, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1640 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1639);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1641 = __riscv_vand_vx_u8mf2(v1617, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1642 = __riscv_vmseq_vx_u8mf2_b16(v1641, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1643 = __riscv_vadd_vx_i8mf2_mu(v1642, v1640, v1640, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1644 = __riscv_vwmacc_vx_i16m1(v1566, v1611, v1643, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1645 = v19 + 1112;
      const uint8_t* v1646 = (const uint8_t*) v1645;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1647 = __riscv_vle8_v_u8mf2(v1646, 8);
      const uint8_t* v1648 = v19 + 600;
      const uint8_t* v1649 = (const uint8_t*) v1648;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1650 = __riscv_vle8_v_u8mf2(v1649, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1651 = __riscv_vand_vx_u8mf2(v1647, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1652 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1651);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1653 = __riscv_vand_vx_u8mf2(v1650, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1654 = __riscv_vmseq_vx_u8mf2_b16(v1653, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1655 = __riscv_vadd_vx_i8mf2_mu(v1654, v1652, v1652, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1656 = __riscv_vwmacc_vx_i16m1(v1578, v1602, v1655, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1657 = __riscv_vsrl_vx_u8mf2(v1647, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1658 = __riscv_vand_vx_u8mf2(v1657, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1659 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1658);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1660 = __riscv_vand_vx_u8mf2(v1650, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1661 = __riscv_vmseq_vx_u8mf2_b16(v1660, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1662 = __riscv_vadd_vx_i8mf2_mu(v1661, v1659, v1659, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1663 = __riscv_vwmacc_vx_i16m1(v1585, v1605, v1662, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1664 = __riscv_vsrl_vx_u8mf2(v1647, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1665 = __riscv_vand_vx_u8mf2(v1664, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1666 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1665);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1667 = __riscv_vand_vx_u8mf2(v1650, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1668 = __riscv_vmseq_vx_u8mf2_b16(v1667, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1669 = __riscv_vadd_vx_i8mf2_mu(v1668, v1666, v1666, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1670 = __riscv_vwmacc_vx_i16m1(v1592, v1608, v1669, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1671 = __riscv_vsrl_vx_u8mf2(v1647, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1672 = __riscv_vand_vx_u8mf2(v1671, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1673 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1672);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1674 = __riscv_vand_vx_u8mf2(v1650, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1675 = __riscv_vmseq_vx_u8mf2_b16(v1674, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1676 = __riscv_vadd_vx_i8mf2_mu(v1675, v1673, v1673, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1677 = __riscv_vwmacc_vx_i16m1(v1599, v1611, v1676, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1678 = v21 + 24;
      const int8_t* v1679 = (const int8_t*) v1678;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1680 = *(const int8_t *)(v1679);
      const uint8_t* v1681 = v21 + 56;
      const int8_t* v1682 = (const int8_t*) v1681;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1683 = *(const int8_t *)(v1682);
      const uint8_t* v1684 = v21 + 88;
      const int8_t* v1685 = (const int8_t*) v1684;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1686 = *(const int8_t *)(v1685);
      const uint8_t* v1687 = v21 + 120;
      const int8_t* v1688 = (const int8_t*) v1687;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1689 = *(const int8_t *)(v1688);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1690 = v19 + 1120;
      const uint8_t* v1691 = (const uint8_t*) v1690;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1692 = __riscv_vle8_v_u8mf2(v1691, 8);
      const uint8_t* v1693 = v19 + 608;
      const uint8_t* v1694 = (const uint8_t*) v1693;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1695 = __riscv_vle8_v_u8mf2(v1694, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1696 = __riscv_vand_vx_u8mf2(v1692, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1697 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1696);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1698 = __riscv_vand_vx_u8mf2(v1695, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1699 = __riscv_vmseq_vx_u8mf2_b16(v1698, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1700 = __riscv_vadd_vx_i8mf2_mu(v1699, v1697, v1697, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1701 = __riscv_vwmacc_vx_i16m1(v1623, v1680, v1700, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1702 = __riscv_vsrl_vx_u8mf2(v1692, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1703 = __riscv_vand_vx_u8mf2(v1702, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1704 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1703);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1705 = __riscv_vand_vx_u8mf2(v1695, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1706 = __riscv_vmseq_vx_u8mf2_b16(v1705, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1707 = __riscv_vadd_vx_i8mf2_mu(v1706, v1704, v1704, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1708 = __riscv_vwmacc_vx_i16m1(v1630, v1683, v1707, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1709 = __riscv_vsrl_vx_u8mf2(v1692, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1710 = __riscv_vand_vx_u8mf2(v1709, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1711 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1710);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1712 = __riscv_vand_vx_u8mf2(v1695, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1713 = __riscv_vmseq_vx_u8mf2_b16(v1712, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1714 = __riscv_vadd_vx_i8mf2_mu(v1713, v1711, v1711, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1715 = __riscv_vwmacc_vx_i16m1(v1637, v1686, v1714, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1716 = __riscv_vsrl_vx_u8mf2(v1692, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1717 = __riscv_vand_vx_u8mf2(v1716, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1718 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1717);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1719 = __riscv_vand_vx_u8mf2(v1695, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1720 = __riscv_vmseq_vx_u8mf2_b16(v1719, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1721 = __riscv_vadd_vx_i8mf2_mu(v1720, v1718, v1718, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1722 = __riscv_vwmacc_vx_i16m1(v1644, v1689, v1721, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1723 = v19 + 1128;
      const uint8_t* v1724 = (const uint8_t*) v1723;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1725 = __riscv_vle8_v_u8mf2(v1724, 8);
      const uint8_t* v1726 = v19 + 616;
      const uint8_t* v1727 = (const uint8_t*) v1726;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1728 = __riscv_vle8_v_u8mf2(v1727, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1729 = __riscv_vand_vx_u8mf2(v1725, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1730 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1729);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1731 = __riscv_vand_vx_u8mf2(v1728, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1732 = __riscv_vmseq_vx_u8mf2_b16(v1731, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1733 = __riscv_vadd_vx_i8mf2_mu(v1732, v1730, v1730, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1734 = __riscv_vwmacc_vx_i16m1(v1656, v1680, v1733, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1735 = __riscv_vsrl_vx_u8mf2(v1725, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1736 = __riscv_vand_vx_u8mf2(v1735, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1737 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1736);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1738 = __riscv_vand_vx_u8mf2(v1728, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1739 = __riscv_vmseq_vx_u8mf2_b16(v1738, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1740 = __riscv_vadd_vx_i8mf2_mu(v1739, v1737, v1737, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1741 = __riscv_vwmacc_vx_i16m1(v1663, v1683, v1740, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1742 = __riscv_vsrl_vx_u8mf2(v1725, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1743 = __riscv_vand_vx_u8mf2(v1742, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1744 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1743);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1745 = __riscv_vand_vx_u8mf2(v1728, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1746 = __riscv_vmseq_vx_u8mf2_b16(v1745, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1747 = __riscv_vadd_vx_i8mf2_mu(v1746, v1744, v1744, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1748 = __riscv_vwmacc_vx_i16m1(v1670, v1686, v1747, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1749 = __riscv_vsrl_vx_u8mf2(v1725, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1750 = __riscv_vand_vx_u8mf2(v1749, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1751 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1750);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1752 = __riscv_vand_vx_u8mf2(v1728, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1753 = __riscv_vmseq_vx_u8mf2_b16(v1752, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1754 = __riscv_vadd_vx_i8mf2_mu(v1753, v1751, v1751, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1755 = __riscv_vwmacc_vx_i16m1(v1677, v1689, v1754, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1756 = v21 + 25;
      const int8_t* v1757 = (const int8_t*) v1756;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1758 = *(const int8_t *)(v1757);
      const uint8_t* v1759 = v21 + 57;
      const int8_t* v1760 = (const int8_t*) v1759;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1761 = *(const int8_t *)(v1760);
      const uint8_t* v1762 = v21 + 89;
      const int8_t* v1763 = (const int8_t*) v1762;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1764 = *(const int8_t *)(v1763);
      const uint8_t* v1765 = v21 + 121;
      const int8_t* v1766 = (const int8_t*) v1765;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1767 = *(const int8_t *)(v1766);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1768 = v19 + 1136;
      const uint8_t* v1769 = (const uint8_t*) v1768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1770 = __riscv_vle8_v_u8mf2(v1769, 8);
      const uint8_t* v1771 = v19 + 624;
      const uint8_t* v1772 = (const uint8_t*) v1771;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1773 = __riscv_vle8_v_u8mf2(v1772, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1774 = __riscv_vand_vx_u8mf2(v1770, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1775 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1774);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1776 = __riscv_vand_vx_u8mf2(v1773, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1777 = __riscv_vmseq_vx_u8mf2_b16(v1776, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1778 = __riscv_vadd_vx_i8mf2_mu(v1777, v1775, v1775, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1779 = __riscv_vwmacc_vx_i16m1(v1701, v1758, v1778, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1780 = __riscv_vsrl_vx_u8mf2(v1770, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1781 = __riscv_vand_vx_u8mf2(v1780, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1782 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1781);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1783 = __riscv_vand_vx_u8mf2(v1773, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1784 = __riscv_vmseq_vx_u8mf2_b16(v1783, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1785 = __riscv_vadd_vx_i8mf2_mu(v1784, v1782, v1782, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1786 = __riscv_vwmacc_vx_i16m1(v1708, v1761, v1785, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1787 = __riscv_vsrl_vx_u8mf2(v1770, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1788 = __riscv_vand_vx_u8mf2(v1787, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1789 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1788);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1790 = __riscv_vand_vx_u8mf2(v1773, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1791 = __riscv_vmseq_vx_u8mf2_b16(v1790, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1792 = __riscv_vadd_vx_i8mf2_mu(v1791, v1789, v1789, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1793 = __riscv_vwmacc_vx_i16m1(v1715, v1764, v1792, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1794 = __riscv_vsrl_vx_u8mf2(v1770, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1795 = __riscv_vand_vx_u8mf2(v1794, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1796 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1795);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1797 = __riscv_vand_vx_u8mf2(v1773, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1798 = __riscv_vmseq_vx_u8mf2_b16(v1797, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1799 = __riscv_vadd_vx_i8mf2_mu(v1798, v1796, v1796, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1800 = __riscv_vwmacc_vx_i16m1(v1722, v1767, v1799, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1801 = v19 + 1144;
      const uint8_t* v1802 = (const uint8_t*) v1801;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1803 = __riscv_vle8_v_u8mf2(v1802, 8);
      const uint8_t* v1804 = v19 + 632;
      const uint8_t* v1805 = (const uint8_t*) v1804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1806 = __riscv_vle8_v_u8mf2(v1805, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1807 = __riscv_vand_vx_u8mf2(v1803, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1808 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1807);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1809 = __riscv_vand_vx_u8mf2(v1806, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1810 = __riscv_vmseq_vx_u8mf2_b16(v1809, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1811 = __riscv_vadd_vx_i8mf2_mu(v1810, v1808, v1808, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1812 = __riscv_vwmacc_vx_i16m1(v1734, v1758, v1811, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1813 = __riscv_vsrl_vx_u8mf2(v1803, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1814 = __riscv_vand_vx_u8mf2(v1813, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1815 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1814);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1816 = __riscv_vand_vx_u8mf2(v1806, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1817 = __riscv_vmseq_vx_u8mf2_b16(v1816, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1818 = __riscv_vadd_vx_i8mf2_mu(v1817, v1815, v1815, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1819 = __riscv_vwmacc_vx_i16m1(v1741, v1761, v1818, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1820 = __riscv_vsrl_vx_u8mf2(v1803, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1821 = __riscv_vand_vx_u8mf2(v1820, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1822 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1821);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1823 = __riscv_vand_vx_u8mf2(v1806, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1824 = __riscv_vmseq_vx_u8mf2_b16(v1823, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1825 = __riscv_vadd_vx_i8mf2_mu(v1824, v1822, v1822, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1826 = __riscv_vwmacc_vx_i16m1(v1748, v1764, v1825, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1827 = __riscv_vsrl_vx_u8mf2(v1803, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1828 = __riscv_vand_vx_u8mf2(v1827, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1829 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1828);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1830 = __riscv_vand_vx_u8mf2(v1806, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1831 = __riscv_vmseq_vx_u8mf2_b16(v1830, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1832 = __riscv_vadd_vx_i8mf2_mu(v1831, v1829, v1829, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1833 = __riscv_vwmacc_vx_i16m1(v1755, v1767, v1832, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1834 = v21 + 26;
      const int8_t* v1835 = (const int8_t*) v1834;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1836 = *(const int8_t *)(v1835);
      const uint8_t* v1837 = v21 + 58;
      const int8_t* v1838 = (const int8_t*) v1837;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1839 = *(const int8_t *)(v1838);
      const uint8_t* v1840 = v21 + 90;
      const int8_t* v1841 = (const int8_t*) v1840;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1842 = *(const int8_t *)(v1841);
      const uint8_t* v1843 = v21 + 122;
      const int8_t* v1844 = (const int8_t*) v1843;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1845 = *(const int8_t *)(v1844);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1846 = v19 + 1152;
      const uint8_t* v1847 = (const uint8_t*) v1846;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1848 = __riscv_vle8_v_u8mf2(v1847, 8);
      const uint8_t* v1849 = v19 + 640;
      const uint8_t* v1850 = (const uint8_t*) v1849;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1851 = __riscv_vle8_v_u8mf2(v1850, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1852 = __riscv_vand_vx_u8mf2(v1848, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1853 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1852);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1854 = __riscv_vand_vx_u8mf2(v1851, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1855 = __riscv_vmseq_vx_u8mf2_b16(v1854, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1856 = __riscv_vadd_vx_i8mf2_mu(v1855, v1853, v1853, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1857 = __riscv_vwmacc_vx_i16m1(v1779, v1836, v1856, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1858 = __riscv_vsrl_vx_u8mf2(v1848, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1859 = __riscv_vand_vx_u8mf2(v1858, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1860 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1859);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1861 = __riscv_vand_vx_u8mf2(v1851, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1862 = __riscv_vmseq_vx_u8mf2_b16(v1861, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1863 = __riscv_vadd_vx_i8mf2_mu(v1862, v1860, v1860, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1864 = __riscv_vwmacc_vx_i16m1(v1786, v1839, v1863, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1865 = __riscv_vsrl_vx_u8mf2(v1848, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1866 = __riscv_vand_vx_u8mf2(v1865, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1867 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1866);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1868 = __riscv_vand_vx_u8mf2(v1851, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1869 = __riscv_vmseq_vx_u8mf2_b16(v1868, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1870 = __riscv_vadd_vx_i8mf2_mu(v1869, v1867, v1867, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1871 = __riscv_vwmacc_vx_i16m1(v1793, v1842, v1870, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1872 = __riscv_vsrl_vx_u8mf2(v1848, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1873 = __riscv_vand_vx_u8mf2(v1872, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1874 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1873);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1875 = __riscv_vand_vx_u8mf2(v1851, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1876 = __riscv_vmseq_vx_u8mf2_b16(v1875, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1877 = __riscv_vadd_vx_i8mf2_mu(v1876, v1874, v1874, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1878 = __riscv_vwmacc_vx_i16m1(v1800, v1845, v1877, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1879 = v19 + 1160;
      const uint8_t* v1880 = (const uint8_t*) v1879;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1881 = __riscv_vle8_v_u8mf2(v1880, 8);
      const uint8_t* v1882 = v19 + 648;
      const uint8_t* v1883 = (const uint8_t*) v1882;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1884 = __riscv_vle8_v_u8mf2(v1883, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1885 = __riscv_vand_vx_u8mf2(v1881, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1886 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1885);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1887 = __riscv_vand_vx_u8mf2(v1884, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1888 = __riscv_vmseq_vx_u8mf2_b16(v1887, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1889 = __riscv_vadd_vx_i8mf2_mu(v1888, v1886, v1886, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1890 = __riscv_vwmacc_vx_i16m1(v1812, v1836, v1889, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1891 = __riscv_vsrl_vx_u8mf2(v1881, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1892 = __riscv_vand_vx_u8mf2(v1891, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1893 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1892);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1894 = __riscv_vand_vx_u8mf2(v1884, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1895 = __riscv_vmseq_vx_u8mf2_b16(v1894, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1896 = __riscv_vadd_vx_i8mf2_mu(v1895, v1893, v1893, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1897 = __riscv_vwmacc_vx_i16m1(v1819, v1839, v1896, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1898 = __riscv_vsrl_vx_u8mf2(v1881, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1899 = __riscv_vand_vx_u8mf2(v1898, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1900 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1899);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1901 = __riscv_vand_vx_u8mf2(v1884, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1902 = __riscv_vmseq_vx_u8mf2_b16(v1901, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1903 = __riscv_vadd_vx_i8mf2_mu(v1902, v1900, v1900, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1904 = __riscv_vwmacc_vx_i16m1(v1826, v1842, v1903, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1905 = __riscv_vsrl_vx_u8mf2(v1881, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1906 = __riscv_vand_vx_u8mf2(v1905, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1907 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1906);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1908 = __riscv_vand_vx_u8mf2(v1884, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1909 = __riscv_vmseq_vx_u8mf2_b16(v1908, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1910 = __riscv_vadd_vx_i8mf2_mu(v1909, v1907, v1907, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1911 = __riscv_vwmacc_vx_i16m1(v1833, v1845, v1910, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1912 = v21 + 27;
      const int8_t* v1913 = (const int8_t*) v1912;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1914 = *(const int8_t *)(v1913);
      const uint8_t* v1915 = v21 + 59;
      const int8_t* v1916 = (const int8_t*) v1915;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1917 = *(const int8_t *)(v1916);
      const uint8_t* v1918 = v21 + 91;
      const int8_t* v1919 = (const int8_t*) v1918;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1920 = *(const int8_t *)(v1919);
      const uint8_t* v1921 = v21 + 123;
      const int8_t* v1922 = (const int8_t*) v1921;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1923 = *(const int8_t *)(v1922);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1924 = v19 + 1168;
      const uint8_t* v1925 = (const uint8_t*) v1924;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1926 = __riscv_vle8_v_u8mf2(v1925, 8);
      const uint8_t* v1927 = v19 + 656;
      const uint8_t* v1928 = (const uint8_t*) v1927;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1929 = __riscv_vle8_v_u8mf2(v1928, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1930 = __riscv_vand_vx_u8mf2(v1926, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1931 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1930);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1932 = __riscv_vand_vx_u8mf2(v1929, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1933 = __riscv_vmseq_vx_u8mf2_b16(v1932, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1934 = __riscv_vadd_vx_i8mf2_mu(v1933, v1931, v1931, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1935 = __riscv_vwmacc_vx_i16m1(v1857, v1914, v1934, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1936 = __riscv_vsrl_vx_u8mf2(v1926, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1937 = __riscv_vand_vx_u8mf2(v1936, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1938 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1937);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1939 = __riscv_vand_vx_u8mf2(v1929, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1940 = __riscv_vmseq_vx_u8mf2_b16(v1939, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1941 = __riscv_vadd_vx_i8mf2_mu(v1940, v1938, v1938, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1942 = __riscv_vwmacc_vx_i16m1(v1864, v1917, v1941, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1943 = __riscv_vsrl_vx_u8mf2(v1926, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1944 = __riscv_vand_vx_u8mf2(v1943, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1945 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1944);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1946 = __riscv_vand_vx_u8mf2(v1929, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1947 = __riscv_vmseq_vx_u8mf2_b16(v1946, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1948 = __riscv_vadd_vx_i8mf2_mu(v1947, v1945, v1945, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1949 = __riscv_vwmacc_vx_i16m1(v1871, v1920, v1948, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1950 = __riscv_vsrl_vx_u8mf2(v1926, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1951 = __riscv_vand_vx_u8mf2(v1950, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1952 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1951);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1953 = __riscv_vand_vx_u8mf2(v1929, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1954 = __riscv_vmseq_vx_u8mf2_b16(v1953, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1955 = __riscv_vadd_vx_i8mf2_mu(v1954, v1952, v1952, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1956 = __riscv_vwmacc_vx_i16m1(v1878, v1923, v1955, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v1957 = v19 + 1176;
      const uint8_t* v1958 = (const uint8_t*) v1957;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1959 = __riscv_vle8_v_u8mf2(v1958, 8);
      const uint8_t* v1960 = v19 + 664;
      const uint8_t* v1961 = (const uint8_t*) v1960;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1962 = __riscv_vle8_v_u8mf2(v1961, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1963 = __riscv_vand_vx_u8mf2(v1959, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1964 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1963);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1965 = __riscv_vand_vx_u8mf2(v1962, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1966 = __riscv_vmseq_vx_u8mf2_b16(v1965, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1967 = __riscv_vadd_vx_i8mf2_mu(v1966, v1964, v1964, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1968 = __riscv_vwmacc_vx_i16m1(v1890, v1914, v1967, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1969 = __riscv_vsrl_vx_u8mf2(v1959, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1970 = __riscv_vand_vx_u8mf2(v1969, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1971 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1970);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1972 = __riscv_vand_vx_u8mf2(v1962, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1973 = __riscv_vmseq_vx_u8mf2_b16(v1972, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1974 = __riscv_vadd_vx_i8mf2_mu(v1973, v1971, v1971, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1975 = __riscv_vwmacc_vx_i16m1(v1897, v1917, v1974, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1976 = __riscv_vsrl_vx_u8mf2(v1959, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1977 = __riscv_vand_vx_u8mf2(v1976, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1978 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1977);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1979 = __riscv_vand_vx_u8mf2(v1962, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1980 = __riscv_vmseq_vx_u8mf2_b16(v1979, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1981 = __riscv_vadd_vx_i8mf2_mu(v1980, v1978, v1978, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1982 = __riscv_vwmacc_vx_i16m1(v1904, v1920, v1981, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1983 = __riscv_vsrl_vx_u8mf2(v1959, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1984 = __riscv_vand_vx_u8mf2(v1983, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1985 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1984);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1986 = __riscv_vand_vx_u8mf2(v1962, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v1987 = __riscv_vmseq_vx_u8mf2_b16(v1986, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v1988 = __riscv_vadd_vx_i8mf2_mu(v1987, v1985, v1985, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1989 = __riscv_vwmacc_vx_i16m1(v1911, v1923, v1988, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1990 = v21 + 28;
      const int8_t* v1991 = (const int8_t*) v1990;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1992 = *(const int8_t *)(v1991);
      const uint8_t* v1993 = v21 + 60;
      const int8_t* v1994 = (const int8_t*) v1993;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1995 = *(const int8_t *)(v1994);
      const uint8_t* v1996 = v21 + 92;
      const int8_t* v1997 = (const int8_t*) v1996;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1998 = *(const int8_t *)(v1997);
      const uint8_t* v1999 = v21 + 124;
      const int8_t* v2000 = (const int8_t*) v1999;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2001 = *(const int8_t *)(v2000);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2002 = v19 + 1184;
      const uint8_t* v2003 = (const uint8_t*) v2002;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2004 = __riscv_vle8_v_u8mf2(v2003, 8);
      const uint8_t* v2005 = v19 + 672;
      const uint8_t* v2006 = (const uint8_t*) v2005;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2007 = __riscv_vle8_v_u8mf2(v2006, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2008 = __riscv_vand_vx_u8mf2(v2004, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2009 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2008);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2010 = __riscv_vand_vx_u8mf2(v2007, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2011 = __riscv_vmseq_vx_u8mf2_b16(v2010, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2012 = __riscv_vadd_vx_i8mf2_mu(v2011, v2009, v2009, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2013 = __riscv_vwmacc_vx_i16m1(v1935, v1992, v2012, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2014 = __riscv_vsrl_vx_u8mf2(v2004, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2015 = __riscv_vand_vx_u8mf2(v2014, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2016 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2015);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2017 = __riscv_vand_vx_u8mf2(v2007, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2018 = __riscv_vmseq_vx_u8mf2_b16(v2017, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2019 = __riscv_vadd_vx_i8mf2_mu(v2018, v2016, v2016, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2020 = __riscv_vwmacc_vx_i16m1(v1942, v1995, v2019, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2021 = __riscv_vsrl_vx_u8mf2(v2004, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2022 = __riscv_vand_vx_u8mf2(v2021, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2023 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2022);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2024 = __riscv_vand_vx_u8mf2(v2007, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2025 = __riscv_vmseq_vx_u8mf2_b16(v2024, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2026 = __riscv_vadd_vx_i8mf2_mu(v2025, v2023, v2023, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2027 = __riscv_vwmacc_vx_i16m1(v1949, v1998, v2026, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2028 = __riscv_vsrl_vx_u8mf2(v2004, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2029 = __riscv_vand_vx_u8mf2(v2028, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2030 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2029);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2031 = __riscv_vand_vx_u8mf2(v2007, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2032 = __riscv_vmseq_vx_u8mf2_b16(v2031, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2033 = __riscv_vadd_vx_i8mf2_mu(v2032, v2030, v2030, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2034 = __riscv_vwmacc_vx_i16m1(v1956, v2001, v2033, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2035 = v19 + 1192;
      const uint8_t* v2036 = (const uint8_t*) v2035;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2037 = __riscv_vle8_v_u8mf2(v2036, 8);
      const uint8_t* v2038 = v19 + 680;
      const uint8_t* v2039 = (const uint8_t*) v2038;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2040 = __riscv_vle8_v_u8mf2(v2039, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2041 = __riscv_vand_vx_u8mf2(v2037, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2042 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2041);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2043 = __riscv_vand_vx_u8mf2(v2040, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2044 = __riscv_vmseq_vx_u8mf2_b16(v2043, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2045 = __riscv_vadd_vx_i8mf2_mu(v2044, v2042, v2042, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2046 = __riscv_vwmacc_vx_i16m1(v1968, v1992, v2045, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2047 = __riscv_vsrl_vx_u8mf2(v2037, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2048 = __riscv_vand_vx_u8mf2(v2047, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2049 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2048);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2050 = __riscv_vand_vx_u8mf2(v2040, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2051 = __riscv_vmseq_vx_u8mf2_b16(v2050, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2052 = __riscv_vadd_vx_i8mf2_mu(v2051, v2049, v2049, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2053 = __riscv_vwmacc_vx_i16m1(v1975, v1995, v2052, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2054 = __riscv_vsrl_vx_u8mf2(v2037, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2055 = __riscv_vand_vx_u8mf2(v2054, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2056 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2055);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2057 = __riscv_vand_vx_u8mf2(v2040, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2058 = __riscv_vmseq_vx_u8mf2_b16(v2057, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2059 = __riscv_vadd_vx_i8mf2_mu(v2058, v2056, v2056, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2060 = __riscv_vwmacc_vx_i16m1(v1982, v1998, v2059, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2061 = __riscv_vsrl_vx_u8mf2(v2037, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2062 = __riscv_vand_vx_u8mf2(v2061, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2063 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2062);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2064 = __riscv_vand_vx_u8mf2(v2040, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2065 = __riscv_vmseq_vx_u8mf2_b16(v2064, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2066 = __riscv_vadd_vx_i8mf2_mu(v2065, v2063, v2063, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2067 = __riscv_vwmacc_vx_i16m1(v1989, v2001, v2066, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2068 = v21 + 29;
      const int8_t* v2069 = (const int8_t*) v2068;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2070 = *(const int8_t *)(v2069);
      const uint8_t* v2071 = v21 + 61;
      const int8_t* v2072 = (const int8_t*) v2071;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2073 = *(const int8_t *)(v2072);
      const uint8_t* v2074 = v21 + 93;
      const int8_t* v2075 = (const int8_t*) v2074;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2076 = *(const int8_t *)(v2075);
      const uint8_t* v2077 = v21 + 125;
      const int8_t* v2078 = (const int8_t*) v2077;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2079 = *(const int8_t *)(v2078);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2080 = v19 + 1200;
      const uint8_t* v2081 = (const uint8_t*) v2080;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2082 = __riscv_vle8_v_u8mf2(v2081, 8);
      const uint8_t* v2083 = v19 + 688;
      const uint8_t* v2084 = (const uint8_t*) v2083;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2085 = __riscv_vle8_v_u8mf2(v2084, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2086 = __riscv_vand_vx_u8mf2(v2082, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2087 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2086);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2088 = __riscv_vand_vx_u8mf2(v2085, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2089 = __riscv_vmseq_vx_u8mf2_b16(v2088, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2090 = __riscv_vadd_vx_i8mf2_mu(v2089, v2087, v2087, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2091 = __riscv_vwmacc_vx_i16m1(v2013, v2070, v2090, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2092 = __riscv_vsrl_vx_u8mf2(v2082, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2093 = __riscv_vand_vx_u8mf2(v2092, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2094 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2093);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2095 = __riscv_vand_vx_u8mf2(v2085, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2096 = __riscv_vmseq_vx_u8mf2_b16(v2095, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2097 = __riscv_vadd_vx_i8mf2_mu(v2096, v2094, v2094, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2098 = __riscv_vwmacc_vx_i16m1(v2020, v2073, v2097, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2099 = __riscv_vsrl_vx_u8mf2(v2082, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2100 = __riscv_vand_vx_u8mf2(v2099, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2101 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2100);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2102 = __riscv_vand_vx_u8mf2(v2085, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2103 = __riscv_vmseq_vx_u8mf2_b16(v2102, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2104 = __riscv_vadd_vx_i8mf2_mu(v2103, v2101, v2101, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2105 = __riscv_vwmacc_vx_i16m1(v2027, v2076, v2104, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2106 = __riscv_vsrl_vx_u8mf2(v2082, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2107 = __riscv_vand_vx_u8mf2(v2106, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2108 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2107);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2109 = __riscv_vand_vx_u8mf2(v2085, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2110 = __riscv_vmseq_vx_u8mf2_b16(v2109, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2111 = __riscv_vadd_vx_i8mf2_mu(v2110, v2108, v2108, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2112 = __riscv_vwmacc_vx_i16m1(v2034, v2079, v2111, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2113 = v19 + 1208;
      const uint8_t* v2114 = (const uint8_t*) v2113;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2115 = __riscv_vle8_v_u8mf2(v2114, 8);
      const uint8_t* v2116 = v19 + 696;
      const uint8_t* v2117 = (const uint8_t*) v2116;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2118 = __riscv_vle8_v_u8mf2(v2117, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2119 = __riscv_vand_vx_u8mf2(v2115, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2120 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2119);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2121 = __riscv_vand_vx_u8mf2(v2118, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2122 = __riscv_vmseq_vx_u8mf2_b16(v2121, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2123 = __riscv_vadd_vx_i8mf2_mu(v2122, v2120, v2120, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2124 = __riscv_vwmacc_vx_i16m1(v2046, v2070, v2123, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2125 = __riscv_vsrl_vx_u8mf2(v2115, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2126 = __riscv_vand_vx_u8mf2(v2125, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2127 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2126);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2128 = __riscv_vand_vx_u8mf2(v2118, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2129 = __riscv_vmseq_vx_u8mf2_b16(v2128, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2130 = __riscv_vadd_vx_i8mf2_mu(v2129, v2127, v2127, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2131 = __riscv_vwmacc_vx_i16m1(v2053, v2073, v2130, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2132 = __riscv_vsrl_vx_u8mf2(v2115, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2133 = __riscv_vand_vx_u8mf2(v2132, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2134 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2133);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2135 = __riscv_vand_vx_u8mf2(v2118, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2136 = __riscv_vmseq_vx_u8mf2_b16(v2135, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2137 = __riscv_vadd_vx_i8mf2_mu(v2136, v2134, v2134, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2138 = __riscv_vwmacc_vx_i16m1(v2060, v2076, v2137, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2139 = __riscv_vsrl_vx_u8mf2(v2115, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2140 = __riscv_vand_vx_u8mf2(v2139, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2141 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2140);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2142 = __riscv_vand_vx_u8mf2(v2118, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2143 = __riscv_vmseq_vx_u8mf2_b16(v2142, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2144 = __riscv_vadd_vx_i8mf2_mu(v2143, v2141, v2141, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2145 = __riscv_vwmacc_vx_i16m1(v2067, v2079, v2144, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2146 = v21 + 30;
      const int8_t* v2147 = (const int8_t*) v2146;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2148 = *(const int8_t *)(v2147);
      const uint8_t* v2149 = v21 + 62;
      const int8_t* v2150 = (const int8_t*) v2149;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2151 = *(const int8_t *)(v2150);
      const uint8_t* v2152 = v21 + 94;
      const int8_t* v2153 = (const int8_t*) v2152;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2154 = *(const int8_t *)(v2153);
      const uint8_t* v2155 = v21 + 126;
      const int8_t* v2156 = (const int8_t*) v2155;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2157 = *(const int8_t *)(v2156);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2158 = v19 + 1216;
      const uint8_t* v2159 = (const uint8_t*) v2158;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2160 = __riscv_vle8_v_u8mf2(v2159, 8);
      const uint8_t* v2161 = v19 + 704;
      const uint8_t* v2162 = (const uint8_t*) v2161;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2163 = __riscv_vle8_v_u8mf2(v2162, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2164 = __riscv_vand_vx_u8mf2(v2160, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2165 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2164);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2166 = __riscv_vand_vx_u8mf2(v2163, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2167 = __riscv_vmseq_vx_u8mf2_b16(v2166, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2168 = __riscv_vadd_vx_i8mf2_mu(v2167, v2165, v2165, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2169 = __riscv_vwmacc_vx_i16m1(v2091, v2148, v2168, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2170 = __riscv_vsrl_vx_u8mf2(v2160, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2171 = __riscv_vand_vx_u8mf2(v2170, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2172 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2171);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2173 = __riscv_vand_vx_u8mf2(v2163, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2174 = __riscv_vmseq_vx_u8mf2_b16(v2173, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2175 = __riscv_vadd_vx_i8mf2_mu(v2174, v2172, v2172, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2176 = __riscv_vwmacc_vx_i16m1(v2098, v2151, v2175, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2177 = __riscv_vsrl_vx_u8mf2(v2160, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2178 = __riscv_vand_vx_u8mf2(v2177, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2179 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2178);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2180 = __riscv_vand_vx_u8mf2(v2163, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2181 = __riscv_vmseq_vx_u8mf2_b16(v2180, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2182 = __riscv_vadd_vx_i8mf2_mu(v2181, v2179, v2179, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2183 = __riscv_vwmacc_vx_i16m1(v2105, v2154, v2182, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2184 = __riscv_vsrl_vx_u8mf2(v2160, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2185 = __riscv_vand_vx_u8mf2(v2184, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2186 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2185);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2187 = __riscv_vand_vx_u8mf2(v2163, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2188 = __riscv_vmseq_vx_u8mf2_b16(v2187, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2189 = __riscv_vadd_vx_i8mf2_mu(v2188, v2186, v2186, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2190 = __riscv_vwmacc_vx_i16m1(v2112, v2157, v2189, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2191 = v19 + 1224;
      const uint8_t* v2192 = (const uint8_t*) v2191;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2193 = __riscv_vle8_v_u8mf2(v2192, 8);
      const uint8_t* v2194 = v19 + 712;
      const uint8_t* v2195 = (const uint8_t*) v2194;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2196 = __riscv_vle8_v_u8mf2(v2195, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2197 = __riscv_vand_vx_u8mf2(v2193, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2198 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2197);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2199 = __riscv_vand_vx_u8mf2(v2196, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2200 = __riscv_vmseq_vx_u8mf2_b16(v2199, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2201 = __riscv_vadd_vx_i8mf2_mu(v2200, v2198, v2198, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2202 = __riscv_vwmacc_vx_i16m1(v2124, v2148, v2201, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2203 = __riscv_vsrl_vx_u8mf2(v2193, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2204 = __riscv_vand_vx_u8mf2(v2203, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2205 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2204);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2206 = __riscv_vand_vx_u8mf2(v2196, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2207 = __riscv_vmseq_vx_u8mf2_b16(v2206, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2208 = __riscv_vadd_vx_i8mf2_mu(v2207, v2205, v2205, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2209 = __riscv_vwmacc_vx_i16m1(v2131, v2151, v2208, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2210 = __riscv_vsrl_vx_u8mf2(v2193, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2211 = __riscv_vand_vx_u8mf2(v2210, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2212 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2211);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2213 = __riscv_vand_vx_u8mf2(v2196, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2214 = __riscv_vmseq_vx_u8mf2_b16(v2213, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2215 = __riscv_vadd_vx_i8mf2_mu(v2214, v2212, v2212, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2216 = __riscv_vwmacc_vx_i16m1(v2138, v2154, v2215, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2217 = __riscv_vsrl_vx_u8mf2(v2193, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2218 = __riscv_vand_vx_u8mf2(v2217, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2219 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2218);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2220 = __riscv_vand_vx_u8mf2(v2196, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2221 = __riscv_vmseq_vx_u8mf2_b16(v2220, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2222 = __riscv_vadd_vx_i8mf2_mu(v2221, v2219, v2219, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2223 = __riscv_vwmacc_vx_i16m1(v2145, v2157, v2222, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2224 = v21 + 31;
      const int8_t* v2225 = (const int8_t*) v2224;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2226 = *(const int8_t *)(v2225);
      const uint8_t* v2227 = v21 + 63;
      const int8_t* v2228 = (const int8_t*) v2227;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2229 = *(const int8_t *)(v2228);
      const uint8_t* v2230 = v21 + 95;
      const int8_t* v2231 = (const int8_t*) v2230;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2232 = *(const int8_t *)(v2231);
      const uint8_t* v2233 = v21 + 127;
      const int8_t* v2234 = (const int8_t*) v2233;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2235 = *(const int8_t *)(v2234);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2236 = v19 + 1232;
      const uint8_t* v2237 = (const uint8_t*) v2236;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2238 = __riscv_vle8_v_u8mf2(v2237, 8);
      const uint8_t* v2239 = v19 + 720;
      const uint8_t* v2240 = (const uint8_t*) v2239;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2241 = __riscv_vle8_v_u8mf2(v2240, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2242 = __riscv_vand_vx_u8mf2(v2238, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2243 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2242);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2244 = __riscv_vand_vx_u8mf2(v2241, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2245 = __riscv_vmseq_vx_u8mf2_b16(v2244, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2246 = __riscv_vadd_vx_i8mf2_mu(v2245, v2243, v2243, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2247 = __riscv_vwmacc_vx_i16m1(v2169, v2226, v2246, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2248 = __riscv_vsrl_vx_u8mf2(v2238, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2249 = __riscv_vand_vx_u8mf2(v2248, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2250 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2249);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2251 = __riscv_vand_vx_u8mf2(v2241, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2252 = __riscv_vmseq_vx_u8mf2_b16(v2251, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2253 = __riscv_vadd_vx_i8mf2_mu(v2252, v2250, v2250, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2254 = __riscv_vwmacc_vx_i16m1(v2176, v2229, v2253, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2255 = __riscv_vsrl_vx_u8mf2(v2238, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2256 = __riscv_vand_vx_u8mf2(v2255, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2257 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2256);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2258 = __riscv_vand_vx_u8mf2(v2241, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2259 = __riscv_vmseq_vx_u8mf2_b16(v2258, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2260 = __riscv_vadd_vx_i8mf2_mu(v2259, v2257, v2257, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2261 = __riscv_vwmacc_vx_i16m1(v2183, v2232, v2260, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2262 = __riscv_vsrl_vx_u8mf2(v2238, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2263 = __riscv_vand_vx_u8mf2(v2262, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2264 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2263);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2265 = __riscv_vand_vx_u8mf2(v2241, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2266 = __riscv_vmseq_vx_u8mf2_b16(v2265, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2267 = __riscv_vadd_vx_i8mf2_mu(v2266, v2264, v2264, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2268 = __riscv_vwmacc_vx_i16m1(v2190, v2235, v2267, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2269 = v19 + 1240;
      const uint8_t* v2270 = (const uint8_t*) v2269;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2271 = __riscv_vle8_v_u8mf2(v2270, 8);
      const uint8_t* v2272 = v19 + 728;
      const uint8_t* v2273 = (const uint8_t*) v2272;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2274 = __riscv_vle8_v_u8mf2(v2273, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2275 = __riscv_vand_vx_u8mf2(v2271, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2276 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2275);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2277 = __riscv_vand_vx_u8mf2(v2274, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2278 = __riscv_vmseq_vx_u8mf2_b16(v2277, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2279 = __riscv_vadd_vx_i8mf2_mu(v2278, v2276, v2276, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2280 = __riscv_vwmacc_vx_i16m1(v2202, v2226, v2279, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2281 = __riscv_vsrl_vx_u8mf2(v2271, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2282 = __riscv_vand_vx_u8mf2(v2281, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2283 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2282);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2284 = __riscv_vand_vx_u8mf2(v2274, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2285 = __riscv_vmseq_vx_u8mf2_b16(v2284, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2286 = __riscv_vadd_vx_i8mf2_mu(v2285, v2283, v2283, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2287 = __riscv_vwmacc_vx_i16m1(v2209, v2229, v2286, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2288 = __riscv_vsrl_vx_u8mf2(v2271, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2289 = __riscv_vand_vx_u8mf2(v2288, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2290 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2289);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2291 = __riscv_vand_vx_u8mf2(v2274, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2292 = __riscv_vmseq_vx_u8mf2_b16(v2291, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2293 = __riscv_vadd_vx_i8mf2_mu(v2292, v2290, v2290, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2294 = __riscv_vwmacc_vx_i16m1(v2216, v2232, v2293, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2295 = __riscv_vsrl_vx_u8mf2(v2271, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2296 = __riscv_vand_vx_u8mf2(v2295, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2297 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2296);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2298 = __riscv_vand_vx_u8mf2(v2274, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2299 = __riscv_vmseq_vx_u8mf2_b16(v2298, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2300 = __riscv_vadd_vx_i8mf2_mu(v2299, v2297, v2297, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2301 = __riscv_vwmacc_vx_i16m1(v2223, v2235, v2300, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2302 = v21 + 32;
      const int8_t* v2303 = (const int8_t*) v2302;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2304 = *(const int8_t *)(v2303);
      const uint8_t* v2305 = v21 + 64;
      const int8_t* v2306 = (const int8_t*) v2305;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2307 = *(const int8_t *)(v2306);
      const uint8_t* v2308 = v21 + 96;
      const int8_t* v2309 = (const int8_t*) v2308;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2310 = *(const int8_t *)(v2309);
      const uint8_t* v2311 = v21 + 128;
      const int8_t* v2312 = (const int8_t*) v2311;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2313 = *(const int8_t *)(v2312);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2314 = v19 + 1248;
      const uint8_t* v2315 = (const uint8_t*) v2314;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2316 = __riscv_vle8_v_u8mf2(v2315, 8);
      const uint8_t* v2317 = v19 + 736;
      const uint8_t* v2318 = (const uint8_t*) v2317;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2319 = __riscv_vle8_v_u8mf2(v2318, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2320 = __riscv_vand_vx_u8mf2(v2316, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2321 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2320);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2322 = __riscv_vand_vx_u8mf2(v2319, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2323 = __riscv_vmseq_vx_u8mf2_b16(v2322, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2324 = __riscv_vadd_vx_i8mf2_mu(v2323, v2321, v2321, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2325 = __riscv_vwmacc_vx_i16m1(v2247, v2304, v2324, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2326 = __riscv_vsrl_vx_u8mf2(v2316, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2327 = __riscv_vand_vx_u8mf2(v2326, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2328 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2327);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2329 = __riscv_vand_vx_u8mf2(v2319, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2330 = __riscv_vmseq_vx_u8mf2_b16(v2329, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2331 = __riscv_vadd_vx_i8mf2_mu(v2330, v2328, v2328, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2332 = __riscv_vwmacc_vx_i16m1(v2254, v2307, v2331, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2333 = __riscv_vsrl_vx_u8mf2(v2316, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2334 = __riscv_vand_vx_u8mf2(v2333, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2335 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2334);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2336 = __riscv_vand_vx_u8mf2(v2319, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2337 = __riscv_vmseq_vx_u8mf2_b16(v2336, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2338 = __riscv_vadd_vx_i8mf2_mu(v2337, v2335, v2335, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2339 = __riscv_vwmacc_vx_i16m1(v2261, v2310, v2338, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2340 = __riscv_vsrl_vx_u8mf2(v2316, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2341 = __riscv_vand_vx_u8mf2(v2340, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2342 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2341);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2343 = __riscv_vand_vx_u8mf2(v2319, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2344 = __riscv_vmseq_vx_u8mf2_b16(v2343, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2345 = __riscv_vadd_vx_i8mf2_mu(v2344, v2342, v2342, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2346 = __riscv_vwmacc_vx_i16m1(v2268, v2313, v2345, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2347 = v19 + 1256;
      const uint8_t* v2348 = (const uint8_t*) v2347;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2349 = __riscv_vle8_v_u8mf2(v2348, 8);
      const uint8_t* v2350 = v19 + 744;
      const uint8_t* v2351 = (const uint8_t*) v2350;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2352 = __riscv_vle8_v_u8mf2(v2351, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2353 = __riscv_vand_vx_u8mf2(v2349, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2354 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2353);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2355 = __riscv_vand_vx_u8mf2(v2352, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2356 = __riscv_vmseq_vx_u8mf2_b16(v2355, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2357 = __riscv_vadd_vx_i8mf2_mu(v2356, v2354, v2354, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2358 = __riscv_vwmacc_vx_i16m1(v2280, v2304, v2357, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2359 = __riscv_vsrl_vx_u8mf2(v2349, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2360 = __riscv_vand_vx_u8mf2(v2359, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2361 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2360);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2362 = __riscv_vand_vx_u8mf2(v2352, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2363 = __riscv_vmseq_vx_u8mf2_b16(v2362, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2364 = __riscv_vadd_vx_i8mf2_mu(v2363, v2361, v2361, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2365 = __riscv_vwmacc_vx_i16m1(v2287, v2307, v2364, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2366 = __riscv_vsrl_vx_u8mf2(v2349, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2367 = __riscv_vand_vx_u8mf2(v2366, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2368 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2367);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2369 = __riscv_vand_vx_u8mf2(v2352, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2370 = __riscv_vmseq_vx_u8mf2_b16(v2369, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2371 = __riscv_vadd_vx_i8mf2_mu(v2370, v2368, v2368, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2372 = __riscv_vwmacc_vx_i16m1(v2294, v2310, v2371, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2373 = __riscv_vsrl_vx_u8mf2(v2349, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2374 = __riscv_vand_vx_u8mf2(v2373, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2375 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2374);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2376 = __riscv_vand_vx_u8mf2(v2352, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2377 = __riscv_vmseq_vx_u8mf2_b16(v2376, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2378 = __riscv_vadd_vx_i8mf2_mu(v2377, v2375, v2375, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2379 = __riscv_vwmacc_vx_i16m1(v2301, v2313, v2378, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2380 = v21 + 33;
      const int8_t* v2381 = (const int8_t*) v2380;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2382 = *(const int8_t *)(v2381);
      const uint8_t* v2383 = v21 + 65;
      const int8_t* v2384 = (const int8_t*) v2383;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2385 = *(const int8_t *)(v2384);
      const uint8_t* v2386 = v21 + 97;
      const int8_t* v2387 = (const int8_t*) v2386;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2388 = *(const int8_t *)(v2387);
      const uint8_t* v2389 = v21 + 129;
      const int8_t* v2390 = (const int8_t*) v2389;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2391 = *(const int8_t *)(v2390);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2392 = v19 + 1264;
      const uint8_t* v2393 = (const uint8_t*) v2392;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2394 = __riscv_vle8_v_u8mf2(v2393, 8);
      const uint8_t* v2395 = v19 + 752;
      const uint8_t* v2396 = (const uint8_t*) v2395;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2397 = __riscv_vle8_v_u8mf2(v2396, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2398 = __riscv_vand_vx_u8mf2(v2394, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2399 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2398);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2400 = __riscv_vand_vx_u8mf2(v2397, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2401 = __riscv_vmseq_vx_u8mf2_b16(v2400, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2402 = __riscv_vadd_vx_i8mf2_mu(v2401, v2399, v2399, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2403 = __riscv_vwmacc_vx_i16m1(v2325, v2382, v2402, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2404 = __riscv_vsrl_vx_u8mf2(v2394, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2405 = __riscv_vand_vx_u8mf2(v2404, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2406 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2405);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2407 = __riscv_vand_vx_u8mf2(v2397, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2408 = __riscv_vmseq_vx_u8mf2_b16(v2407, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2409 = __riscv_vadd_vx_i8mf2_mu(v2408, v2406, v2406, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2410 = __riscv_vwmacc_vx_i16m1(v2332, v2385, v2409, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2411 = __riscv_vsrl_vx_u8mf2(v2394, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2412 = __riscv_vand_vx_u8mf2(v2411, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2413 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2412);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2414 = __riscv_vand_vx_u8mf2(v2397, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2415 = __riscv_vmseq_vx_u8mf2_b16(v2414, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2416 = __riscv_vadd_vx_i8mf2_mu(v2415, v2413, v2413, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2417 = __riscv_vwmacc_vx_i16m1(v2339, v2388, v2416, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2418 = __riscv_vsrl_vx_u8mf2(v2394, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2419 = __riscv_vand_vx_u8mf2(v2418, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2420 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2419);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2421 = __riscv_vand_vx_u8mf2(v2397, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2422 = __riscv_vmseq_vx_u8mf2_b16(v2421, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2423 = __riscv_vadd_vx_i8mf2_mu(v2422, v2420, v2420, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2424 = __riscv_vwmacc_vx_i16m1(v2346, v2391, v2423, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2425 = v19 + 1272;
      const uint8_t* v2426 = (const uint8_t*) v2425;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2427 = __riscv_vle8_v_u8mf2(v2426, 8);
      const uint8_t* v2428 = v19 + 760;
      const uint8_t* v2429 = (const uint8_t*) v2428;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2430 = __riscv_vle8_v_u8mf2(v2429, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2431 = __riscv_vand_vx_u8mf2(v2427, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2432 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2431);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2433 = __riscv_vand_vx_u8mf2(v2430, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2434 = __riscv_vmseq_vx_u8mf2_b16(v2433, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2435 = __riscv_vadd_vx_i8mf2_mu(v2434, v2432, v2432, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2436 = __riscv_vwmacc_vx_i16m1(v2358, v2382, v2435, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2437 = __riscv_vsrl_vx_u8mf2(v2427, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2438 = __riscv_vand_vx_u8mf2(v2437, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2439 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2438);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2440 = __riscv_vand_vx_u8mf2(v2430, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2441 = __riscv_vmseq_vx_u8mf2_b16(v2440, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2442 = __riscv_vadd_vx_i8mf2_mu(v2441, v2439, v2439, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2443 = __riscv_vwmacc_vx_i16m1(v2365, v2385, v2442, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2444 = __riscv_vsrl_vx_u8mf2(v2427, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2445 = __riscv_vand_vx_u8mf2(v2444, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2446 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2445);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2447 = __riscv_vand_vx_u8mf2(v2430, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2448 = __riscv_vmseq_vx_u8mf2_b16(v2447, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2449 = __riscv_vadd_vx_i8mf2_mu(v2448, v2446, v2446, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2450 = __riscv_vwmacc_vx_i16m1(v2372, v2388, v2449, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2451 = __riscv_vsrl_vx_u8mf2(v2427, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2452 = __riscv_vand_vx_u8mf2(v2451, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2453 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2452);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2454 = __riscv_vand_vx_u8mf2(v2430, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2455 = __riscv_vmseq_vx_u8mf2_b16(v2454, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2456 = __riscv_vadd_vx_i8mf2_mu(v2455, v2453, v2453, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2457 = __riscv_vwmacc_vx_i16m1(v2379, v2391, v2456, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2458 = v21 + 34;
      const int8_t* v2459 = (const int8_t*) v2458;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2460 = *(const int8_t *)(v2459);
      const uint8_t* v2461 = v21 + 66;
      const int8_t* v2462 = (const int8_t*) v2461;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2463 = *(const int8_t *)(v2462);
      const uint8_t* v2464 = v21 + 98;
      const int8_t* v2465 = (const int8_t*) v2464;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2466 = *(const int8_t *)(v2465);
      const uint8_t* v2467 = v21 + 130;
      const int8_t* v2468 = (const int8_t*) v2467;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2469 = *(const int8_t *)(v2468);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2470 = v19 + 1280;
      const uint8_t* v2471 = (const uint8_t*) v2470;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2472 = __riscv_vle8_v_u8mf2(v2471, 8);
      const uint8_t* v2473 = v19 + 768;
      const uint8_t* v2474 = (const uint8_t*) v2473;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2475 = __riscv_vle8_v_u8mf2(v2474, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2476 = __riscv_vand_vx_u8mf2(v2472, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2477 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2476);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2478 = __riscv_vand_vx_u8mf2(v2475, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2479 = __riscv_vmseq_vx_u8mf2_b16(v2478, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2480 = __riscv_vadd_vx_i8mf2_mu(v2479, v2477, v2477, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2481 = __riscv_vwmacc_vx_i16m1(v2403, v2460, v2480, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2482 = __riscv_vsrl_vx_u8mf2(v2472, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2483 = __riscv_vand_vx_u8mf2(v2482, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2484 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2483);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2485 = __riscv_vand_vx_u8mf2(v2475, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2486 = __riscv_vmseq_vx_u8mf2_b16(v2485, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2487 = __riscv_vadd_vx_i8mf2_mu(v2486, v2484, v2484, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2488 = __riscv_vwmacc_vx_i16m1(v2410, v2463, v2487, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2489 = __riscv_vsrl_vx_u8mf2(v2472, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2490 = __riscv_vand_vx_u8mf2(v2489, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2491 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2490);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2492 = __riscv_vand_vx_u8mf2(v2475, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2493 = __riscv_vmseq_vx_u8mf2_b16(v2492, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2494 = __riscv_vadd_vx_i8mf2_mu(v2493, v2491, v2491, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2495 = __riscv_vwmacc_vx_i16m1(v2417, v2466, v2494, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2496 = __riscv_vsrl_vx_u8mf2(v2472, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2497 = __riscv_vand_vx_u8mf2(v2496, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2498 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2497);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2499 = __riscv_vand_vx_u8mf2(v2475, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2500 = __riscv_vmseq_vx_u8mf2_b16(v2499, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2501 = __riscv_vadd_vx_i8mf2_mu(v2500, v2498, v2498, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2502 = __riscv_vwmacc_vx_i16m1(v2424, v2469, v2501, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2503 = v19 + 1288;
      const uint8_t* v2504 = (const uint8_t*) v2503;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2505 = __riscv_vle8_v_u8mf2(v2504, 8);
      const uint8_t* v2506 = v19 + 776;
      const uint8_t* v2507 = (const uint8_t*) v2506;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2508 = __riscv_vle8_v_u8mf2(v2507, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2509 = __riscv_vand_vx_u8mf2(v2505, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2510 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2509);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2511 = __riscv_vand_vx_u8mf2(v2508, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2512 = __riscv_vmseq_vx_u8mf2_b16(v2511, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2513 = __riscv_vadd_vx_i8mf2_mu(v2512, v2510, v2510, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2514 = __riscv_vwmacc_vx_i16m1(v2436, v2460, v2513, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2515 = __riscv_vsrl_vx_u8mf2(v2505, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2516 = __riscv_vand_vx_u8mf2(v2515, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2517 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2516);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2518 = __riscv_vand_vx_u8mf2(v2508, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2519 = __riscv_vmseq_vx_u8mf2_b16(v2518, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2520 = __riscv_vadd_vx_i8mf2_mu(v2519, v2517, v2517, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2521 = __riscv_vwmacc_vx_i16m1(v2443, v2463, v2520, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2522 = __riscv_vsrl_vx_u8mf2(v2505, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2523 = __riscv_vand_vx_u8mf2(v2522, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2524 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2523);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2525 = __riscv_vand_vx_u8mf2(v2508, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2526 = __riscv_vmseq_vx_u8mf2_b16(v2525, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2527 = __riscv_vadd_vx_i8mf2_mu(v2526, v2524, v2524, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2528 = __riscv_vwmacc_vx_i16m1(v2450, v2466, v2527, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2529 = __riscv_vsrl_vx_u8mf2(v2505, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2530 = __riscv_vand_vx_u8mf2(v2529, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2531 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2530);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2532 = __riscv_vand_vx_u8mf2(v2508, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2533 = __riscv_vmseq_vx_u8mf2_b16(v2532, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2534 = __riscv_vadd_vx_i8mf2_mu(v2533, v2531, v2531, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2535 = __riscv_vwmacc_vx_i16m1(v2457, v2469, v2534, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2536 = v21 + 35;
      const int8_t* v2537 = (const int8_t*) v2536;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2538 = *(const int8_t *)(v2537);
      const uint8_t* v2539 = v21 + 67;
      const int8_t* v2540 = (const int8_t*) v2539;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2541 = *(const int8_t *)(v2540);
      const uint8_t* v2542 = v21 + 99;
      const int8_t* v2543 = (const int8_t*) v2542;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2544 = *(const int8_t *)(v2543);
      const uint8_t* v2545 = v21 + 131;
      const int8_t* v2546 = (const int8_t*) v2545;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2547 = *(const int8_t *)(v2546);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2548 = v19 + 1296;
      const uint8_t* v2549 = (const uint8_t*) v2548;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2550 = __riscv_vle8_v_u8mf2(v2549, 8);
      const uint8_t* v2551 = v19 + 784;
      const uint8_t* v2552 = (const uint8_t*) v2551;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2553 = __riscv_vle8_v_u8mf2(v2552, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2554 = __riscv_vand_vx_u8mf2(v2550, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2555 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2554);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2556 = __riscv_vand_vx_u8mf2(v2553, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2557 = __riscv_vmseq_vx_u8mf2_b16(v2556, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2558 = __riscv_vadd_vx_i8mf2_mu(v2557, v2555, v2555, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2559 = __riscv_vwmacc_vx_i16m1(v2481, v2538, v2558, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2560 = __riscv_vsrl_vx_u8mf2(v2550, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2561 = __riscv_vand_vx_u8mf2(v2560, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2562 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2561);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2563 = __riscv_vand_vx_u8mf2(v2553, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2564 = __riscv_vmseq_vx_u8mf2_b16(v2563, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2565 = __riscv_vadd_vx_i8mf2_mu(v2564, v2562, v2562, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2566 = __riscv_vwmacc_vx_i16m1(v2488, v2541, v2565, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2567 = __riscv_vsrl_vx_u8mf2(v2550, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2568 = __riscv_vand_vx_u8mf2(v2567, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2569 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2568);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2570 = __riscv_vand_vx_u8mf2(v2553, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2571 = __riscv_vmseq_vx_u8mf2_b16(v2570, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2572 = __riscv_vadd_vx_i8mf2_mu(v2571, v2569, v2569, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2573 = __riscv_vwmacc_vx_i16m1(v2495, v2544, v2572, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2574 = __riscv_vsrl_vx_u8mf2(v2550, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2575 = __riscv_vand_vx_u8mf2(v2574, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2576 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2575);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2577 = __riscv_vand_vx_u8mf2(v2553, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2578 = __riscv_vmseq_vx_u8mf2_b16(v2577, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2579 = __riscv_vadd_vx_i8mf2_mu(v2578, v2576, v2576, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2580 = __riscv_vwmacc_vx_i16m1(v2502, v2547, v2579, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2581 = v19 + 1304;
      const uint8_t* v2582 = (const uint8_t*) v2581;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2583 = __riscv_vle8_v_u8mf2(v2582, 8);
      const uint8_t* v2584 = v19 + 792;
      const uint8_t* v2585 = (const uint8_t*) v2584;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2586 = __riscv_vle8_v_u8mf2(v2585, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2587 = __riscv_vand_vx_u8mf2(v2583, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2588 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2587);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2589 = __riscv_vand_vx_u8mf2(v2586, 1, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2590 = __riscv_vmseq_vx_u8mf2_b16(v2589, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2591 = __riscv_vadd_vx_i8mf2_mu(v2590, v2588, v2588, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2592 = __riscv_vwmacc_vx_i16m1(v2514, v2538, v2591, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2593 = __riscv_vsrl_vx_u8mf2(v2583, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2594 = __riscv_vand_vx_u8mf2(v2593, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2595 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2594);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2596 = __riscv_vand_vx_u8mf2(v2586, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2597 = __riscv_vmseq_vx_u8mf2_b16(v2596, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2598 = __riscv_vadd_vx_i8mf2_mu(v2597, v2595, v2595, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2599 = __riscv_vwmacc_vx_i16m1(v2521, v2541, v2598, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2600 = __riscv_vsrl_vx_u8mf2(v2583, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2601 = __riscv_vand_vx_u8mf2(v2600, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2602 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2601);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2603 = __riscv_vand_vx_u8mf2(v2586, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2604 = __riscv_vmseq_vx_u8mf2_b16(v2603, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2605 = __riscv_vadd_vx_i8mf2_mu(v2604, v2602, v2602, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2606 = __riscv_vwmacc_vx_i16m1(v2528, v2544, v2605, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2607 = __riscv_vsrl_vx_u8mf2(v2583, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2608 = __riscv_vand_vx_u8mf2(v2607, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2609 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2608);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2610 = __riscv_vand_vx_u8mf2(v2586, 8, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2611 = __riscv_vmseq_vx_u8mf2_b16(v2610, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2612 = __riscv_vadd_vx_i8mf2_mu(v2611, v2609, v2609, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2613 = __riscv_vwmacc_vx_i16m1(v2535, v2547, v2612, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v2614 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2615 = __riscv_vwmacc_vv_i32m2(v2614, v1329, v2559, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2616 = __riscv_vwmacc_vv_i32m2(v2615, v1333, v2566, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2617 = __riscv_vwmacc_vv_i32m2(v2616, v1337, v2573, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2618 = __riscv_vwmacc_vv_i32m2(v2617, v1341, v2580, 8);
      v24 = v2618;
      vint32m2_t v2619 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2620 = __riscv_vwmacc_vv_i32m2(v2619, v1345, v2592, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2621 = __riscv_vwmacc_vv_i32m2(v2620, v1349, v2599, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2622 = __riscv_vwmacc_vv_i32m2(v2621, v1353, v2606, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2623 = __riscv_vwmacc_vv_i32m2(v2622, v1357, v2613, 8);
      v26 = v2623;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
      const uint8_t* v2624 = v19 + 160;
      const int8_t* v2625 = (const int8_t*) v2624;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v2626 = __riscv_vle8_v_i8mf2(v2625, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v2627 = __riscv_vsext_vf2_i16m1(v2626, 8);
      const uint8_t* v2628 = v19 + 192;
      const int8_t* v2629 = (const int8_t*) v2628;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v2630 = __riscv_vle8_v_i8mf2(v2629, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v2631 = __riscv_vsext_vf2_i16m1(v2630, 8);
      const uint8_t* v2632 = v19 + 224;
      const int8_t* v2633 = (const int8_t*) v2632;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v2634 = __riscv_vle8_v_i8mf2(v2633, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v2635 = __riscv_vsext_vf2_i16m1(v2634, 8);
      const uint8_t* v2636 = v19 + 256;
      const int8_t* v2637 = (const int8_t*) v2636;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v2638 = __riscv_vle8_v_i8mf2(v2637, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v2639 = __riscv_vsext_vf2_i16m1(v2638, 8);
      const uint8_t* v2640 = v19 + 168;
      const int8_t* v2641 = (const int8_t*) v2640;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v2642 = __riscv_vle8_v_i8mf2(v2641, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v2643 = __riscv_vsext_vf2_i16m1(v2642, 8);
      const uint8_t* v2644 = v19 + 200;
      const int8_t* v2645 = (const int8_t*) v2644;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v2646 = __riscv_vle8_v_i8mf2(v2645, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v2647 = __riscv_vsext_vf2_i16m1(v2646, 8);
      const uint8_t* v2648 = v19 + 232;
      const int8_t* v2649 = (const int8_t*) v2648;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v2650 = __riscv_vle8_v_i8mf2(v2649, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v2651 = __riscv_vsext_vf2_i16m1(v2650, 8);
      const uint8_t* v2652 = v19 + 264;
      const int8_t* v2653 = (const int8_t*) v2652;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v2654 = __riscv_vle8_v_i8mf2(v2653, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v2655 = __riscv_vsext_vf2_i16m1(v2654, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2656 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2657 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2658 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2659 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2660 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2661 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2662 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2663 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2664 = v21 + 132;
      const int8_t* v2665 = (const int8_t*) v2664;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2666 = *(const int8_t *)(v2665);
      const uint8_t* v2667 = v21 + 164;
      const int8_t* v2668 = (const int8_t*) v2667;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2669 = *(const int8_t *)(v2668);
      const uint8_t* v2670 = v21 + 196;
      const int8_t* v2671 = (const int8_t*) v2670;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2672 = *(const int8_t *)(v2671);
      const uint8_t* v2673 = v21 + 228;
      const int8_t* v2674 = (const int8_t*) v2673;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2675 = *(const int8_t *)(v2674);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2676 = v19 + 1312;
      const uint8_t* v2677 = (const uint8_t*) v2676;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2678 = __riscv_vle8_v_u8mf2(v2677, 8);
      const uint8_t* v2679 = v19 + 288;
      const uint8_t* v2680 = (const uint8_t*) v2679;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2681 = __riscv_vle8_v_u8mf2(v2680, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2682 = __riscv_vand_vx_u8mf2(v2678, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2683 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2682);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2684 = __riscv_vand_vx_u8mf2(v2681, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2685 = __riscv_vmseq_vx_u8mf2_b16(v2684, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2686 = __riscv_vadd_vx_i8mf2_mu(v2685, v2683, v2683, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2687 = __riscv_vwmacc_vx_i16m1(v2656, v2666, v2686, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2688 = __riscv_vsrl_vx_u8mf2(v2678, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2689 = __riscv_vand_vx_u8mf2(v2688, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2690 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2689);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2691 = __riscv_vand_vx_u8mf2(v2681, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2692 = __riscv_vmseq_vx_u8mf2_b16(v2691, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2693 = __riscv_vadd_vx_i8mf2_mu(v2692, v2690, v2690, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2694 = __riscv_vwmacc_vx_i16m1(v2657, v2669, v2693, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2695 = __riscv_vsrl_vx_u8mf2(v2678, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2696 = __riscv_vand_vx_u8mf2(v2695, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2697 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2696);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2698 = __riscv_vand_vx_u8mf2(v2681, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2699 = __riscv_vmseq_vx_u8mf2_b16(v2698, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2700 = __riscv_vadd_vx_i8mf2_mu(v2699, v2697, v2697, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2701 = __riscv_vwmacc_vx_i16m1(v2658, v2672, v2700, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2702 = __riscv_vsrl_vx_u8mf2(v2678, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2703 = __riscv_vand_vx_u8mf2(v2702, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2704 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2703);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2705 = __riscv_vand_vx_u8mf2(v2681, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2706 = __riscv_vmseq_vx_u8mf2_b16(v2705, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2707 = __riscv_vadd_vx_i8mf2_mu(v2706, v2704, v2704, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2708 = __riscv_vwmacc_vx_i16m1(v2659, v2675, v2707, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2709 = v19 + 1320;
      const uint8_t* v2710 = (const uint8_t*) v2709;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2711 = __riscv_vle8_v_u8mf2(v2710, 8);
      const uint8_t* v2712 = v19 + 296;
      const uint8_t* v2713 = (const uint8_t*) v2712;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2714 = __riscv_vle8_v_u8mf2(v2713, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2715 = __riscv_vand_vx_u8mf2(v2711, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2716 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2715);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2717 = __riscv_vand_vx_u8mf2(v2714, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2718 = __riscv_vmseq_vx_u8mf2_b16(v2717, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2719 = __riscv_vadd_vx_i8mf2_mu(v2718, v2716, v2716, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2720 = __riscv_vwmacc_vx_i16m1(v2660, v2666, v2719, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2721 = __riscv_vsrl_vx_u8mf2(v2711, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2722 = __riscv_vand_vx_u8mf2(v2721, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2723 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2722);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2724 = __riscv_vand_vx_u8mf2(v2714, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2725 = __riscv_vmseq_vx_u8mf2_b16(v2724, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2726 = __riscv_vadd_vx_i8mf2_mu(v2725, v2723, v2723, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2727 = __riscv_vwmacc_vx_i16m1(v2661, v2669, v2726, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2728 = __riscv_vsrl_vx_u8mf2(v2711, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2729 = __riscv_vand_vx_u8mf2(v2728, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2730 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2729);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2731 = __riscv_vand_vx_u8mf2(v2714, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2732 = __riscv_vmseq_vx_u8mf2_b16(v2731, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2733 = __riscv_vadd_vx_i8mf2_mu(v2732, v2730, v2730, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2734 = __riscv_vwmacc_vx_i16m1(v2662, v2672, v2733, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2735 = __riscv_vsrl_vx_u8mf2(v2711, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2736 = __riscv_vand_vx_u8mf2(v2735, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2737 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2736);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2738 = __riscv_vand_vx_u8mf2(v2714, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2739 = __riscv_vmseq_vx_u8mf2_b16(v2738, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2740 = __riscv_vadd_vx_i8mf2_mu(v2739, v2737, v2737, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2741 = __riscv_vwmacc_vx_i16m1(v2663, v2675, v2740, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2742 = v21 + 133;
      const int8_t* v2743 = (const int8_t*) v2742;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2744 = *(const int8_t *)(v2743);
      const uint8_t* v2745 = v21 + 165;
      const int8_t* v2746 = (const int8_t*) v2745;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2747 = *(const int8_t *)(v2746);
      const uint8_t* v2748 = v21 + 197;
      const int8_t* v2749 = (const int8_t*) v2748;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2750 = *(const int8_t *)(v2749);
      const uint8_t* v2751 = v21 + 229;
      const int8_t* v2752 = (const int8_t*) v2751;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2753 = *(const int8_t *)(v2752);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2754 = v19 + 1328;
      const uint8_t* v2755 = (const uint8_t*) v2754;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2756 = __riscv_vle8_v_u8mf2(v2755, 8);
      const uint8_t* v2757 = v19 + 304;
      const uint8_t* v2758 = (const uint8_t*) v2757;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2759 = __riscv_vle8_v_u8mf2(v2758, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2760 = __riscv_vand_vx_u8mf2(v2756, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2761 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2760);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2762 = __riscv_vand_vx_u8mf2(v2759, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2763 = __riscv_vmseq_vx_u8mf2_b16(v2762, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2764 = __riscv_vadd_vx_i8mf2_mu(v2763, v2761, v2761, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2765 = __riscv_vwmacc_vx_i16m1(v2687, v2744, v2764, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2766 = __riscv_vsrl_vx_u8mf2(v2756, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2767 = __riscv_vand_vx_u8mf2(v2766, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2768 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2767);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2769 = __riscv_vand_vx_u8mf2(v2759, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2770 = __riscv_vmseq_vx_u8mf2_b16(v2769, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2771 = __riscv_vadd_vx_i8mf2_mu(v2770, v2768, v2768, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2772 = __riscv_vwmacc_vx_i16m1(v2694, v2747, v2771, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2773 = __riscv_vsrl_vx_u8mf2(v2756, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2774 = __riscv_vand_vx_u8mf2(v2773, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2775 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2774);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2776 = __riscv_vand_vx_u8mf2(v2759, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2777 = __riscv_vmseq_vx_u8mf2_b16(v2776, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2778 = __riscv_vadd_vx_i8mf2_mu(v2777, v2775, v2775, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2779 = __riscv_vwmacc_vx_i16m1(v2701, v2750, v2778, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2780 = __riscv_vsrl_vx_u8mf2(v2756, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2781 = __riscv_vand_vx_u8mf2(v2780, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2782 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2781);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2783 = __riscv_vand_vx_u8mf2(v2759, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2784 = __riscv_vmseq_vx_u8mf2_b16(v2783, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2785 = __riscv_vadd_vx_i8mf2_mu(v2784, v2782, v2782, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2786 = __riscv_vwmacc_vx_i16m1(v2708, v2753, v2785, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2787 = v19 + 1336;
      const uint8_t* v2788 = (const uint8_t*) v2787;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2789 = __riscv_vle8_v_u8mf2(v2788, 8);
      const uint8_t* v2790 = v19 + 312;
      const uint8_t* v2791 = (const uint8_t*) v2790;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2792 = __riscv_vle8_v_u8mf2(v2791, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2793 = __riscv_vand_vx_u8mf2(v2789, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2794 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2793);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2795 = __riscv_vand_vx_u8mf2(v2792, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2796 = __riscv_vmseq_vx_u8mf2_b16(v2795, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2797 = __riscv_vadd_vx_i8mf2_mu(v2796, v2794, v2794, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2798 = __riscv_vwmacc_vx_i16m1(v2720, v2744, v2797, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2799 = __riscv_vsrl_vx_u8mf2(v2789, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2800 = __riscv_vand_vx_u8mf2(v2799, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2801 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2800);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2802 = __riscv_vand_vx_u8mf2(v2792, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2803 = __riscv_vmseq_vx_u8mf2_b16(v2802, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2804 = __riscv_vadd_vx_i8mf2_mu(v2803, v2801, v2801, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2805 = __riscv_vwmacc_vx_i16m1(v2727, v2747, v2804, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2806 = __riscv_vsrl_vx_u8mf2(v2789, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2807 = __riscv_vand_vx_u8mf2(v2806, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2808 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2807);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2809 = __riscv_vand_vx_u8mf2(v2792, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2810 = __riscv_vmseq_vx_u8mf2_b16(v2809, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2811 = __riscv_vadd_vx_i8mf2_mu(v2810, v2808, v2808, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2812 = __riscv_vwmacc_vx_i16m1(v2734, v2750, v2811, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2813 = __riscv_vsrl_vx_u8mf2(v2789, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2814 = __riscv_vand_vx_u8mf2(v2813, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2815 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2814);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2816 = __riscv_vand_vx_u8mf2(v2792, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2817 = __riscv_vmseq_vx_u8mf2_b16(v2816, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2818 = __riscv_vadd_vx_i8mf2_mu(v2817, v2815, v2815, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2819 = __riscv_vwmacc_vx_i16m1(v2741, v2753, v2818, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2820 = v21 + 134;
      const int8_t* v2821 = (const int8_t*) v2820;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2822 = *(const int8_t *)(v2821);
      const uint8_t* v2823 = v21 + 166;
      const int8_t* v2824 = (const int8_t*) v2823;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2825 = *(const int8_t *)(v2824);
      const uint8_t* v2826 = v21 + 198;
      const int8_t* v2827 = (const int8_t*) v2826;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2828 = *(const int8_t *)(v2827);
      const uint8_t* v2829 = v21 + 230;
      const int8_t* v2830 = (const int8_t*) v2829;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2831 = *(const int8_t *)(v2830);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2832 = v19 + 1344;
      const uint8_t* v2833 = (const uint8_t*) v2832;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2834 = __riscv_vle8_v_u8mf2(v2833, 8);
      const uint8_t* v2835 = v19 + 320;
      const uint8_t* v2836 = (const uint8_t*) v2835;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2837 = __riscv_vle8_v_u8mf2(v2836, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2838 = __riscv_vand_vx_u8mf2(v2834, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2839 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2838);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2840 = __riscv_vand_vx_u8mf2(v2837, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2841 = __riscv_vmseq_vx_u8mf2_b16(v2840, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2842 = __riscv_vadd_vx_i8mf2_mu(v2841, v2839, v2839, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2843 = __riscv_vwmacc_vx_i16m1(v2765, v2822, v2842, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2844 = __riscv_vsrl_vx_u8mf2(v2834, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2845 = __riscv_vand_vx_u8mf2(v2844, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2846 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2845);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2847 = __riscv_vand_vx_u8mf2(v2837, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2848 = __riscv_vmseq_vx_u8mf2_b16(v2847, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2849 = __riscv_vadd_vx_i8mf2_mu(v2848, v2846, v2846, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2850 = __riscv_vwmacc_vx_i16m1(v2772, v2825, v2849, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2851 = __riscv_vsrl_vx_u8mf2(v2834, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2852 = __riscv_vand_vx_u8mf2(v2851, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2853 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2852);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2854 = __riscv_vand_vx_u8mf2(v2837, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2855 = __riscv_vmseq_vx_u8mf2_b16(v2854, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2856 = __riscv_vadd_vx_i8mf2_mu(v2855, v2853, v2853, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2857 = __riscv_vwmacc_vx_i16m1(v2779, v2828, v2856, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2858 = __riscv_vsrl_vx_u8mf2(v2834, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2859 = __riscv_vand_vx_u8mf2(v2858, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2860 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2859);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2861 = __riscv_vand_vx_u8mf2(v2837, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2862 = __riscv_vmseq_vx_u8mf2_b16(v2861, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2863 = __riscv_vadd_vx_i8mf2_mu(v2862, v2860, v2860, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2864 = __riscv_vwmacc_vx_i16m1(v2786, v2831, v2863, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2865 = v19 + 1352;
      const uint8_t* v2866 = (const uint8_t*) v2865;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2867 = __riscv_vle8_v_u8mf2(v2866, 8);
      const uint8_t* v2868 = v19 + 328;
      const uint8_t* v2869 = (const uint8_t*) v2868;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2870 = __riscv_vle8_v_u8mf2(v2869, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2871 = __riscv_vand_vx_u8mf2(v2867, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2872 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2871);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2873 = __riscv_vand_vx_u8mf2(v2870, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2874 = __riscv_vmseq_vx_u8mf2_b16(v2873, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2875 = __riscv_vadd_vx_i8mf2_mu(v2874, v2872, v2872, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2876 = __riscv_vwmacc_vx_i16m1(v2798, v2822, v2875, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2877 = __riscv_vsrl_vx_u8mf2(v2867, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2878 = __riscv_vand_vx_u8mf2(v2877, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2879 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2878);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2880 = __riscv_vand_vx_u8mf2(v2870, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2881 = __riscv_vmseq_vx_u8mf2_b16(v2880, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2882 = __riscv_vadd_vx_i8mf2_mu(v2881, v2879, v2879, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2883 = __riscv_vwmacc_vx_i16m1(v2805, v2825, v2882, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2884 = __riscv_vsrl_vx_u8mf2(v2867, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2885 = __riscv_vand_vx_u8mf2(v2884, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2886 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2885);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2887 = __riscv_vand_vx_u8mf2(v2870, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2888 = __riscv_vmseq_vx_u8mf2_b16(v2887, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2889 = __riscv_vadd_vx_i8mf2_mu(v2888, v2886, v2886, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2890 = __riscv_vwmacc_vx_i16m1(v2812, v2828, v2889, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2891 = __riscv_vsrl_vx_u8mf2(v2867, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2892 = __riscv_vand_vx_u8mf2(v2891, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2893 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2892);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2894 = __riscv_vand_vx_u8mf2(v2870, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2895 = __riscv_vmseq_vx_u8mf2_b16(v2894, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2896 = __riscv_vadd_vx_i8mf2_mu(v2895, v2893, v2893, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2897 = __riscv_vwmacc_vx_i16m1(v2819, v2831, v2896, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2898 = v21 + 135;
      const int8_t* v2899 = (const int8_t*) v2898;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2900 = *(const int8_t *)(v2899);
      const uint8_t* v2901 = v21 + 167;
      const int8_t* v2902 = (const int8_t*) v2901;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2903 = *(const int8_t *)(v2902);
      const uint8_t* v2904 = v21 + 199;
      const int8_t* v2905 = (const int8_t*) v2904;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2906 = *(const int8_t *)(v2905);
      const uint8_t* v2907 = v21 + 231;
      const int8_t* v2908 = (const int8_t*) v2907;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2909 = *(const int8_t *)(v2908);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2910 = v19 + 1360;
      const uint8_t* v2911 = (const uint8_t*) v2910;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2912 = __riscv_vle8_v_u8mf2(v2911, 8);
      const uint8_t* v2913 = v19 + 336;
      const uint8_t* v2914 = (const uint8_t*) v2913;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2915 = __riscv_vle8_v_u8mf2(v2914, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2916 = __riscv_vand_vx_u8mf2(v2912, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2917 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2916);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2918 = __riscv_vand_vx_u8mf2(v2915, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2919 = __riscv_vmseq_vx_u8mf2_b16(v2918, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2920 = __riscv_vadd_vx_i8mf2_mu(v2919, v2917, v2917, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2921 = __riscv_vwmacc_vx_i16m1(v2843, v2900, v2920, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2922 = __riscv_vsrl_vx_u8mf2(v2912, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2923 = __riscv_vand_vx_u8mf2(v2922, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2924 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2923);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2925 = __riscv_vand_vx_u8mf2(v2915, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2926 = __riscv_vmseq_vx_u8mf2_b16(v2925, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2927 = __riscv_vadd_vx_i8mf2_mu(v2926, v2924, v2924, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2928 = __riscv_vwmacc_vx_i16m1(v2850, v2903, v2927, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2929 = __riscv_vsrl_vx_u8mf2(v2912, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2930 = __riscv_vand_vx_u8mf2(v2929, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2931 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2930);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2932 = __riscv_vand_vx_u8mf2(v2915, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2933 = __riscv_vmseq_vx_u8mf2_b16(v2932, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2934 = __riscv_vadd_vx_i8mf2_mu(v2933, v2931, v2931, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2935 = __riscv_vwmacc_vx_i16m1(v2857, v2906, v2934, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2936 = __riscv_vsrl_vx_u8mf2(v2912, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2937 = __riscv_vand_vx_u8mf2(v2936, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2938 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2937);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2939 = __riscv_vand_vx_u8mf2(v2915, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2940 = __riscv_vmseq_vx_u8mf2_b16(v2939, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2941 = __riscv_vadd_vx_i8mf2_mu(v2940, v2938, v2938, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2942 = __riscv_vwmacc_vx_i16m1(v2864, v2909, v2941, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2943 = v19 + 1368;
      const uint8_t* v2944 = (const uint8_t*) v2943;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2945 = __riscv_vle8_v_u8mf2(v2944, 8);
      const uint8_t* v2946 = v19 + 344;
      const uint8_t* v2947 = (const uint8_t*) v2946;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2948 = __riscv_vle8_v_u8mf2(v2947, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2949 = __riscv_vand_vx_u8mf2(v2945, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2950 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2949);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2951 = __riscv_vand_vx_u8mf2(v2948, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2952 = __riscv_vmseq_vx_u8mf2_b16(v2951, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2953 = __riscv_vadd_vx_i8mf2_mu(v2952, v2950, v2950, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2954 = __riscv_vwmacc_vx_i16m1(v2876, v2900, v2953, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2955 = __riscv_vsrl_vx_u8mf2(v2945, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2956 = __riscv_vand_vx_u8mf2(v2955, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2957 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2956);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2958 = __riscv_vand_vx_u8mf2(v2948, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2959 = __riscv_vmseq_vx_u8mf2_b16(v2958, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2960 = __riscv_vadd_vx_i8mf2_mu(v2959, v2957, v2957, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2961 = __riscv_vwmacc_vx_i16m1(v2883, v2903, v2960, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2962 = __riscv_vsrl_vx_u8mf2(v2945, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2963 = __riscv_vand_vx_u8mf2(v2962, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2964 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2963);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2965 = __riscv_vand_vx_u8mf2(v2948, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2966 = __riscv_vmseq_vx_u8mf2_b16(v2965, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2967 = __riscv_vadd_vx_i8mf2_mu(v2966, v2964, v2964, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2968 = __riscv_vwmacc_vx_i16m1(v2890, v2906, v2967, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2969 = __riscv_vsrl_vx_u8mf2(v2945, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2970 = __riscv_vand_vx_u8mf2(v2969, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2971 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2970);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2972 = __riscv_vand_vx_u8mf2(v2948, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2973 = __riscv_vmseq_vx_u8mf2_b16(v2972, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2974 = __riscv_vadd_vx_i8mf2_mu(v2973, v2971, v2971, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2975 = __riscv_vwmacc_vx_i16m1(v2897, v2909, v2974, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2976 = v21 + 136;
      const int8_t* v2977 = (const int8_t*) v2976;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2978 = *(const int8_t *)(v2977);
      const uint8_t* v2979 = v21 + 168;
      const int8_t* v2980 = (const int8_t*) v2979;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2981 = *(const int8_t *)(v2980);
      const uint8_t* v2982 = v21 + 200;
      const int8_t* v2983 = (const int8_t*) v2982;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2984 = *(const int8_t *)(v2983);
      const uint8_t* v2985 = v21 + 232;
      const int8_t* v2986 = (const int8_t*) v2985;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2987 = *(const int8_t *)(v2986);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v2988 = v19 + 1376;
      const uint8_t* v2989 = (const uint8_t*) v2988;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2990 = __riscv_vle8_v_u8mf2(v2989, 8);
      const uint8_t* v2991 = v19 + 352;
      const uint8_t* v2992 = (const uint8_t*) v2991;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2993 = __riscv_vle8_v_u8mf2(v2992, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2994 = __riscv_vand_vx_u8mf2(v2990, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2995 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2994);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2996 = __riscv_vand_vx_u8mf2(v2993, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v2997 = __riscv_vmseq_vx_u8mf2_b16(v2996, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v2998 = __riscv_vadd_vx_i8mf2_mu(v2997, v2995, v2995, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2999 = __riscv_vwmacc_vx_i16m1(v2921, v2978, v2998, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3000 = __riscv_vsrl_vx_u8mf2(v2990, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3001 = __riscv_vand_vx_u8mf2(v3000, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3002 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3001);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3003 = __riscv_vand_vx_u8mf2(v2993, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3004 = __riscv_vmseq_vx_u8mf2_b16(v3003, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3005 = __riscv_vadd_vx_i8mf2_mu(v3004, v3002, v3002, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3006 = __riscv_vwmacc_vx_i16m1(v2928, v2981, v3005, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3007 = __riscv_vsrl_vx_u8mf2(v2990, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3008 = __riscv_vand_vx_u8mf2(v3007, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3009 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3008);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3010 = __riscv_vand_vx_u8mf2(v2993, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3011 = __riscv_vmseq_vx_u8mf2_b16(v3010, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3012 = __riscv_vadd_vx_i8mf2_mu(v3011, v3009, v3009, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3013 = __riscv_vwmacc_vx_i16m1(v2935, v2984, v3012, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3014 = __riscv_vsrl_vx_u8mf2(v2990, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3015 = __riscv_vand_vx_u8mf2(v3014, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3016 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3015);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3017 = __riscv_vand_vx_u8mf2(v2993, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3018 = __riscv_vmseq_vx_u8mf2_b16(v3017, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3019 = __riscv_vadd_vx_i8mf2_mu(v3018, v3016, v3016, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3020 = __riscv_vwmacc_vx_i16m1(v2942, v2987, v3019, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3021 = v19 + 1384;
      const uint8_t* v3022 = (const uint8_t*) v3021;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3023 = __riscv_vle8_v_u8mf2(v3022, 8);
      const uint8_t* v3024 = v19 + 360;
      const uint8_t* v3025 = (const uint8_t*) v3024;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3026 = __riscv_vle8_v_u8mf2(v3025, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3027 = __riscv_vand_vx_u8mf2(v3023, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3028 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3027);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3029 = __riscv_vand_vx_u8mf2(v3026, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3030 = __riscv_vmseq_vx_u8mf2_b16(v3029, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3031 = __riscv_vadd_vx_i8mf2_mu(v3030, v3028, v3028, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3032 = __riscv_vwmacc_vx_i16m1(v2954, v2978, v3031, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3033 = __riscv_vsrl_vx_u8mf2(v3023, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3034 = __riscv_vand_vx_u8mf2(v3033, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3035 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3034);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3036 = __riscv_vand_vx_u8mf2(v3026, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3037 = __riscv_vmseq_vx_u8mf2_b16(v3036, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3038 = __riscv_vadd_vx_i8mf2_mu(v3037, v3035, v3035, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3039 = __riscv_vwmacc_vx_i16m1(v2961, v2981, v3038, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3040 = __riscv_vsrl_vx_u8mf2(v3023, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3041 = __riscv_vand_vx_u8mf2(v3040, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3042 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3041);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3043 = __riscv_vand_vx_u8mf2(v3026, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3044 = __riscv_vmseq_vx_u8mf2_b16(v3043, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3045 = __riscv_vadd_vx_i8mf2_mu(v3044, v3042, v3042, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3046 = __riscv_vwmacc_vx_i16m1(v2968, v2984, v3045, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3047 = __riscv_vsrl_vx_u8mf2(v3023, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3048 = __riscv_vand_vx_u8mf2(v3047, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3049 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3048);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3050 = __riscv_vand_vx_u8mf2(v3026, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3051 = __riscv_vmseq_vx_u8mf2_b16(v3050, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3052 = __riscv_vadd_vx_i8mf2_mu(v3051, v3049, v3049, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3053 = __riscv_vwmacc_vx_i16m1(v2975, v2987, v3052, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3054 = v21 + 137;
      const int8_t* v3055 = (const int8_t*) v3054;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3056 = *(const int8_t *)(v3055);
      const uint8_t* v3057 = v21 + 169;
      const int8_t* v3058 = (const int8_t*) v3057;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3059 = *(const int8_t *)(v3058);
      const uint8_t* v3060 = v21 + 201;
      const int8_t* v3061 = (const int8_t*) v3060;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3062 = *(const int8_t *)(v3061);
      const uint8_t* v3063 = v21 + 233;
      const int8_t* v3064 = (const int8_t*) v3063;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3065 = *(const int8_t *)(v3064);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3066 = v19 + 1392;
      const uint8_t* v3067 = (const uint8_t*) v3066;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3068 = __riscv_vle8_v_u8mf2(v3067, 8);
      const uint8_t* v3069 = v19 + 368;
      const uint8_t* v3070 = (const uint8_t*) v3069;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3071 = __riscv_vle8_v_u8mf2(v3070, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3072 = __riscv_vand_vx_u8mf2(v3068, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3073 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3072);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3074 = __riscv_vand_vx_u8mf2(v3071, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3075 = __riscv_vmseq_vx_u8mf2_b16(v3074, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3076 = __riscv_vadd_vx_i8mf2_mu(v3075, v3073, v3073, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3077 = __riscv_vwmacc_vx_i16m1(v2999, v3056, v3076, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3078 = __riscv_vsrl_vx_u8mf2(v3068, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3079 = __riscv_vand_vx_u8mf2(v3078, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3080 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3079);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3081 = __riscv_vand_vx_u8mf2(v3071, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3082 = __riscv_vmseq_vx_u8mf2_b16(v3081, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3083 = __riscv_vadd_vx_i8mf2_mu(v3082, v3080, v3080, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3084 = __riscv_vwmacc_vx_i16m1(v3006, v3059, v3083, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3085 = __riscv_vsrl_vx_u8mf2(v3068, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3086 = __riscv_vand_vx_u8mf2(v3085, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3087 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3086);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3088 = __riscv_vand_vx_u8mf2(v3071, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3089 = __riscv_vmseq_vx_u8mf2_b16(v3088, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3090 = __riscv_vadd_vx_i8mf2_mu(v3089, v3087, v3087, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3091 = __riscv_vwmacc_vx_i16m1(v3013, v3062, v3090, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3092 = __riscv_vsrl_vx_u8mf2(v3068, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3093 = __riscv_vand_vx_u8mf2(v3092, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3094 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3093);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3095 = __riscv_vand_vx_u8mf2(v3071, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3096 = __riscv_vmseq_vx_u8mf2_b16(v3095, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3097 = __riscv_vadd_vx_i8mf2_mu(v3096, v3094, v3094, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3098 = __riscv_vwmacc_vx_i16m1(v3020, v3065, v3097, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3099 = v19 + 1400;
      const uint8_t* v3100 = (const uint8_t*) v3099;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3101 = __riscv_vle8_v_u8mf2(v3100, 8);
      const uint8_t* v3102 = v19 + 376;
      const uint8_t* v3103 = (const uint8_t*) v3102;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3104 = __riscv_vle8_v_u8mf2(v3103, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3105 = __riscv_vand_vx_u8mf2(v3101, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3106 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3105);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3107 = __riscv_vand_vx_u8mf2(v3104, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3108 = __riscv_vmseq_vx_u8mf2_b16(v3107, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3109 = __riscv_vadd_vx_i8mf2_mu(v3108, v3106, v3106, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3110 = __riscv_vwmacc_vx_i16m1(v3032, v3056, v3109, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3111 = __riscv_vsrl_vx_u8mf2(v3101, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3112 = __riscv_vand_vx_u8mf2(v3111, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3113 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3112);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3114 = __riscv_vand_vx_u8mf2(v3104, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3115 = __riscv_vmseq_vx_u8mf2_b16(v3114, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3116 = __riscv_vadd_vx_i8mf2_mu(v3115, v3113, v3113, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3117 = __riscv_vwmacc_vx_i16m1(v3039, v3059, v3116, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3118 = __riscv_vsrl_vx_u8mf2(v3101, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3119 = __riscv_vand_vx_u8mf2(v3118, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3120 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3119);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3121 = __riscv_vand_vx_u8mf2(v3104, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3122 = __riscv_vmseq_vx_u8mf2_b16(v3121, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3123 = __riscv_vadd_vx_i8mf2_mu(v3122, v3120, v3120, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3124 = __riscv_vwmacc_vx_i16m1(v3046, v3062, v3123, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3125 = __riscv_vsrl_vx_u8mf2(v3101, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3126 = __riscv_vand_vx_u8mf2(v3125, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3127 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3126);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3128 = __riscv_vand_vx_u8mf2(v3104, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3129 = __riscv_vmseq_vx_u8mf2_b16(v3128, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3130 = __riscv_vadd_vx_i8mf2_mu(v3129, v3127, v3127, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3131 = __riscv_vwmacc_vx_i16m1(v3053, v3065, v3130, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3132 = v21 + 138;
      const int8_t* v3133 = (const int8_t*) v3132;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3134 = *(const int8_t *)(v3133);
      const uint8_t* v3135 = v21 + 170;
      const int8_t* v3136 = (const int8_t*) v3135;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3137 = *(const int8_t *)(v3136);
      const uint8_t* v3138 = v21 + 202;
      const int8_t* v3139 = (const int8_t*) v3138;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3140 = *(const int8_t *)(v3139);
      const uint8_t* v3141 = v21 + 234;
      const int8_t* v3142 = (const int8_t*) v3141;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3143 = *(const int8_t *)(v3142);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3144 = v19 + 1408;
      const uint8_t* v3145 = (const uint8_t*) v3144;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3146 = __riscv_vle8_v_u8mf2(v3145, 8);
      const uint8_t* v3147 = v19 + 384;
      const uint8_t* v3148 = (const uint8_t*) v3147;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3149 = __riscv_vle8_v_u8mf2(v3148, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3150 = __riscv_vand_vx_u8mf2(v3146, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3151 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3150);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3152 = __riscv_vand_vx_u8mf2(v3149, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3153 = __riscv_vmseq_vx_u8mf2_b16(v3152, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3154 = __riscv_vadd_vx_i8mf2_mu(v3153, v3151, v3151, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3155 = __riscv_vwmacc_vx_i16m1(v3077, v3134, v3154, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3156 = __riscv_vsrl_vx_u8mf2(v3146, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3157 = __riscv_vand_vx_u8mf2(v3156, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3158 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3157);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3159 = __riscv_vand_vx_u8mf2(v3149, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3160 = __riscv_vmseq_vx_u8mf2_b16(v3159, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3161 = __riscv_vadd_vx_i8mf2_mu(v3160, v3158, v3158, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3162 = __riscv_vwmacc_vx_i16m1(v3084, v3137, v3161, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3163 = __riscv_vsrl_vx_u8mf2(v3146, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3164 = __riscv_vand_vx_u8mf2(v3163, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3165 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3164);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3166 = __riscv_vand_vx_u8mf2(v3149, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3167 = __riscv_vmseq_vx_u8mf2_b16(v3166, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3168 = __riscv_vadd_vx_i8mf2_mu(v3167, v3165, v3165, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3169 = __riscv_vwmacc_vx_i16m1(v3091, v3140, v3168, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3170 = __riscv_vsrl_vx_u8mf2(v3146, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3171 = __riscv_vand_vx_u8mf2(v3170, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3172 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3171);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3173 = __riscv_vand_vx_u8mf2(v3149, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3174 = __riscv_vmseq_vx_u8mf2_b16(v3173, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3175 = __riscv_vadd_vx_i8mf2_mu(v3174, v3172, v3172, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3176 = __riscv_vwmacc_vx_i16m1(v3098, v3143, v3175, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3177 = v19 + 1416;
      const uint8_t* v3178 = (const uint8_t*) v3177;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3179 = __riscv_vle8_v_u8mf2(v3178, 8);
      const uint8_t* v3180 = v19 + 392;
      const uint8_t* v3181 = (const uint8_t*) v3180;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3182 = __riscv_vle8_v_u8mf2(v3181, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3183 = __riscv_vand_vx_u8mf2(v3179, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3184 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3183);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3185 = __riscv_vand_vx_u8mf2(v3182, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3186 = __riscv_vmseq_vx_u8mf2_b16(v3185, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3187 = __riscv_vadd_vx_i8mf2_mu(v3186, v3184, v3184, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3188 = __riscv_vwmacc_vx_i16m1(v3110, v3134, v3187, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3189 = __riscv_vsrl_vx_u8mf2(v3179, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3190 = __riscv_vand_vx_u8mf2(v3189, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3191 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3190);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3192 = __riscv_vand_vx_u8mf2(v3182, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3193 = __riscv_vmseq_vx_u8mf2_b16(v3192, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3194 = __riscv_vadd_vx_i8mf2_mu(v3193, v3191, v3191, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3195 = __riscv_vwmacc_vx_i16m1(v3117, v3137, v3194, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3196 = __riscv_vsrl_vx_u8mf2(v3179, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3197 = __riscv_vand_vx_u8mf2(v3196, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3198 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3197);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3199 = __riscv_vand_vx_u8mf2(v3182, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3200 = __riscv_vmseq_vx_u8mf2_b16(v3199, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3201 = __riscv_vadd_vx_i8mf2_mu(v3200, v3198, v3198, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3202 = __riscv_vwmacc_vx_i16m1(v3124, v3140, v3201, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3203 = __riscv_vsrl_vx_u8mf2(v3179, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3204 = __riscv_vand_vx_u8mf2(v3203, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3205 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3204);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3206 = __riscv_vand_vx_u8mf2(v3182, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3207 = __riscv_vmseq_vx_u8mf2_b16(v3206, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3208 = __riscv_vadd_vx_i8mf2_mu(v3207, v3205, v3205, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3209 = __riscv_vwmacc_vx_i16m1(v3131, v3143, v3208, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3210 = v21 + 139;
      const int8_t* v3211 = (const int8_t*) v3210;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3212 = *(const int8_t *)(v3211);
      const uint8_t* v3213 = v21 + 171;
      const int8_t* v3214 = (const int8_t*) v3213;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3215 = *(const int8_t *)(v3214);
      const uint8_t* v3216 = v21 + 203;
      const int8_t* v3217 = (const int8_t*) v3216;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3218 = *(const int8_t *)(v3217);
      const uint8_t* v3219 = v21 + 235;
      const int8_t* v3220 = (const int8_t*) v3219;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3221 = *(const int8_t *)(v3220);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3222 = v19 + 1424;
      const uint8_t* v3223 = (const uint8_t*) v3222;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3224 = __riscv_vle8_v_u8mf2(v3223, 8);
      const uint8_t* v3225 = v19 + 400;
      const uint8_t* v3226 = (const uint8_t*) v3225;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3227 = __riscv_vle8_v_u8mf2(v3226, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3228 = __riscv_vand_vx_u8mf2(v3224, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3229 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3228);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3230 = __riscv_vand_vx_u8mf2(v3227, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3231 = __riscv_vmseq_vx_u8mf2_b16(v3230, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3232 = __riscv_vadd_vx_i8mf2_mu(v3231, v3229, v3229, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3233 = __riscv_vwmacc_vx_i16m1(v3155, v3212, v3232, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3234 = __riscv_vsrl_vx_u8mf2(v3224, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3235 = __riscv_vand_vx_u8mf2(v3234, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3236 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3235);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3237 = __riscv_vand_vx_u8mf2(v3227, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3238 = __riscv_vmseq_vx_u8mf2_b16(v3237, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3239 = __riscv_vadd_vx_i8mf2_mu(v3238, v3236, v3236, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3240 = __riscv_vwmacc_vx_i16m1(v3162, v3215, v3239, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3241 = __riscv_vsrl_vx_u8mf2(v3224, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3242 = __riscv_vand_vx_u8mf2(v3241, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3243 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3242);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3244 = __riscv_vand_vx_u8mf2(v3227, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3245 = __riscv_vmseq_vx_u8mf2_b16(v3244, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3246 = __riscv_vadd_vx_i8mf2_mu(v3245, v3243, v3243, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3247 = __riscv_vwmacc_vx_i16m1(v3169, v3218, v3246, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3248 = __riscv_vsrl_vx_u8mf2(v3224, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3249 = __riscv_vand_vx_u8mf2(v3248, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3250 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3249);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3251 = __riscv_vand_vx_u8mf2(v3227, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3252 = __riscv_vmseq_vx_u8mf2_b16(v3251, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3253 = __riscv_vadd_vx_i8mf2_mu(v3252, v3250, v3250, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3254 = __riscv_vwmacc_vx_i16m1(v3176, v3221, v3253, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3255 = v19 + 1432;
      const uint8_t* v3256 = (const uint8_t*) v3255;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3257 = __riscv_vle8_v_u8mf2(v3256, 8);
      const uint8_t* v3258 = v19 + 408;
      const uint8_t* v3259 = (const uint8_t*) v3258;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3260 = __riscv_vle8_v_u8mf2(v3259, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3261 = __riscv_vand_vx_u8mf2(v3257, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3262 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3261);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3263 = __riscv_vand_vx_u8mf2(v3260, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3264 = __riscv_vmseq_vx_u8mf2_b16(v3263, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3265 = __riscv_vadd_vx_i8mf2_mu(v3264, v3262, v3262, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3266 = __riscv_vwmacc_vx_i16m1(v3188, v3212, v3265, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3267 = __riscv_vsrl_vx_u8mf2(v3257, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3268 = __riscv_vand_vx_u8mf2(v3267, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3269 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3268);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3270 = __riscv_vand_vx_u8mf2(v3260, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3271 = __riscv_vmseq_vx_u8mf2_b16(v3270, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3272 = __riscv_vadd_vx_i8mf2_mu(v3271, v3269, v3269, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3273 = __riscv_vwmacc_vx_i16m1(v3195, v3215, v3272, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3274 = __riscv_vsrl_vx_u8mf2(v3257, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3275 = __riscv_vand_vx_u8mf2(v3274, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3276 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3275);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3277 = __riscv_vand_vx_u8mf2(v3260, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3278 = __riscv_vmseq_vx_u8mf2_b16(v3277, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3279 = __riscv_vadd_vx_i8mf2_mu(v3278, v3276, v3276, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3280 = __riscv_vwmacc_vx_i16m1(v3202, v3218, v3279, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3281 = __riscv_vsrl_vx_u8mf2(v3257, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3282 = __riscv_vand_vx_u8mf2(v3281, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3283 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3282);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3284 = __riscv_vand_vx_u8mf2(v3260, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3285 = __riscv_vmseq_vx_u8mf2_b16(v3284, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3286 = __riscv_vadd_vx_i8mf2_mu(v3285, v3283, v3283, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3287 = __riscv_vwmacc_vx_i16m1(v3209, v3221, v3286, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3288 = v21 + 140;
      const int8_t* v3289 = (const int8_t*) v3288;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3290 = *(const int8_t *)(v3289);
      const uint8_t* v3291 = v21 + 172;
      const int8_t* v3292 = (const int8_t*) v3291;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3293 = *(const int8_t *)(v3292);
      const uint8_t* v3294 = v21 + 204;
      const int8_t* v3295 = (const int8_t*) v3294;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3296 = *(const int8_t *)(v3295);
      const uint8_t* v3297 = v21 + 236;
      const int8_t* v3298 = (const int8_t*) v3297;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3299 = *(const int8_t *)(v3298);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3300 = v19 + 1440;
      const uint8_t* v3301 = (const uint8_t*) v3300;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3302 = __riscv_vle8_v_u8mf2(v3301, 8);
      const uint8_t* v3303 = v19 + 416;
      const uint8_t* v3304 = (const uint8_t*) v3303;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3305 = __riscv_vle8_v_u8mf2(v3304, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3306 = __riscv_vand_vx_u8mf2(v3302, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3307 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3306);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3308 = __riscv_vand_vx_u8mf2(v3305, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3309 = __riscv_vmseq_vx_u8mf2_b16(v3308, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3310 = __riscv_vadd_vx_i8mf2_mu(v3309, v3307, v3307, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3311 = __riscv_vwmacc_vx_i16m1(v3233, v3290, v3310, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3312 = __riscv_vsrl_vx_u8mf2(v3302, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3313 = __riscv_vand_vx_u8mf2(v3312, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3314 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3313);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3315 = __riscv_vand_vx_u8mf2(v3305, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3316 = __riscv_vmseq_vx_u8mf2_b16(v3315, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3317 = __riscv_vadd_vx_i8mf2_mu(v3316, v3314, v3314, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3318 = __riscv_vwmacc_vx_i16m1(v3240, v3293, v3317, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3319 = __riscv_vsrl_vx_u8mf2(v3302, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3320 = __riscv_vand_vx_u8mf2(v3319, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3321 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3320);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3322 = __riscv_vand_vx_u8mf2(v3305, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3323 = __riscv_vmseq_vx_u8mf2_b16(v3322, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3324 = __riscv_vadd_vx_i8mf2_mu(v3323, v3321, v3321, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3325 = __riscv_vwmacc_vx_i16m1(v3247, v3296, v3324, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3326 = __riscv_vsrl_vx_u8mf2(v3302, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3327 = __riscv_vand_vx_u8mf2(v3326, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3328 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3327);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3329 = __riscv_vand_vx_u8mf2(v3305, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3330 = __riscv_vmseq_vx_u8mf2_b16(v3329, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3331 = __riscv_vadd_vx_i8mf2_mu(v3330, v3328, v3328, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3332 = __riscv_vwmacc_vx_i16m1(v3254, v3299, v3331, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3333 = v19 + 1448;
      const uint8_t* v3334 = (const uint8_t*) v3333;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3335 = __riscv_vle8_v_u8mf2(v3334, 8);
      const uint8_t* v3336 = v19 + 424;
      const uint8_t* v3337 = (const uint8_t*) v3336;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3338 = __riscv_vle8_v_u8mf2(v3337, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3339 = __riscv_vand_vx_u8mf2(v3335, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3340 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3339);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3341 = __riscv_vand_vx_u8mf2(v3338, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3342 = __riscv_vmseq_vx_u8mf2_b16(v3341, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3343 = __riscv_vadd_vx_i8mf2_mu(v3342, v3340, v3340, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3344 = __riscv_vwmacc_vx_i16m1(v3266, v3290, v3343, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3345 = __riscv_vsrl_vx_u8mf2(v3335, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3346 = __riscv_vand_vx_u8mf2(v3345, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3347 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3346);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3348 = __riscv_vand_vx_u8mf2(v3338, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3349 = __riscv_vmseq_vx_u8mf2_b16(v3348, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3350 = __riscv_vadd_vx_i8mf2_mu(v3349, v3347, v3347, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3351 = __riscv_vwmacc_vx_i16m1(v3273, v3293, v3350, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3352 = __riscv_vsrl_vx_u8mf2(v3335, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3353 = __riscv_vand_vx_u8mf2(v3352, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3354 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3353);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3355 = __riscv_vand_vx_u8mf2(v3338, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3356 = __riscv_vmseq_vx_u8mf2_b16(v3355, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3357 = __riscv_vadd_vx_i8mf2_mu(v3356, v3354, v3354, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3358 = __riscv_vwmacc_vx_i16m1(v3280, v3296, v3357, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3359 = __riscv_vsrl_vx_u8mf2(v3335, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3360 = __riscv_vand_vx_u8mf2(v3359, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3361 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3360);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3362 = __riscv_vand_vx_u8mf2(v3338, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3363 = __riscv_vmseq_vx_u8mf2_b16(v3362, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3364 = __riscv_vadd_vx_i8mf2_mu(v3363, v3361, v3361, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3365 = __riscv_vwmacc_vx_i16m1(v3287, v3299, v3364, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3366 = v21 + 141;
      const int8_t* v3367 = (const int8_t*) v3366;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3368 = *(const int8_t *)(v3367);
      const uint8_t* v3369 = v21 + 173;
      const int8_t* v3370 = (const int8_t*) v3369;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3371 = *(const int8_t *)(v3370);
      const uint8_t* v3372 = v21 + 205;
      const int8_t* v3373 = (const int8_t*) v3372;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3374 = *(const int8_t *)(v3373);
      const uint8_t* v3375 = v21 + 237;
      const int8_t* v3376 = (const int8_t*) v3375;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3377 = *(const int8_t *)(v3376);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3378 = v19 + 1456;
      const uint8_t* v3379 = (const uint8_t*) v3378;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3380 = __riscv_vle8_v_u8mf2(v3379, 8);
      const uint8_t* v3381 = v19 + 432;
      const uint8_t* v3382 = (const uint8_t*) v3381;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3383 = __riscv_vle8_v_u8mf2(v3382, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3384 = __riscv_vand_vx_u8mf2(v3380, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3385 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3384);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3386 = __riscv_vand_vx_u8mf2(v3383, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3387 = __riscv_vmseq_vx_u8mf2_b16(v3386, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3388 = __riscv_vadd_vx_i8mf2_mu(v3387, v3385, v3385, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3389 = __riscv_vwmacc_vx_i16m1(v3311, v3368, v3388, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3390 = __riscv_vsrl_vx_u8mf2(v3380, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3391 = __riscv_vand_vx_u8mf2(v3390, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3392 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3391);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3393 = __riscv_vand_vx_u8mf2(v3383, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3394 = __riscv_vmseq_vx_u8mf2_b16(v3393, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3395 = __riscv_vadd_vx_i8mf2_mu(v3394, v3392, v3392, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3396 = __riscv_vwmacc_vx_i16m1(v3318, v3371, v3395, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3397 = __riscv_vsrl_vx_u8mf2(v3380, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3398 = __riscv_vand_vx_u8mf2(v3397, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3399 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3398);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3400 = __riscv_vand_vx_u8mf2(v3383, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3401 = __riscv_vmseq_vx_u8mf2_b16(v3400, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3402 = __riscv_vadd_vx_i8mf2_mu(v3401, v3399, v3399, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3403 = __riscv_vwmacc_vx_i16m1(v3325, v3374, v3402, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3404 = __riscv_vsrl_vx_u8mf2(v3380, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3405 = __riscv_vand_vx_u8mf2(v3404, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3406 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3405);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3407 = __riscv_vand_vx_u8mf2(v3383, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3408 = __riscv_vmseq_vx_u8mf2_b16(v3407, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3409 = __riscv_vadd_vx_i8mf2_mu(v3408, v3406, v3406, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3410 = __riscv_vwmacc_vx_i16m1(v3332, v3377, v3409, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3411 = v19 + 1464;
      const uint8_t* v3412 = (const uint8_t*) v3411;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3413 = __riscv_vle8_v_u8mf2(v3412, 8);
      const uint8_t* v3414 = v19 + 440;
      const uint8_t* v3415 = (const uint8_t*) v3414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3416 = __riscv_vle8_v_u8mf2(v3415, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3417 = __riscv_vand_vx_u8mf2(v3413, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3418 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3417);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3419 = __riscv_vand_vx_u8mf2(v3416, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3420 = __riscv_vmseq_vx_u8mf2_b16(v3419, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3421 = __riscv_vadd_vx_i8mf2_mu(v3420, v3418, v3418, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3422 = __riscv_vwmacc_vx_i16m1(v3344, v3368, v3421, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3423 = __riscv_vsrl_vx_u8mf2(v3413, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3424 = __riscv_vand_vx_u8mf2(v3423, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3425 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3424);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3426 = __riscv_vand_vx_u8mf2(v3416, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3427 = __riscv_vmseq_vx_u8mf2_b16(v3426, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3428 = __riscv_vadd_vx_i8mf2_mu(v3427, v3425, v3425, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3429 = __riscv_vwmacc_vx_i16m1(v3351, v3371, v3428, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3430 = __riscv_vsrl_vx_u8mf2(v3413, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3431 = __riscv_vand_vx_u8mf2(v3430, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3432 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3431);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3433 = __riscv_vand_vx_u8mf2(v3416, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3434 = __riscv_vmseq_vx_u8mf2_b16(v3433, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3435 = __riscv_vadd_vx_i8mf2_mu(v3434, v3432, v3432, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3436 = __riscv_vwmacc_vx_i16m1(v3358, v3374, v3435, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3437 = __riscv_vsrl_vx_u8mf2(v3413, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3438 = __riscv_vand_vx_u8mf2(v3437, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3439 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3438);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3440 = __riscv_vand_vx_u8mf2(v3416, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3441 = __riscv_vmseq_vx_u8mf2_b16(v3440, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3442 = __riscv_vadd_vx_i8mf2_mu(v3441, v3439, v3439, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3443 = __riscv_vwmacc_vx_i16m1(v3365, v3377, v3442, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3444 = v21 + 142;
      const int8_t* v3445 = (const int8_t*) v3444;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3446 = *(const int8_t *)(v3445);
      const uint8_t* v3447 = v21 + 174;
      const int8_t* v3448 = (const int8_t*) v3447;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3449 = *(const int8_t *)(v3448);
      const uint8_t* v3450 = v21 + 206;
      const int8_t* v3451 = (const int8_t*) v3450;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3452 = *(const int8_t *)(v3451);
      const uint8_t* v3453 = v21 + 238;
      const int8_t* v3454 = (const int8_t*) v3453;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3455 = *(const int8_t *)(v3454);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3456 = v19 + 1472;
      const uint8_t* v3457 = (const uint8_t*) v3456;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3458 = __riscv_vle8_v_u8mf2(v3457, 8);
      const uint8_t* v3459 = v19 + 448;
      const uint8_t* v3460 = (const uint8_t*) v3459;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3461 = __riscv_vle8_v_u8mf2(v3460, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3462 = __riscv_vand_vx_u8mf2(v3458, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3463 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3462);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3464 = __riscv_vand_vx_u8mf2(v3461, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3465 = __riscv_vmseq_vx_u8mf2_b16(v3464, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3466 = __riscv_vadd_vx_i8mf2_mu(v3465, v3463, v3463, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3467 = __riscv_vwmacc_vx_i16m1(v3389, v3446, v3466, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3468 = __riscv_vsrl_vx_u8mf2(v3458, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3469 = __riscv_vand_vx_u8mf2(v3468, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3470 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3469);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3471 = __riscv_vand_vx_u8mf2(v3461, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3472 = __riscv_vmseq_vx_u8mf2_b16(v3471, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3473 = __riscv_vadd_vx_i8mf2_mu(v3472, v3470, v3470, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3474 = __riscv_vwmacc_vx_i16m1(v3396, v3449, v3473, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3475 = __riscv_vsrl_vx_u8mf2(v3458, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3476 = __riscv_vand_vx_u8mf2(v3475, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3477 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3476);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3478 = __riscv_vand_vx_u8mf2(v3461, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3479 = __riscv_vmseq_vx_u8mf2_b16(v3478, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3480 = __riscv_vadd_vx_i8mf2_mu(v3479, v3477, v3477, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3481 = __riscv_vwmacc_vx_i16m1(v3403, v3452, v3480, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3482 = __riscv_vsrl_vx_u8mf2(v3458, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3483 = __riscv_vand_vx_u8mf2(v3482, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3484 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3483);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3485 = __riscv_vand_vx_u8mf2(v3461, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3486 = __riscv_vmseq_vx_u8mf2_b16(v3485, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3487 = __riscv_vadd_vx_i8mf2_mu(v3486, v3484, v3484, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3488 = __riscv_vwmacc_vx_i16m1(v3410, v3455, v3487, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3489 = v19 + 1480;
      const uint8_t* v3490 = (const uint8_t*) v3489;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3491 = __riscv_vle8_v_u8mf2(v3490, 8);
      const uint8_t* v3492 = v19 + 456;
      const uint8_t* v3493 = (const uint8_t*) v3492;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3494 = __riscv_vle8_v_u8mf2(v3493, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3495 = __riscv_vand_vx_u8mf2(v3491, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3496 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3495);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3497 = __riscv_vand_vx_u8mf2(v3494, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3498 = __riscv_vmseq_vx_u8mf2_b16(v3497, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3499 = __riscv_vadd_vx_i8mf2_mu(v3498, v3496, v3496, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3500 = __riscv_vwmacc_vx_i16m1(v3422, v3446, v3499, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3501 = __riscv_vsrl_vx_u8mf2(v3491, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3502 = __riscv_vand_vx_u8mf2(v3501, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3503 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3502);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3504 = __riscv_vand_vx_u8mf2(v3494, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3505 = __riscv_vmseq_vx_u8mf2_b16(v3504, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3506 = __riscv_vadd_vx_i8mf2_mu(v3505, v3503, v3503, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3507 = __riscv_vwmacc_vx_i16m1(v3429, v3449, v3506, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3508 = __riscv_vsrl_vx_u8mf2(v3491, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3509 = __riscv_vand_vx_u8mf2(v3508, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3510 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3509);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3511 = __riscv_vand_vx_u8mf2(v3494, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3512 = __riscv_vmseq_vx_u8mf2_b16(v3511, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3513 = __riscv_vadd_vx_i8mf2_mu(v3512, v3510, v3510, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3514 = __riscv_vwmacc_vx_i16m1(v3436, v3452, v3513, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3515 = __riscv_vsrl_vx_u8mf2(v3491, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3516 = __riscv_vand_vx_u8mf2(v3515, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3517 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3516);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3518 = __riscv_vand_vx_u8mf2(v3494, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3519 = __riscv_vmseq_vx_u8mf2_b16(v3518, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3520 = __riscv_vadd_vx_i8mf2_mu(v3519, v3517, v3517, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3521 = __riscv_vwmacc_vx_i16m1(v3443, v3455, v3520, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3522 = v21 + 143;
      const int8_t* v3523 = (const int8_t*) v3522;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3524 = *(const int8_t *)(v3523);
      const uint8_t* v3525 = v21 + 175;
      const int8_t* v3526 = (const int8_t*) v3525;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3527 = *(const int8_t *)(v3526);
      const uint8_t* v3528 = v21 + 207;
      const int8_t* v3529 = (const int8_t*) v3528;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3530 = *(const int8_t *)(v3529);
      const uint8_t* v3531 = v21 + 239;
      const int8_t* v3532 = (const int8_t*) v3531;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3533 = *(const int8_t *)(v3532);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3534 = v19 + 1488;
      const uint8_t* v3535 = (const uint8_t*) v3534;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3536 = __riscv_vle8_v_u8mf2(v3535, 8);
      const uint8_t* v3537 = v19 + 464;
      const uint8_t* v3538 = (const uint8_t*) v3537;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3539 = __riscv_vle8_v_u8mf2(v3538, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3540 = __riscv_vand_vx_u8mf2(v3536, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3541 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3540);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3542 = __riscv_vand_vx_u8mf2(v3539, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3543 = __riscv_vmseq_vx_u8mf2_b16(v3542, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3544 = __riscv_vadd_vx_i8mf2_mu(v3543, v3541, v3541, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3545 = __riscv_vwmacc_vx_i16m1(v3467, v3524, v3544, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3546 = __riscv_vsrl_vx_u8mf2(v3536, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3547 = __riscv_vand_vx_u8mf2(v3546, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3548 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3547);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3549 = __riscv_vand_vx_u8mf2(v3539, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3550 = __riscv_vmseq_vx_u8mf2_b16(v3549, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3551 = __riscv_vadd_vx_i8mf2_mu(v3550, v3548, v3548, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3552 = __riscv_vwmacc_vx_i16m1(v3474, v3527, v3551, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3553 = __riscv_vsrl_vx_u8mf2(v3536, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3554 = __riscv_vand_vx_u8mf2(v3553, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3555 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3554);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3556 = __riscv_vand_vx_u8mf2(v3539, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3557 = __riscv_vmseq_vx_u8mf2_b16(v3556, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3558 = __riscv_vadd_vx_i8mf2_mu(v3557, v3555, v3555, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3559 = __riscv_vwmacc_vx_i16m1(v3481, v3530, v3558, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3560 = __riscv_vsrl_vx_u8mf2(v3536, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3561 = __riscv_vand_vx_u8mf2(v3560, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3562 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3561);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3563 = __riscv_vand_vx_u8mf2(v3539, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3564 = __riscv_vmseq_vx_u8mf2_b16(v3563, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3565 = __riscv_vadd_vx_i8mf2_mu(v3564, v3562, v3562, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3566 = __riscv_vwmacc_vx_i16m1(v3488, v3533, v3565, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3567 = v19 + 1496;
      const uint8_t* v3568 = (const uint8_t*) v3567;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3569 = __riscv_vle8_v_u8mf2(v3568, 8);
      const uint8_t* v3570 = v19 + 472;
      const uint8_t* v3571 = (const uint8_t*) v3570;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3572 = __riscv_vle8_v_u8mf2(v3571, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3573 = __riscv_vand_vx_u8mf2(v3569, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3574 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3573);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3575 = __riscv_vand_vx_u8mf2(v3572, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3576 = __riscv_vmseq_vx_u8mf2_b16(v3575, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3577 = __riscv_vadd_vx_i8mf2_mu(v3576, v3574, v3574, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3578 = __riscv_vwmacc_vx_i16m1(v3500, v3524, v3577, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3579 = __riscv_vsrl_vx_u8mf2(v3569, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3580 = __riscv_vand_vx_u8mf2(v3579, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3581 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3580);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3582 = __riscv_vand_vx_u8mf2(v3572, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3583 = __riscv_vmseq_vx_u8mf2_b16(v3582, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3584 = __riscv_vadd_vx_i8mf2_mu(v3583, v3581, v3581, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3585 = __riscv_vwmacc_vx_i16m1(v3507, v3527, v3584, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3586 = __riscv_vsrl_vx_u8mf2(v3569, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3587 = __riscv_vand_vx_u8mf2(v3586, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3588 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3587);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3589 = __riscv_vand_vx_u8mf2(v3572, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3590 = __riscv_vmseq_vx_u8mf2_b16(v3589, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3591 = __riscv_vadd_vx_i8mf2_mu(v3590, v3588, v3588, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3592 = __riscv_vwmacc_vx_i16m1(v3514, v3530, v3591, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3593 = __riscv_vsrl_vx_u8mf2(v3569, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3594 = __riscv_vand_vx_u8mf2(v3593, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3595 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3594);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3596 = __riscv_vand_vx_u8mf2(v3572, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3597 = __riscv_vmseq_vx_u8mf2_b16(v3596, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3598 = __riscv_vadd_vx_i8mf2_mu(v3597, v3595, v3595, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3599 = __riscv_vwmacc_vx_i16m1(v3521, v3533, v3598, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3600 = v21 + 144;
      const int8_t* v3601 = (const int8_t*) v3600;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3602 = *(const int8_t *)(v3601);
      const uint8_t* v3603 = v21 + 176;
      const int8_t* v3604 = (const int8_t*) v3603;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3605 = *(const int8_t *)(v3604);
      const uint8_t* v3606 = v21 + 208;
      const int8_t* v3607 = (const int8_t*) v3606;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3608 = *(const int8_t *)(v3607);
      const uint8_t* v3609 = v21 + 240;
      const int8_t* v3610 = (const int8_t*) v3609;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3611 = *(const int8_t *)(v3610);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3612 = v19 + 1504;
      const uint8_t* v3613 = (const uint8_t*) v3612;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3614 = __riscv_vle8_v_u8mf2(v3613, 8);
      const uint8_t* v3615 = v19 + 480;
      const uint8_t* v3616 = (const uint8_t*) v3615;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3617 = __riscv_vle8_v_u8mf2(v3616, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3618 = __riscv_vand_vx_u8mf2(v3614, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3619 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3618);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3620 = __riscv_vand_vx_u8mf2(v3617, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3621 = __riscv_vmseq_vx_u8mf2_b16(v3620, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3622 = __riscv_vadd_vx_i8mf2_mu(v3621, v3619, v3619, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3623 = __riscv_vwmacc_vx_i16m1(v3545, v3602, v3622, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3624 = __riscv_vsrl_vx_u8mf2(v3614, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3625 = __riscv_vand_vx_u8mf2(v3624, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3626 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3625);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3627 = __riscv_vand_vx_u8mf2(v3617, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3628 = __riscv_vmseq_vx_u8mf2_b16(v3627, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3629 = __riscv_vadd_vx_i8mf2_mu(v3628, v3626, v3626, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3630 = __riscv_vwmacc_vx_i16m1(v3552, v3605, v3629, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3631 = __riscv_vsrl_vx_u8mf2(v3614, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3632 = __riscv_vand_vx_u8mf2(v3631, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3633 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3632);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3634 = __riscv_vand_vx_u8mf2(v3617, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3635 = __riscv_vmseq_vx_u8mf2_b16(v3634, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3636 = __riscv_vadd_vx_i8mf2_mu(v3635, v3633, v3633, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3637 = __riscv_vwmacc_vx_i16m1(v3559, v3608, v3636, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3638 = __riscv_vsrl_vx_u8mf2(v3614, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3639 = __riscv_vand_vx_u8mf2(v3638, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3640 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3639);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3641 = __riscv_vand_vx_u8mf2(v3617, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3642 = __riscv_vmseq_vx_u8mf2_b16(v3641, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3643 = __riscv_vadd_vx_i8mf2_mu(v3642, v3640, v3640, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3644 = __riscv_vwmacc_vx_i16m1(v3566, v3611, v3643, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3645 = v19 + 1512;
      const uint8_t* v3646 = (const uint8_t*) v3645;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3647 = __riscv_vle8_v_u8mf2(v3646, 8);
      const uint8_t* v3648 = v19 + 488;
      const uint8_t* v3649 = (const uint8_t*) v3648;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3650 = __riscv_vle8_v_u8mf2(v3649, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3651 = __riscv_vand_vx_u8mf2(v3647, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3652 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3651);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3653 = __riscv_vand_vx_u8mf2(v3650, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3654 = __riscv_vmseq_vx_u8mf2_b16(v3653, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3655 = __riscv_vadd_vx_i8mf2_mu(v3654, v3652, v3652, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3656 = __riscv_vwmacc_vx_i16m1(v3578, v3602, v3655, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3657 = __riscv_vsrl_vx_u8mf2(v3647, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3658 = __riscv_vand_vx_u8mf2(v3657, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3659 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3658);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3660 = __riscv_vand_vx_u8mf2(v3650, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3661 = __riscv_vmseq_vx_u8mf2_b16(v3660, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3662 = __riscv_vadd_vx_i8mf2_mu(v3661, v3659, v3659, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3663 = __riscv_vwmacc_vx_i16m1(v3585, v3605, v3662, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3664 = __riscv_vsrl_vx_u8mf2(v3647, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3665 = __riscv_vand_vx_u8mf2(v3664, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3666 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3665);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3667 = __riscv_vand_vx_u8mf2(v3650, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3668 = __riscv_vmseq_vx_u8mf2_b16(v3667, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3669 = __riscv_vadd_vx_i8mf2_mu(v3668, v3666, v3666, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3670 = __riscv_vwmacc_vx_i16m1(v3592, v3608, v3669, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3671 = __riscv_vsrl_vx_u8mf2(v3647, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3672 = __riscv_vand_vx_u8mf2(v3671, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3673 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3672);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3674 = __riscv_vand_vx_u8mf2(v3650, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3675 = __riscv_vmseq_vx_u8mf2_b16(v3674, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3676 = __riscv_vadd_vx_i8mf2_mu(v3675, v3673, v3673, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3677 = __riscv_vwmacc_vx_i16m1(v3599, v3611, v3676, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3678 = v21 + 145;
      const int8_t* v3679 = (const int8_t*) v3678;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3680 = *(const int8_t *)(v3679);
      const uint8_t* v3681 = v21 + 177;
      const int8_t* v3682 = (const int8_t*) v3681;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3683 = *(const int8_t *)(v3682);
      const uint8_t* v3684 = v21 + 209;
      const int8_t* v3685 = (const int8_t*) v3684;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3686 = *(const int8_t *)(v3685);
      const uint8_t* v3687 = v21 + 241;
      const int8_t* v3688 = (const int8_t*) v3687;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3689 = *(const int8_t *)(v3688);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3690 = v19 + 1520;
      const uint8_t* v3691 = (const uint8_t*) v3690;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3692 = __riscv_vle8_v_u8mf2(v3691, 8);
      const uint8_t* v3693 = v19 + 496;
      const uint8_t* v3694 = (const uint8_t*) v3693;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3695 = __riscv_vle8_v_u8mf2(v3694, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3696 = __riscv_vand_vx_u8mf2(v3692, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3697 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3696);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3698 = __riscv_vand_vx_u8mf2(v3695, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3699 = __riscv_vmseq_vx_u8mf2_b16(v3698, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3700 = __riscv_vadd_vx_i8mf2_mu(v3699, v3697, v3697, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3701 = __riscv_vwmacc_vx_i16m1(v3623, v3680, v3700, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3702 = __riscv_vsrl_vx_u8mf2(v3692, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3703 = __riscv_vand_vx_u8mf2(v3702, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3704 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3703);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3705 = __riscv_vand_vx_u8mf2(v3695, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3706 = __riscv_vmseq_vx_u8mf2_b16(v3705, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3707 = __riscv_vadd_vx_i8mf2_mu(v3706, v3704, v3704, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3708 = __riscv_vwmacc_vx_i16m1(v3630, v3683, v3707, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3709 = __riscv_vsrl_vx_u8mf2(v3692, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3710 = __riscv_vand_vx_u8mf2(v3709, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3711 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3710);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3712 = __riscv_vand_vx_u8mf2(v3695, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3713 = __riscv_vmseq_vx_u8mf2_b16(v3712, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3714 = __riscv_vadd_vx_i8mf2_mu(v3713, v3711, v3711, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3715 = __riscv_vwmacc_vx_i16m1(v3637, v3686, v3714, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3716 = __riscv_vsrl_vx_u8mf2(v3692, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3717 = __riscv_vand_vx_u8mf2(v3716, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3718 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3717);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3719 = __riscv_vand_vx_u8mf2(v3695, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3720 = __riscv_vmseq_vx_u8mf2_b16(v3719, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3721 = __riscv_vadd_vx_i8mf2_mu(v3720, v3718, v3718, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3722 = __riscv_vwmacc_vx_i16m1(v3644, v3689, v3721, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3723 = v19 + 1528;
      const uint8_t* v3724 = (const uint8_t*) v3723;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3725 = __riscv_vle8_v_u8mf2(v3724, 8);
      const uint8_t* v3726 = v19 + 504;
      const uint8_t* v3727 = (const uint8_t*) v3726;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3728 = __riscv_vle8_v_u8mf2(v3727, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3729 = __riscv_vand_vx_u8mf2(v3725, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3730 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3729);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3731 = __riscv_vand_vx_u8mf2(v3728, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3732 = __riscv_vmseq_vx_u8mf2_b16(v3731, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3733 = __riscv_vadd_vx_i8mf2_mu(v3732, v3730, v3730, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3734 = __riscv_vwmacc_vx_i16m1(v3656, v3680, v3733, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3735 = __riscv_vsrl_vx_u8mf2(v3725, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3736 = __riscv_vand_vx_u8mf2(v3735, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3737 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3736);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3738 = __riscv_vand_vx_u8mf2(v3728, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3739 = __riscv_vmseq_vx_u8mf2_b16(v3738, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3740 = __riscv_vadd_vx_i8mf2_mu(v3739, v3737, v3737, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3741 = __riscv_vwmacc_vx_i16m1(v3663, v3683, v3740, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3742 = __riscv_vsrl_vx_u8mf2(v3725, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3743 = __riscv_vand_vx_u8mf2(v3742, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3744 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3743);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3745 = __riscv_vand_vx_u8mf2(v3728, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3746 = __riscv_vmseq_vx_u8mf2_b16(v3745, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3747 = __riscv_vadd_vx_i8mf2_mu(v3746, v3744, v3744, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3748 = __riscv_vwmacc_vx_i16m1(v3670, v3686, v3747, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3749 = __riscv_vsrl_vx_u8mf2(v3725, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3750 = __riscv_vand_vx_u8mf2(v3749, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3751 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3750);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3752 = __riscv_vand_vx_u8mf2(v3728, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3753 = __riscv_vmseq_vx_u8mf2_b16(v3752, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3754 = __riscv_vadd_vx_i8mf2_mu(v3753, v3751, v3751, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3755 = __riscv_vwmacc_vx_i16m1(v3677, v3689, v3754, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3756 = v21 + 146;
      const int8_t* v3757 = (const int8_t*) v3756;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3758 = *(const int8_t *)(v3757);
      const uint8_t* v3759 = v21 + 178;
      const int8_t* v3760 = (const int8_t*) v3759;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3761 = *(const int8_t *)(v3760);
      const uint8_t* v3762 = v21 + 210;
      const int8_t* v3763 = (const int8_t*) v3762;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3764 = *(const int8_t *)(v3763);
      const uint8_t* v3765 = v21 + 242;
      const int8_t* v3766 = (const int8_t*) v3765;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3767 = *(const int8_t *)(v3766);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3768 = v19 + 1536;
      const uint8_t* v3769 = (const uint8_t*) v3768;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3770 = __riscv_vle8_v_u8mf2(v3769, 8);
      const uint8_t* v3771 = v19 + 512;
      const uint8_t* v3772 = (const uint8_t*) v3771;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3773 = __riscv_vle8_v_u8mf2(v3772, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3774 = __riscv_vand_vx_u8mf2(v3770, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3775 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3774);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3776 = __riscv_vand_vx_u8mf2(v3773, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3777 = __riscv_vmseq_vx_u8mf2_b16(v3776, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3778 = __riscv_vadd_vx_i8mf2_mu(v3777, v3775, v3775, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3779 = __riscv_vwmacc_vx_i16m1(v3701, v3758, v3778, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3780 = __riscv_vsrl_vx_u8mf2(v3770, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3781 = __riscv_vand_vx_u8mf2(v3780, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3782 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3781);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3783 = __riscv_vand_vx_u8mf2(v3773, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3784 = __riscv_vmseq_vx_u8mf2_b16(v3783, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3785 = __riscv_vadd_vx_i8mf2_mu(v3784, v3782, v3782, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3786 = __riscv_vwmacc_vx_i16m1(v3708, v3761, v3785, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3787 = __riscv_vsrl_vx_u8mf2(v3770, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3788 = __riscv_vand_vx_u8mf2(v3787, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3789 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3788);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3790 = __riscv_vand_vx_u8mf2(v3773, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3791 = __riscv_vmseq_vx_u8mf2_b16(v3790, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3792 = __riscv_vadd_vx_i8mf2_mu(v3791, v3789, v3789, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3793 = __riscv_vwmacc_vx_i16m1(v3715, v3764, v3792, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3794 = __riscv_vsrl_vx_u8mf2(v3770, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3795 = __riscv_vand_vx_u8mf2(v3794, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3796 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3795);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3797 = __riscv_vand_vx_u8mf2(v3773, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3798 = __riscv_vmseq_vx_u8mf2_b16(v3797, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3799 = __riscv_vadd_vx_i8mf2_mu(v3798, v3796, v3796, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3800 = __riscv_vwmacc_vx_i16m1(v3722, v3767, v3799, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3801 = v19 + 1544;
      const uint8_t* v3802 = (const uint8_t*) v3801;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3803 = __riscv_vle8_v_u8mf2(v3802, 8);
      const uint8_t* v3804 = v19 + 520;
      const uint8_t* v3805 = (const uint8_t*) v3804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3806 = __riscv_vle8_v_u8mf2(v3805, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3807 = __riscv_vand_vx_u8mf2(v3803, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3808 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3807);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3809 = __riscv_vand_vx_u8mf2(v3806, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3810 = __riscv_vmseq_vx_u8mf2_b16(v3809, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3811 = __riscv_vadd_vx_i8mf2_mu(v3810, v3808, v3808, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3812 = __riscv_vwmacc_vx_i16m1(v3734, v3758, v3811, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3813 = __riscv_vsrl_vx_u8mf2(v3803, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3814 = __riscv_vand_vx_u8mf2(v3813, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3815 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3814);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3816 = __riscv_vand_vx_u8mf2(v3806, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3817 = __riscv_vmseq_vx_u8mf2_b16(v3816, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3818 = __riscv_vadd_vx_i8mf2_mu(v3817, v3815, v3815, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3819 = __riscv_vwmacc_vx_i16m1(v3741, v3761, v3818, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3820 = __riscv_vsrl_vx_u8mf2(v3803, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3821 = __riscv_vand_vx_u8mf2(v3820, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3822 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3821);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3823 = __riscv_vand_vx_u8mf2(v3806, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3824 = __riscv_vmseq_vx_u8mf2_b16(v3823, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3825 = __riscv_vadd_vx_i8mf2_mu(v3824, v3822, v3822, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3826 = __riscv_vwmacc_vx_i16m1(v3748, v3764, v3825, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3827 = __riscv_vsrl_vx_u8mf2(v3803, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3828 = __riscv_vand_vx_u8mf2(v3827, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3829 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3828);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3830 = __riscv_vand_vx_u8mf2(v3806, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3831 = __riscv_vmseq_vx_u8mf2_b16(v3830, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3832 = __riscv_vadd_vx_i8mf2_mu(v3831, v3829, v3829, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3833 = __riscv_vwmacc_vx_i16m1(v3755, v3767, v3832, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3834 = v21 + 147;
      const int8_t* v3835 = (const int8_t*) v3834;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3836 = *(const int8_t *)(v3835);
      const uint8_t* v3837 = v21 + 179;
      const int8_t* v3838 = (const int8_t*) v3837;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3839 = *(const int8_t *)(v3838);
      const uint8_t* v3840 = v21 + 211;
      const int8_t* v3841 = (const int8_t*) v3840;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3842 = *(const int8_t *)(v3841);
      const uint8_t* v3843 = v21 + 243;
      const int8_t* v3844 = (const int8_t*) v3843;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3845 = *(const int8_t *)(v3844);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3846 = v19 + 1552;
      const uint8_t* v3847 = (const uint8_t*) v3846;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3848 = __riscv_vle8_v_u8mf2(v3847, 8);
      const uint8_t* v3849 = v19 + 528;
      const uint8_t* v3850 = (const uint8_t*) v3849;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3851 = __riscv_vle8_v_u8mf2(v3850, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3852 = __riscv_vand_vx_u8mf2(v3848, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3853 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3852);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3854 = __riscv_vand_vx_u8mf2(v3851, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3855 = __riscv_vmseq_vx_u8mf2_b16(v3854, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3856 = __riscv_vadd_vx_i8mf2_mu(v3855, v3853, v3853, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3857 = __riscv_vwmacc_vx_i16m1(v3779, v3836, v3856, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3858 = __riscv_vsrl_vx_u8mf2(v3848, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3859 = __riscv_vand_vx_u8mf2(v3858, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3860 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3859);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3861 = __riscv_vand_vx_u8mf2(v3851, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3862 = __riscv_vmseq_vx_u8mf2_b16(v3861, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3863 = __riscv_vadd_vx_i8mf2_mu(v3862, v3860, v3860, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3864 = __riscv_vwmacc_vx_i16m1(v3786, v3839, v3863, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3865 = __riscv_vsrl_vx_u8mf2(v3848, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3866 = __riscv_vand_vx_u8mf2(v3865, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3867 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3866);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3868 = __riscv_vand_vx_u8mf2(v3851, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3869 = __riscv_vmseq_vx_u8mf2_b16(v3868, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3870 = __riscv_vadd_vx_i8mf2_mu(v3869, v3867, v3867, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3871 = __riscv_vwmacc_vx_i16m1(v3793, v3842, v3870, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3872 = __riscv_vsrl_vx_u8mf2(v3848, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3873 = __riscv_vand_vx_u8mf2(v3872, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3874 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3873);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3875 = __riscv_vand_vx_u8mf2(v3851, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3876 = __riscv_vmseq_vx_u8mf2_b16(v3875, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3877 = __riscv_vadd_vx_i8mf2_mu(v3876, v3874, v3874, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3878 = __riscv_vwmacc_vx_i16m1(v3800, v3845, v3877, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3879 = v19 + 1560;
      const uint8_t* v3880 = (const uint8_t*) v3879;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3881 = __riscv_vle8_v_u8mf2(v3880, 8);
      const uint8_t* v3882 = v19 + 536;
      const uint8_t* v3883 = (const uint8_t*) v3882;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3884 = __riscv_vle8_v_u8mf2(v3883, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3885 = __riscv_vand_vx_u8mf2(v3881, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3886 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3885);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3887 = __riscv_vand_vx_u8mf2(v3884, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3888 = __riscv_vmseq_vx_u8mf2_b16(v3887, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3889 = __riscv_vadd_vx_i8mf2_mu(v3888, v3886, v3886, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3890 = __riscv_vwmacc_vx_i16m1(v3812, v3836, v3889, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3891 = __riscv_vsrl_vx_u8mf2(v3881, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3892 = __riscv_vand_vx_u8mf2(v3891, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3893 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3892);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3894 = __riscv_vand_vx_u8mf2(v3884, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3895 = __riscv_vmseq_vx_u8mf2_b16(v3894, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3896 = __riscv_vadd_vx_i8mf2_mu(v3895, v3893, v3893, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3897 = __riscv_vwmacc_vx_i16m1(v3819, v3839, v3896, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3898 = __riscv_vsrl_vx_u8mf2(v3881, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3899 = __riscv_vand_vx_u8mf2(v3898, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3900 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3899);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3901 = __riscv_vand_vx_u8mf2(v3884, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3902 = __riscv_vmseq_vx_u8mf2_b16(v3901, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3903 = __riscv_vadd_vx_i8mf2_mu(v3902, v3900, v3900, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3904 = __riscv_vwmacc_vx_i16m1(v3826, v3842, v3903, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3905 = __riscv_vsrl_vx_u8mf2(v3881, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3906 = __riscv_vand_vx_u8mf2(v3905, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3907 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3906);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3908 = __riscv_vand_vx_u8mf2(v3884, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3909 = __riscv_vmseq_vx_u8mf2_b16(v3908, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3910 = __riscv_vadd_vx_i8mf2_mu(v3909, v3907, v3907, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3911 = __riscv_vwmacc_vx_i16m1(v3833, v3845, v3910, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v3912 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3913 = __riscv_vwmacc_vv_i32m2(v3912, v2627, v3857, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3914 = __riscv_vwmacc_vv_i32m2(v3913, v2631, v3864, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3915 = __riscv_vwmacc_vv_i32m2(v3914, v2635, v3871, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3916 = __riscv_vwmacc_vv_i32m2(v3915, v2639, v3878, 8);
      v24 = v3916;
      vint32m2_t v3917 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3918 = __riscv_vwmacc_vv_i32m2(v3917, v2643, v3890, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3919 = __riscv_vwmacc_vv_i32m2(v3918, v2647, v3897, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3920 = __riscv_vwmacc_vv_i32m2(v3919, v2651, v3904, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3921 = __riscv_vwmacc_vv_i32m2(v3920, v2655, v3911, 8);
      v26 = v3921;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
      const uint8_t* v3922 = v19 + 176;
      const int8_t* v3923 = (const int8_t*) v3922;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v3924 = __riscv_vle8_v_i8mf2(v3923, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v3925 = __riscv_vsext_vf2_i16m1(v3924, 8);
      const uint8_t* v3926 = v19 + 208;
      const int8_t* v3927 = (const int8_t*) v3926;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v3928 = __riscv_vle8_v_i8mf2(v3927, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v3929 = __riscv_vsext_vf2_i16m1(v3928, 8);
      const uint8_t* v3930 = v19 + 240;
      const int8_t* v3931 = (const int8_t*) v3930;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v3932 = __riscv_vle8_v_i8mf2(v3931, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v3933 = __riscv_vsext_vf2_i16m1(v3932, 8);
      const uint8_t* v3934 = v19 + 272;
      const int8_t* v3935 = (const int8_t*) v3934;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v3936 = __riscv_vle8_v_i8mf2(v3935, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v3937 = __riscv_vsext_vf2_i16m1(v3936, 8);
      const uint8_t* v3938 = v19 + 184;
      const int8_t* v3939 = (const int8_t*) v3938;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v3940 = __riscv_vle8_v_i8mf2(v3939, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v3941 = __riscv_vsext_vf2_i16m1(v3940, 8);
      const uint8_t* v3942 = v19 + 216;
      const int8_t* v3943 = (const int8_t*) v3942;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v3944 = __riscv_vle8_v_i8mf2(v3943, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v3945 = __riscv_vsext_vf2_i16m1(v3944, 8);
      const uint8_t* v3946 = v19 + 248;
      const int8_t* v3947 = (const int8_t*) v3946;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v3948 = __riscv_vle8_v_i8mf2(v3947, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v3949 = __riscv_vsext_vf2_i16m1(v3948, 8);
      const uint8_t* v3950 = v19 + 280;
      const int8_t* v3951 = (const int8_t*) v3950;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v3952 = __riscv_vle8_v_i8mf2(v3951, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v3953 = __riscv_vsext_vf2_i16m1(v3952, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3954 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3955 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3956 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3957 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3958 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3959 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3960 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3961 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3962 = v21 + 148;
      const int8_t* v3963 = (const int8_t*) v3962;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3964 = *(const int8_t *)(v3963);
      const uint8_t* v3965 = v21 + 180;
      const int8_t* v3966 = (const int8_t*) v3965;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3967 = *(const int8_t *)(v3966);
      const uint8_t* v3968 = v21 + 212;
      const int8_t* v3969 = (const int8_t*) v3968;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3970 = *(const int8_t *)(v3969);
      const uint8_t* v3971 = v21 + 244;
      const int8_t* v3972 = (const int8_t*) v3971;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3973 = *(const int8_t *)(v3972);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v3974 = v19 + 1568;
      const uint8_t* v3975 = (const uint8_t*) v3974;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3976 = __riscv_vle8_v_u8mf2(v3975, 8);
      const uint8_t* v3977 = v19 + 544;
      const uint8_t* v3978 = (const uint8_t*) v3977;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3979 = __riscv_vle8_v_u8mf2(v3978, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3980 = __riscv_vand_vx_u8mf2(v3976, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3981 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3980);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3982 = __riscv_vand_vx_u8mf2(v3979, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3983 = __riscv_vmseq_vx_u8mf2_b16(v3982, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3984 = __riscv_vadd_vx_i8mf2_mu(v3983, v3981, v3981, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3985 = __riscv_vwmacc_vx_i16m1(v3954, v3964, v3984, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3986 = __riscv_vsrl_vx_u8mf2(v3976, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3987 = __riscv_vand_vx_u8mf2(v3986, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3988 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3987);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3989 = __riscv_vand_vx_u8mf2(v3979, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3990 = __riscv_vmseq_vx_u8mf2_b16(v3989, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3991 = __riscv_vadd_vx_i8mf2_mu(v3990, v3988, v3988, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3992 = __riscv_vwmacc_vx_i16m1(v3955, v3967, v3991, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3993 = __riscv_vsrl_vx_u8mf2(v3976, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3994 = __riscv_vand_vx_u8mf2(v3993, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3995 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3994);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3996 = __riscv_vand_vx_u8mf2(v3979, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v3997 = __riscv_vmseq_vx_u8mf2_b16(v3996, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v3998 = __riscv_vadd_vx_i8mf2_mu(v3997, v3995, v3995, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3999 = __riscv_vwmacc_vx_i16m1(v3956, v3970, v3998, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4000 = __riscv_vsrl_vx_u8mf2(v3976, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4001 = __riscv_vand_vx_u8mf2(v4000, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4002 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4001);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4003 = __riscv_vand_vx_u8mf2(v3979, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4004 = __riscv_vmseq_vx_u8mf2_b16(v4003, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4005 = __riscv_vadd_vx_i8mf2_mu(v4004, v4002, v4002, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4006 = __riscv_vwmacc_vx_i16m1(v3957, v3973, v4005, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4007 = v19 + 1576;
      const uint8_t* v4008 = (const uint8_t*) v4007;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4009 = __riscv_vle8_v_u8mf2(v4008, 8);
      const uint8_t* v4010 = v19 + 552;
      const uint8_t* v4011 = (const uint8_t*) v4010;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4012 = __riscv_vle8_v_u8mf2(v4011, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4013 = __riscv_vand_vx_u8mf2(v4009, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4014 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4013);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4015 = __riscv_vand_vx_u8mf2(v4012, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4016 = __riscv_vmseq_vx_u8mf2_b16(v4015, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4017 = __riscv_vadd_vx_i8mf2_mu(v4016, v4014, v4014, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4018 = __riscv_vwmacc_vx_i16m1(v3958, v3964, v4017, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4019 = __riscv_vsrl_vx_u8mf2(v4009, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4020 = __riscv_vand_vx_u8mf2(v4019, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4021 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4020);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4022 = __riscv_vand_vx_u8mf2(v4012, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4023 = __riscv_vmseq_vx_u8mf2_b16(v4022, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4024 = __riscv_vadd_vx_i8mf2_mu(v4023, v4021, v4021, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4025 = __riscv_vwmacc_vx_i16m1(v3959, v3967, v4024, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4026 = __riscv_vsrl_vx_u8mf2(v4009, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4027 = __riscv_vand_vx_u8mf2(v4026, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4028 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4027);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4029 = __riscv_vand_vx_u8mf2(v4012, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4030 = __riscv_vmseq_vx_u8mf2_b16(v4029, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4031 = __riscv_vadd_vx_i8mf2_mu(v4030, v4028, v4028, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4032 = __riscv_vwmacc_vx_i16m1(v3960, v3970, v4031, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4033 = __riscv_vsrl_vx_u8mf2(v4009, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4034 = __riscv_vand_vx_u8mf2(v4033, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4035 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4034);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4036 = __riscv_vand_vx_u8mf2(v4012, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4037 = __riscv_vmseq_vx_u8mf2_b16(v4036, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4038 = __riscv_vadd_vx_i8mf2_mu(v4037, v4035, v4035, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4039 = __riscv_vwmacc_vx_i16m1(v3961, v3973, v4038, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4040 = v21 + 149;
      const int8_t* v4041 = (const int8_t*) v4040;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4042 = *(const int8_t *)(v4041);
      const uint8_t* v4043 = v21 + 181;
      const int8_t* v4044 = (const int8_t*) v4043;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4045 = *(const int8_t *)(v4044);
      const uint8_t* v4046 = v21 + 213;
      const int8_t* v4047 = (const int8_t*) v4046;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4048 = *(const int8_t *)(v4047);
      const uint8_t* v4049 = v21 + 245;
      const int8_t* v4050 = (const int8_t*) v4049;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4051 = *(const int8_t *)(v4050);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4052 = v19 + 1584;
      const uint8_t* v4053 = (const uint8_t*) v4052;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4054 = __riscv_vle8_v_u8mf2(v4053, 8);
      const uint8_t* v4055 = v19 + 560;
      const uint8_t* v4056 = (const uint8_t*) v4055;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4057 = __riscv_vle8_v_u8mf2(v4056, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4058 = __riscv_vand_vx_u8mf2(v4054, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4059 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4058);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4060 = __riscv_vand_vx_u8mf2(v4057, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4061 = __riscv_vmseq_vx_u8mf2_b16(v4060, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4062 = __riscv_vadd_vx_i8mf2_mu(v4061, v4059, v4059, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4063 = __riscv_vwmacc_vx_i16m1(v3985, v4042, v4062, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4064 = __riscv_vsrl_vx_u8mf2(v4054, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4065 = __riscv_vand_vx_u8mf2(v4064, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4066 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4065);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4067 = __riscv_vand_vx_u8mf2(v4057, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4068 = __riscv_vmseq_vx_u8mf2_b16(v4067, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4069 = __riscv_vadd_vx_i8mf2_mu(v4068, v4066, v4066, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4070 = __riscv_vwmacc_vx_i16m1(v3992, v4045, v4069, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4071 = __riscv_vsrl_vx_u8mf2(v4054, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4072 = __riscv_vand_vx_u8mf2(v4071, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4073 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4072);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4074 = __riscv_vand_vx_u8mf2(v4057, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4075 = __riscv_vmseq_vx_u8mf2_b16(v4074, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4076 = __riscv_vadd_vx_i8mf2_mu(v4075, v4073, v4073, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4077 = __riscv_vwmacc_vx_i16m1(v3999, v4048, v4076, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4078 = __riscv_vsrl_vx_u8mf2(v4054, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4079 = __riscv_vand_vx_u8mf2(v4078, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4080 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4079);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4081 = __riscv_vand_vx_u8mf2(v4057, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4082 = __riscv_vmseq_vx_u8mf2_b16(v4081, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4083 = __riscv_vadd_vx_i8mf2_mu(v4082, v4080, v4080, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4084 = __riscv_vwmacc_vx_i16m1(v4006, v4051, v4083, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4085 = v19 + 1592;
      const uint8_t* v4086 = (const uint8_t*) v4085;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4087 = __riscv_vle8_v_u8mf2(v4086, 8);
      const uint8_t* v4088 = v19 + 568;
      const uint8_t* v4089 = (const uint8_t*) v4088;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4090 = __riscv_vle8_v_u8mf2(v4089, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4091 = __riscv_vand_vx_u8mf2(v4087, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4092 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4091);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4093 = __riscv_vand_vx_u8mf2(v4090, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4094 = __riscv_vmseq_vx_u8mf2_b16(v4093, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4095 = __riscv_vadd_vx_i8mf2_mu(v4094, v4092, v4092, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4096 = __riscv_vwmacc_vx_i16m1(v4018, v4042, v4095, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4097 = __riscv_vsrl_vx_u8mf2(v4087, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4098 = __riscv_vand_vx_u8mf2(v4097, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4099 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4098);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4100 = __riscv_vand_vx_u8mf2(v4090, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4101 = __riscv_vmseq_vx_u8mf2_b16(v4100, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4102 = __riscv_vadd_vx_i8mf2_mu(v4101, v4099, v4099, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4103 = __riscv_vwmacc_vx_i16m1(v4025, v4045, v4102, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4104 = __riscv_vsrl_vx_u8mf2(v4087, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4105 = __riscv_vand_vx_u8mf2(v4104, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4106 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4105);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4107 = __riscv_vand_vx_u8mf2(v4090, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4108 = __riscv_vmseq_vx_u8mf2_b16(v4107, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4109 = __riscv_vadd_vx_i8mf2_mu(v4108, v4106, v4106, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4110 = __riscv_vwmacc_vx_i16m1(v4032, v4048, v4109, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4111 = __riscv_vsrl_vx_u8mf2(v4087, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4112 = __riscv_vand_vx_u8mf2(v4111, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4113 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4112);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4114 = __riscv_vand_vx_u8mf2(v4090, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4115 = __riscv_vmseq_vx_u8mf2_b16(v4114, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4116 = __riscv_vadd_vx_i8mf2_mu(v4115, v4113, v4113, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4117 = __riscv_vwmacc_vx_i16m1(v4039, v4051, v4116, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4118 = v21 + 150;
      const int8_t* v4119 = (const int8_t*) v4118;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4120 = *(const int8_t *)(v4119);
      const uint8_t* v4121 = v21 + 182;
      const int8_t* v4122 = (const int8_t*) v4121;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4123 = *(const int8_t *)(v4122);
      const uint8_t* v4124 = v21 + 214;
      const int8_t* v4125 = (const int8_t*) v4124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4126 = *(const int8_t *)(v4125);
      const uint8_t* v4127 = v21 + 246;
      const int8_t* v4128 = (const int8_t*) v4127;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4129 = *(const int8_t *)(v4128);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4130 = v19 + 1600;
      const uint8_t* v4131 = (const uint8_t*) v4130;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4132 = __riscv_vle8_v_u8mf2(v4131, 8);
      const uint8_t* v4133 = v19 + 576;
      const uint8_t* v4134 = (const uint8_t*) v4133;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4135 = __riscv_vle8_v_u8mf2(v4134, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4136 = __riscv_vand_vx_u8mf2(v4132, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4137 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4136);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4138 = __riscv_vand_vx_u8mf2(v4135, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4139 = __riscv_vmseq_vx_u8mf2_b16(v4138, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4140 = __riscv_vadd_vx_i8mf2_mu(v4139, v4137, v4137, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4141 = __riscv_vwmacc_vx_i16m1(v4063, v4120, v4140, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4142 = __riscv_vsrl_vx_u8mf2(v4132, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4143 = __riscv_vand_vx_u8mf2(v4142, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4144 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4143);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4145 = __riscv_vand_vx_u8mf2(v4135, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4146 = __riscv_vmseq_vx_u8mf2_b16(v4145, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4147 = __riscv_vadd_vx_i8mf2_mu(v4146, v4144, v4144, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4148 = __riscv_vwmacc_vx_i16m1(v4070, v4123, v4147, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4149 = __riscv_vsrl_vx_u8mf2(v4132, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4150 = __riscv_vand_vx_u8mf2(v4149, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4151 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4150);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4152 = __riscv_vand_vx_u8mf2(v4135, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4153 = __riscv_vmseq_vx_u8mf2_b16(v4152, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4154 = __riscv_vadd_vx_i8mf2_mu(v4153, v4151, v4151, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4155 = __riscv_vwmacc_vx_i16m1(v4077, v4126, v4154, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4156 = __riscv_vsrl_vx_u8mf2(v4132, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4157 = __riscv_vand_vx_u8mf2(v4156, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4158 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4157);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4159 = __riscv_vand_vx_u8mf2(v4135, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4160 = __riscv_vmseq_vx_u8mf2_b16(v4159, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4161 = __riscv_vadd_vx_i8mf2_mu(v4160, v4158, v4158, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4162 = __riscv_vwmacc_vx_i16m1(v4084, v4129, v4161, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4163 = v19 + 1608;
      const uint8_t* v4164 = (const uint8_t*) v4163;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4165 = __riscv_vle8_v_u8mf2(v4164, 8);
      const uint8_t* v4166 = v19 + 584;
      const uint8_t* v4167 = (const uint8_t*) v4166;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4168 = __riscv_vle8_v_u8mf2(v4167, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4169 = __riscv_vand_vx_u8mf2(v4165, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4170 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4169);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4171 = __riscv_vand_vx_u8mf2(v4168, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4172 = __riscv_vmseq_vx_u8mf2_b16(v4171, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4173 = __riscv_vadd_vx_i8mf2_mu(v4172, v4170, v4170, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4174 = __riscv_vwmacc_vx_i16m1(v4096, v4120, v4173, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4175 = __riscv_vsrl_vx_u8mf2(v4165, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4176 = __riscv_vand_vx_u8mf2(v4175, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4177 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4176);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4178 = __riscv_vand_vx_u8mf2(v4168, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4179 = __riscv_vmseq_vx_u8mf2_b16(v4178, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4180 = __riscv_vadd_vx_i8mf2_mu(v4179, v4177, v4177, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4181 = __riscv_vwmacc_vx_i16m1(v4103, v4123, v4180, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4182 = __riscv_vsrl_vx_u8mf2(v4165, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4183 = __riscv_vand_vx_u8mf2(v4182, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4184 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4183);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4185 = __riscv_vand_vx_u8mf2(v4168, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4186 = __riscv_vmseq_vx_u8mf2_b16(v4185, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4187 = __riscv_vadd_vx_i8mf2_mu(v4186, v4184, v4184, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4188 = __riscv_vwmacc_vx_i16m1(v4110, v4126, v4187, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4189 = __riscv_vsrl_vx_u8mf2(v4165, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4190 = __riscv_vand_vx_u8mf2(v4189, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4191 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4190);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4192 = __riscv_vand_vx_u8mf2(v4168, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4193 = __riscv_vmseq_vx_u8mf2_b16(v4192, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4194 = __riscv_vadd_vx_i8mf2_mu(v4193, v4191, v4191, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4195 = __riscv_vwmacc_vx_i16m1(v4117, v4129, v4194, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4196 = v21 + 151;
      const int8_t* v4197 = (const int8_t*) v4196;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4198 = *(const int8_t *)(v4197);
      const uint8_t* v4199 = v21 + 183;
      const int8_t* v4200 = (const int8_t*) v4199;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4201 = *(const int8_t *)(v4200);
      const uint8_t* v4202 = v21 + 215;
      const int8_t* v4203 = (const int8_t*) v4202;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4204 = *(const int8_t *)(v4203);
      const uint8_t* v4205 = v21 + 247;
      const int8_t* v4206 = (const int8_t*) v4205;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4207 = *(const int8_t *)(v4206);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4208 = v19 + 1616;
      const uint8_t* v4209 = (const uint8_t*) v4208;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4210 = __riscv_vle8_v_u8mf2(v4209, 8);
      const uint8_t* v4211 = v19 + 592;
      const uint8_t* v4212 = (const uint8_t*) v4211;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4213 = __riscv_vle8_v_u8mf2(v4212, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4214 = __riscv_vand_vx_u8mf2(v4210, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4215 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4214);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4216 = __riscv_vand_vx_u8mf2(v4213, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4217 = __riscv_vmseq_vx_u8mf2_b16(v4216, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4218 = __riscv_vadd_vx_i8mf2_mu(v4217, v4215, v4215, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4219 = __riscv_vwmacc_vx_i16m1(v4141, v4198, v4218, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4220 = __riscv_vsrl_vx_u8mf2(v4210, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4221 = __riscv_vand_vx_u8mf2(v4220, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4222 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4221);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4223 = __riscv_vand_vx_u8mf2(v4213, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4224 = __riscv_vmseq_vx_u8mf2_b16(v4223, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4225 = __riscv_vadd_vx_i8mf2_mu(v4224, v4222, v4222, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4226 = __riscv_vwmacc_vx_i16m1(v4148, v4201, v4225, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4227 = __riscv_vsrl_vx_u8mf2(v4210, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4228 = __riscv_vand_vx_u8mf2(v4227, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4229 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4228);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4230 = __riscv_vand_vx_u8mf2(v4213, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4231 = __riscv_vmseq_vx_u8mf2_b16(v4230, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4232 = __riscv_vadd_vx_i8mf2_mu(v4231, v4229, v4229, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4233 = __riscv_vwmacc_vx_i16m1(v4155, v4204, v4232, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4234 = __riscv_vsrl_vx_u8mf2(v4210, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4235 = __riscv_vand_vx_u8mf2(v4234, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4236 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4235);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4237 = __riscv_vand_vx_u8mf2(v4213, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4238 = __riscv_vmseq_vx_u8mf2_b16(v4237, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4239 = __riscv_vadd_vx_i8mf2_mu(v4238, v4236, v4236, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4240 = __riscv_vwmacc_vx_i16m1(v4162, v4207, v4239, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4241 = v19 + 1624;
      const uint8_t* v4242 = (const uint8_t*) v4241;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4243 = __riscv_vle8_v_u8mf2(v4242, 8);
      const uint8_t* v4244 = v19 + 600;
      const uint8_t* v4245 = (const uint8_t*) v4244;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4246 = __riscv_vle8_v_u8mf2(v4245, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4247 = __riscv_vand_vx_u8mf2(v4243, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4248 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4247);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4249 = __riscv_vand_vx_u8mf2(v4246, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4250 = __riscv_vmseq_vx_u8mf2_b16(v4249, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4251 = __riscv_vadd_vx_i8mf2_mu(v4250, v4248, v4248, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4252 = __riscv_vwmacc_vx_i16m1(v4174, v4198, v4251, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4253 = __riscv_vsrl_vx_u8mf2(v4243, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4254 = __riscv_vand_vx_u8mf2(v4253, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4255 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4254);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4256 = __riscv_vand_vx_u8mf2(v4246, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4257 = __riscv_vmseq_vx_u8mf2_b16(v4256, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4258 = __riscv_vadd_vx_i8mf2_mu(v4257, v4255, v4255, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4259 = __riscv_vwmacc_vx_i16m1(v4181, v4201, v4258, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4260 = __riscv_vsrl_vx_u8mf2(v4243, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4261 = __riscv_vand_vx_u8mf2(v4260, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4262 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4261);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4263 = __riscv_vand_vx_u8mf2(v4246, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4264 = __riscv_vmseq_vx_u8mf2_b16(v4263, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4265 = __riscv_vadd_vx_i8mf2_mu(v4264, v4262, v4262, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4266 = __riscv_vwmacc_vx_i16m1(v4188, v4204, v4265, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4267 = __riscv_vsrl_vx_u8mf2(v4243, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4268 = __riscv_vand_vx_u8mf2(v4267, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4269 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4268);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4270 = __riscv_vand_vx_u8mf2(v4246, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4271 = __riscv_vmseq_vx_u8mf2_b16(v4270, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4272 = __riscv_vadd_vx_i8mf2_mu(v4271, v4269, v4269, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4273 = __riscv_vwmacc_vx_i16m1(v4195, v4207, v4272, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4274 = v21 + 152;
      const int8_t* v4275 = (const int8_t*) v4274;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4276 = *(const int8_t *)(v4275);
      const uint8_t* v4277 = v21 + 184;
      const int8_t* v4278 = (const int8_t*) v4277;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4279 = *(const int8_t *)(v4278);
      const uint8_t* v4280 = v21 + 216;
      const int8_t* v4281 = (const int8_t*) v4280;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4282 = *(const int8_t *)(v4281);
      const uint8_t* v4283 = v21 + 248;
      const int8_t* v4284 = (const int8_t*) v4283;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4285 = *(const int8_t *)(v4284);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4286 = v19 + 1632;
      const uint8_t* v4287 = (const uint8_t*) v4286;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4288 = __riscv_vle8_v_u8mf2(v4287, 8);
      const uint8_t* v4289 = v19 + 608;
      const uint8_t* v4290 = (const uint8_t*) v4289;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4291 = __riscv_vle8_v_u8mf2(v4290, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4292 = __riscv_vand_vx_u8mf2(v4288, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4293 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4292);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4294 = __riscv_vand_vx_u8mf2(v4291, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4295 = __riscv_vmseq_vx_u8mf2_b16(v4294, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4296 = __riscv_vadd_vx_i8mf2_mu(v4295, v4293, v4293, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4297 = __riscv_vwmacc_vx_i16m1(v4219, v4276, v4296, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4298 = __riscv_vsrl_vx_u8mf2(v4288, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4299 = __riscv_vand_vx_u8mf2(v4298, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4300 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4299);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4301 = __riscv_vand_vx_u8mf2(v4291, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4302 = __riscv_vmseq_vx_u8mf2_b16(v4301, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4303 = __riscv_vadd_vx_i8mf2_mu(v4302, v4300, v4300, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4304 = __riscv_vwmacc_vx_i16m1(v4226, v4279, v4303, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4305 = __riscv_vsrl_vx_u8mf2(v4288, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4306 = __riscv_vand_vx_u8mf2(v4305, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4307 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4306);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4308 = __riscv_vand_vx_u8mf2(v4291, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4309 = __riscv_vmseq_vx_u8mf2_b16(v4308, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4310 = __riscv_vadd_vx_i8mf2_mu(v4309, v4307, v4307, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4311 = __riscv_vwmacc_vx_i16m1(v4233, v4282, v4310, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4312 = __riscv_vsrl_vx_u8mf2(v4288, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4313 = __riscv_vand_vx_u8mf2(v4312, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4314 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4313);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4315 = __riscv_vand_vx_u8mf2(v4291, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4316 = __riscv_vmseq_vx_u8mf2_b16(v4315, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4317 = __riscv_vadd_vx_i8mf2_mu(v4316, v4314, v4314, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4318 = __riscv_vwmacc_vx_i16m1(v4240, v4285, v4317, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4319 = v19 + 1640;
      const uint8_t* v4320 = (const uint8_t*) v4319;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4321 = __riscv_vle8_v_u8mf2(v4320, 8);
      const uint8_t* v4322 = v19 + 616;
      const uint8_t* v4323 = (const uint8_t*) v4322;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4324 = __riscv_vle8_v_u8mf2(v4323, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4325 = __riscv_vand_vx_u8mf2(v4321, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4326 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4325);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4327 = __riscv_vand_vx_u8mf2(v4324, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4328 = __riscv_vmseq_vx_u8mf2_b16(v4327, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4329 = __riscv_vadd_vx_i8mf2_mu(v4328, v4326, v4326, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4330 = __riscv_vwmacc_vx_i16m1(v4252, v4276, v4329, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4331 = __riscv_vsrl_vx_u8mf2(v4321, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4332 = __riscv_vand_vx_u8mf2(v4331, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4333 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4332);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4334 = __riscv_vand_vx_u8mf2(v4324, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4335 = __riscv_vmseq_vx_u8mf2_b16(v4334, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4336 = __riscv_vadd_vx_i8mf2_mu(v4335, v4333, v4333, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4337 = __riscv_vwmacc_vx_i16m1(v4259, v4279, v4336, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4338 = __riscv_vsrl_vx_u8mf2(v4321, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4339 = __riscv_vand_vx_u8mf2(v4338, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4340 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4339);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4341 = __riscv_vand_vx_u8mf2(v4324, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4342 = __riscv_vmseq_vx_u8mf2_b16(v4341, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4343 = __riscv_vadd_vx_i8mf2_mu(v4342, v4340, v4340, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4344 = __riscv_vwmacc_vx_i16m1(v4266, v4282, v4343, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4345 = __riscv_vsrl_vx_u8mf2(v4321, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4346 = __riscv_vand_vx_u8mf2(v4345, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4347 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4346);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4348 = __riscv_vand_vx_u8mf2(v4324, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4349 = __riscv_vmseq_vx_u8mf2_b16(v4348, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4350 = __riscv_vadd_vx_i8mf2_mu(v4349, v4347, v4347, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4351 = __riscv_vwmacc_vx_i16m1(v4273, v4285, v4350, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4352 = v21 + 153;
      const int8_t* v4353 = (const int8_t*) v4352;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4354 = *(const int8_t *)(v4353);
      const uint8_t* v4355 = v21 + 185;
      const int8_t* v4356 = (const int8_t*) v4355;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4357 = *(const int8_t *)(v4356);
      const uint8_t* v4358 = v21 + 217;
      const int8_t* v4359 = (const int8_t*) v4358;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4360 = *(const int8_t *)(v4359);
      const uint8_t* v4361 = v21 + 249;
      const int8_t* v4362 = (const int8_t*) v4361;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4363 = *(const int8_t *)(v4362);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4364 = v19 + 1648;
      const uint8_t* v4365 = (const uint8_t*) v4364;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4366 = __riscv_vle8_v_u8mf2(v4365, 8);
      const uint8_t* v4367 = v19 + 624;
      const uint8_t* v4368 = (const uint8_t*) v4367;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4369 = __riscv_vle8_v_u8mf2(v4368, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4370 = __riscv_vand_vx_u8mf2(v4366, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4371 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4370);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4372 = __riscv_vand_vx_u8mf2(v4369, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4373 = __riscv_vmseq_vx_u8mf2_b16(v4372, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4374 = __riscv_vadd_vx_i8mf2_mu(v4373, v4371, v4371, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4375 = __riscv_vwmacc_vx_i16m1(v4297, v4354, v4374, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4376 = __riscv_vsrl_vx_u8mf2(v4366, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4377 = __riscv_vand_vx_u8mf2(v4376, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4378 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4377);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4379 = __riscv_vand_vx_u8mf2(v4369, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4380 = __riscv_vmseq_vx_u8mf2_b16(v4379, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4381 = __riscv_vadd_vx_i8mf2_mu(v4380, v4378, v4378, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4382 = __riscv_vwmacc_vx_i16m1(v4304, v4357, v4381, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4383 = __riscv_vsrl_vx_u8mf2(v4366, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4384 = __riscv_vand_vx_u8mf2(v4383, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4385 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4384);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4386 = __riscv_vand_vx_u8mf2(v4369, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4387 = __riscv_vmseq_vx_u8mf2_b16(v4386, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4388 = __riscv_vadd_vx_i8mf2_mu(v4387, v4385, v4385, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4389 = __riscv_vwmacc_vx_i16m1(v4311, v4360, v4388, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4390 = __riscv_vsrl_vx_u8mf2(v4366, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4391 = __riscv_vand_vx_u8mf2(v4390, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4392 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4391);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4393 = __riscv_vand_vx_u8mf2(v4369, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4394 = __riscv_vmseq_vx_u8mf2_b16(v4393, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4395 = __riscv_vadd_vx_i8mf2_mu(v4394, v4392, v4392, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4396 = __riscv_vwmacc_vx_i16m1(v4318, v4363, v4395, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4397 = v19 + 1656;
      const uint8_t* v4398 = (const uint8_t*) v4397;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4399 = __riscv_vle8_v_u8mf2(v4398, 8);
      const uint8_t* v4400 = v19 + 632;
      const uint8_t* v4401 = (const uint8_t*) v4400;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4402 = __riscv_vle8_v_u8mf2(v4401, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4403 = __riscv_vand_vx_u8mf2(v4399, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4404 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4403);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4405 = __riscv_vand_vx_u8mf2(v4402, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4406 = __riscv_vmseq_vx_u8mf2_b16(v4405, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4407 = __riscv_vadd_vx_i8mf2_mu(v4406, v4404, v4404, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4408 = __riscv_vwmacc_vx_i16m1(v4330, v4354, v4407, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4409 = __riscv_vsrl_vx_u8mf2(v4399, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4410 = __riscv_vand_vx_u8mf2(v4409, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4411 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4410);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4412 = __riscv_vand_vx_u8mf2(v4402, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4413 = __riscv_vmseq_vx_u8mf2_b16(v4412, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4414 = __riscv_vadd_vx_i8mf2_mu(v4413, v4411, v4411, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4415 = __riscv_vwmacc_vx_i16m1(v4337, v4357, v4414, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4416 = __riscv_vsrl_vx_u8mf2(v4399, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4417 = __riscv_vand_vx_u8mf2(v4416, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4418 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4417);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4419 = __riscv_vand_vx_u8mf2(v4402, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4420 = __riscv_vmseq_vx_u8mf2_b16(v4419, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4421 = __riscv_vadd_vx_i8mf2_mu(v4420, v4418, v4418, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4422 = __riscv_vwmacc_vx_i16m1(v4344, v4360, v4421, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4423 = __riscv_vsrl_vx_u8mf2(v4399, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4424 = __riscv_vand_vx_u8mf2(v4423, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4425 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4424);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4426 = __riscv_vand_vx_u8mf2(v4402, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4427 = __riscv_vmseq_vx_u8mf2_b16(v4426, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4428 = __riscv_vadd_vx_i8mf2_mu(v4427, v4425, v4425, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4429 = __riscv_vwmacc_vx_i16m1(v4351, v4363, v4428, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4430 = v21 + 154;
      const int8_t* v4431 = (const int8_t*) v4430;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4432 = *(const int8_t *)(v4431);
      const uint8_t* v4433 = v21 + 186;
      const int8_t* v4434 = (const int8_t*) v4433;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4435 = *(const int8_t *)(v4434);
      const uint8_t* v4436 = v21 + 218;
      const int8_t* v4437 = (const int8_t*) v4436;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4438 = *(const int8_t *)(v4437);
      const uint8_t* v4439 = v21 + 250;
      const int8_t* v4440 = (const int8_t*) v4439;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4441 = *(const int8_t *)(v4440);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4442 = v19 + 1664;
      const uint8_t* v4443 = (const uint8_t*) v4442;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4444 = __riscv_vle8_v_u8mf2(v4443, 8);
      const uint8_t* v4445 = v19 + 640;
      const uint8_t* v4446 = (const uint8_t*) v4445;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4447 = __riscv_vle8_v_u8mf2(v4446, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4448 = __riscv_vand_vx_u8mf2(v4444, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4449 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4448);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4450 = __riscv_vand_vx_u8mf2(v4447, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4451 = __riscv_vmseq_vx_u8mf2_b16(v4450, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4452 = __riscv_vadd_vx_i8mf2_mu(v4451, v4449, v4449, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4453 = __riscv_vwmacc_vx_i16m1(v4375, v4432, v4452, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4454 = __riscv_vsrl_vx_u8mf2(v4444, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4455 = __riscv_vand_vx_u8mf2(v4454, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4456 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4455);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4457 = __riscv_vand_vx_u8mf2(v4447, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4458 = __riscv_vmseq_vx_u8mf2_b16(v4457, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4459 = __riscv_vadd_vx_i8mf2_mu(v4458, v4456, v4456, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4460 = __riscv_vwmacc_vx_i16m1(v4382, v4435, v4459, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4461 = __riscv_vsrl_vx_u8mf2(v4444, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4462 = __riscv_vand_vx_u8mf2(v4461, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4463 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4462);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4464 = __riscv_vand_vx_u8mf2(v4447, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4465 = __riscv_vmseq_vx_u8mf2_b16(v4464, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4466 = __riscv_vadd_vx_i8mf2_mu(v4465, v4463, v4463, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4467 = __riscv_vwmacc_vx_i16m1(v4389, v4438, v4466, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4468 = __riscv_vsrl_vx_u8mf2(v4444, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4469 = __riscv_vand_vx_u8mf2(v4468, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4470 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4469);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4471 = __riscv_vand_vx_u8mf2(v4447, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4472 = __riscv_vmseq_vx_u8mf2_b16(v4471, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4473 = __riscv_vadd_vx_i8mf2_mu(v4472, v4470, v4470, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4474 = __riscv_vwmacc_vx_i16m1(v4396, v4441, v4473, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4475 = v19 + 1672;
      const uint8_t* v4476 = (const uint8_t*) v4475;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4477 = __riscv_vle8_v_u8mf2(v4476, 8);
      const uint8_t* v4478 = v19 + 648;
      const uint8_t* v4479 = (const uint8_t*) v4478;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4480 = __riscv_vle8_v_u8mf2(v4479, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4481 = __riscv_vand_vx_u8mf2(v4477, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4482 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4481);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4483 = __riscv_vand_vx_u8mf2(v4480, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4484 = __riscv_vmseq_vx_u8mf2_b16(v4483, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4485 = __riscv_vadd_vx_i8mf2_mu(v4484, v4482, v4482, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4486 = __riscv_vwmacc_vx_i16m1(v4408, v4432, v4485, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4487 = __riscv_vsrl_vx_u8mf2(v4477, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4488 = __riscv_vand_vx_u8mf2(v4487, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4489 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4488);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4490 = __riscv_vand_vx_u8mf2(v4480, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4491 = __riscv_vmseq_vx_u8mf2_b16(v4490, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4492 = __riscv_vadd_vx_i8mf2_mu(v4491, v4489, v4489, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4493 = __riscv_vwmacc_vx_i16m1(v4415, v4435, v4492, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4494 = __riscv_vsrl_vx_u8mf2(v4477, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4495 = __riscv_vand_vx_u8mf2(v4494, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4496 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4495);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4497 = __riscv_vand_vx_u8mf2(v4480, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4498 = __riscv_vmseq_vx_u8mf2_b16(v4497, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4499 = __riscv_vadd_vx_i8mf2_mu(v4498, v4496, v4496, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4500 = __riscv_vwmacc_vx_i16m1(v4422, v4438, v4499, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4501 = __riscv_vsrl_vx_u8mf2(v4477, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4502 = __riscv_vand_vx_u8mf2(v4501, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4503 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4502);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4504 = __riscv_vand_vx_u8mf2(v4480, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4505 = __riscv_vmseq_vx_u8mf2_b16(v4504, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4506 = __riscv_vadd_vx_i8mf2_mu(v4505, v4503, v4503, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4507 = __riscv_vwmacc_vx_i16m1(v4429, v4441, v4506, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4508 = v21 + 155;
      const int8_t* v4509 = (const int8_t*) v4508;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4510 = *(const int8_t *)(v4509);
      const uint8_t* v4511 = v21 + 187;
      const int8_t* v4512 = (const int8_t*) v4511;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4513 = *(const int8_t *)(v4512);
      const uint8_t* v4514 = v21 + 219;
      const int8_t* v4515 = (const int8_t*) v4514;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4516 = *(const int8_t *)(v4515);
      const uint8_t* v4517 = v21 + 251;
      const int8_t* v4518 = (const int8_t*) v4517;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4519 = *(const int8_t *)(v4518);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4520 = v19 + 1680;
      const uint8_t* v4521 = (const uint8_t*) v4520;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4522 = __riscv_vle8_v_u8mf2(v4521, 8);
      const uint8_t* v4523 = v19 + 656;
      const uint8_t* v4524 = (const uint8_t*) v4523;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4525 = __riscv_vle8_v_u8mf2(v4524, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4526 = __riscv_vand_vx_u8mf2(v4522, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4527 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4526);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4528 = __riscv_vand_vx_u8mf2(v4525, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4529 = __riscv_vmseq_vx_u8mf2_b16(v4528, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4530 = __riscv_vadd_vx_i8mf2_mu(v4529, v4527, v4527, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4531 = __riscv_vwmacc_vx_i16m1(v4453, v4510, v4530, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4532 = __riscv_vsrl_vx_u8mf2(v4522, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4533 = __riscv_vand_vx_u8mf2(v4532, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4534 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4533);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4535 = __riscv_vand_vx_u8mf2(v4525, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4536 = __riscv_vmseq_vx_u8mf2_b16(v4535, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4537 = __riscv_vadd_vx_i8mf2_mu(v4536, v4534, v4534, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4538 = __riscv_vwmacc_vx_i16m1(v4460, v4513, v4537, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4539 = __riscv_vsrl_vx_u8mf2(v4522, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4540 = __riscv_vand_vx_u8mf2(v4539, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4541 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4540);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4542 = __riscv_vand_vx_u8mf2(v4525, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4543 = __riscv_vmseq_vx_u8mf2_b16(v4542, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4544 = __riscv_vadd_vx_i8mf2_mu(v4543, v4541, v4541, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4545 = __riscv_vwmacc_vx_i16m1(v4467, v4516, v4544, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4546 = __riscv_vsrl_vx_u8mf2(v4522, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4547 = __riscv_vand_vx_u8mf2(v4546, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4548 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4547);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4549 = __riscv_vand_vx_u8mf2(v4525, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4550 = __riscv_vmseq_vx_u8mf2_b16(v4549, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4551 = __riscv_vadd_vx_i8mf2_mu(v4550, v4548, v4548, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4552 = __riscv_vwmacc_vx_i16m1(v4474, v4519, v4551, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4553 = v19 + 1688;
      const uint8_t* v4554 = (const uint8_t*) v4553;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4555 = __riscv_vle8_v_u8mf2(v4554, 8);
      const uint8_t* v4556 = v19 + 664;
      const uint8_t* v4557 = (const uint8_t*) v4556;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4558 = __riscv_vle8_v_u8mf2(v4557, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4559 = __riscv_vand_vx_u8mf2(v4555, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4560 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4559);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4561 = __riscv_vand_vx_u8mf2(v4558, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4562 = __riscv_vmseq_vx_u8mf2_b16(v4561, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4563 = __riscv_vadd_vx_i8mf2_mu(v4562, v4560, v4560, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4564 = __riscv_vwmacc_vx_i16m1(v4486, v4510, v4563, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4565 = __riscv_vsrl_vx_u8mf2(v4555, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4566 = __riscv_vand_vx_u8mf2(v4565, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4567 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4566);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4568 = __riscv_vand_vx_u8mf2(v4558, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4569 = __riscv_vmseq_vx_u8mf2_b16(v4568, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4570 = __riscv_vadd_vx_i8mf2_mu(v4569, v4567, v4567, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4571 = __riscv_vwmacc_vx_i16m1(v4493, v4513, v4570, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4572 = __riscv_vsrl_vx_u8mf2(v4555, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4573 = __riscv_vand_vx_u8mf2(v4572, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4574 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4573);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4575 = __riscv_vand_vx_u8mf2(v4558, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4576 = __riscv_vmseq_vx_u8mf2_b16(v4575, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4577 = __riscv_vadd_vx_i8mf2_mu(v4576, v4574, v4574, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4578 = __riscv_vwmacc_vx_i16m1(v4500, v4516, v4577, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4579 = __riscv_vsrl_vx_u8mf2(v4555, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4580 = __riscv_vand_vx_u8mf2(v4579, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4581 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4580);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4582 = __riscv_vand_vx_u8mf2(v4558, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4583 = __riscv_vmseq_vx_u8mf2_b16(v4582, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4584 = __riscv_vadd_vx_i8mf2_mu(v4583, v4581, v4581, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4585 = __riscv_vwmacc_vx_i16m1(v4507, v4519, v4584, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4586 = v21 + 156;
      const int8_t* v4587 = (const int8_t*) v4586;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4588 = *(const int8_t *)(v4587);
      const uint8_t* v4589 = v21 + 188;
      const int8_t* v4590 = (const int8_t*) v4589;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4591 = *(const int8_t *)(v4590);
      const uint8_t* v4592 = v21 + 220;
      const int8_t* v4593 = (const int8_t*) v4592;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4594 = *(const int8_t *)(v4593);
      const uint8_t* v4595 = v21 + 252;
      const int8_t* v4596 = (const int8_t*) v4595;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4597 = *(const int8_t *)(v4596);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4598 = v19 + 1696;
      const uint8_t* v4599 = (const uint8_t*) v4598;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4600 = __riscv_vle8_v_u8mf2(v4599, 8);
      const uint8_t* v4601 = v19 + 672;
      const uint8_t* v4602 = (const uint8_t*) v4601;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4603 = __riscv_vle8_v_u8mf2(v4602, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4604 = __riscv_vand_vx_u8mf2(v4600, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4605 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4604);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4606 = __riscv_vand_vx_u8mf2(v4603, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4607 = __riscv_vmseq_vx_u8mf2_b16(v4606, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4608 = __riscv_vadd_vx_i8mf2_mu(v4607, v4605, v4605, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4609 = __riscv_vwmacc_vx_i16m1(v4531, v4588, v4608, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4610 = __riscv_vsrl_vx_u8mf2(v4600, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4611 = __riscv_vand_vx_u8mf2(v4610, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4612 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4611);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4613 = __riscv_vand_vx_u8mf2(v4603, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4614 = __riscv_vmseq_vx_u8mf2_b16(v4613, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4615 = __riscv_vadd_vx_i8mf2_mu(v4614, v4612, v4612, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4616 = __riscv_vwmacc_vx_i16m1(v4538, v4591, v4615, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4617 = __riscv_vsrl_vx_u8mf2(v4600, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4618 = __riscv_vand_vx_u8mf2(v4617, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4619 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4618);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4620 = __riscv_vand_vx_u8mf2(v4603, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4621 = __riscv_vmseq_vx_u8mf2_b16(v4620, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4622 = __riscv_vadd_vx_i8mf2_mu(v4621, v4619, v4619, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4623 = __riscv_vwmacc_vx_i16m1(v4545, v4594, v4622, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4624 = __riscv_vsrl_vx_u8mf2(v4600, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4625 = __riscv_vand_vx_u8mf2(v4624, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4626 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4625);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4627 = __riscv_vand_vx_u8mf2(v4603, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4628 = __riscv_vmseq_vx_u8mf2_b16(v4627, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4629 = __riscv_vadd_vx_i8mf2_mu(v4628, v4626, v4626, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4630 = __riscv_vwmacc_vx_i16m1(v4552, v4597, v4629, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4631 = v19 + 1704;
      const uint8_t* v4632 = (const uint8_t*) v4631;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4633 = __riscv_vle8_v_u8mf2(v4632, 8);
      const uint8_t* v4634 = v19 + 680;
      const uint8_t* v4635 = (const uint8_t*) v4634;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4636 = __riscv_vle8_v_u8mf2(v4635, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4637 = __riscv_vand_vx_u8mf2(v4633, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4638 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4637);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4639 = __riscv_vand_vx_u8mf2(v4636, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4640 = __riscv_vmseq_vx_u8mf2_b16(v4639, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4641 = __riscv_vadd_vx_i8mf2_mu(v4640, v4638, v4638, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4642 = __riscv_vwmacc_vx_i16m1(v4564, v4588, v4641, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4643 = __riscv_vsrl_vx_u8mf2(v4633, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4644 = __riscv_vand_vx_u8mf2(v4643, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4645 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4644);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4646 = __riscv_vand_vx_u8mf2(v4636, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4647 = __riscv_vmseq_vx_u8mf2_b16(v4646, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4648 = __riscv_vadd_vx_i8mf2_mu(v4647, v4645, v4645, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4649 = __riscv_vwmacc_vx_i16m1(v4571, v4591, v4648, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4650 = __riscv_vsrl_vx_u8mf2(v4633, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4651 = __riscv_vand_vx_u8mf2(v4650, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4652 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4651);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4653 = __riscv_vand_vx_u8mf2(v4636, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4654 = __riscv_vmseq_vx_u8mf2_b16(v4653, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4655 = __riscv_vadd_vx_i8mf2_mu(v4654, v4652, v4652, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4656 = __riscv_vwmacc_vx_i16m1(v4578, v4594, v4655, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4657 = __riscv_vsrl_vx_u8mf2(v4633, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4658 = __riscv_vand_vx_u8mf2(v4657, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4659 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4658);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4660 = __riscv_vand_vx_u8mf2(v4636, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4661 = __riscv_vmseq_vx_u8mf2_b16(v4660, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4662 = __riscv_vadd_vx_i8mf2_mu(v4661, v4659, v4659, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4663 = __riscv_vwmacc_vx_i16m1(v4585, v4597, v4662, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4664 = v21 + 157;
      const int8_t* v4665 = (const int8_t*) v4664;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4666 = *(const int8_t *)(v4665);
      const uint8_t* v4667 = v21 + 189;
      const int8_t* v4668 = (const int8_t*) v4667;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4669 = *(const int8_t *)(v4668);
      const uint8_t* v4670 = v21 + 221;
      const int8_t* v4671 = (const int8_t*) v4670;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4672 = *(const int8_t *)(v4671);
      const uint8_t* v4673 = v21 + 253;
      const int8_t* v4674 = (const int8_t*) v4673;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4675 = *(const int8_t *)(v4674);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4676 = v19 + 1712;
      const uint8_t* v4677 = (const uint8_t*) v4676;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4678 = __riscv_vle8_v_u8mf2(v4677, 8);
      const uint8_t* v4679 = v19 + 688;
      const uint8_t* v4680 = (const uint8_t*) v4679;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4681 = __riscv_vle8_v_u8mf2(v4680, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4682 = __riscv_vand_vx_u8mf2(v4678, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4683 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4682);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4684 = __riscv_vand_vx_u8mf2(v4681, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4685 = __riscv_vmseq_vx_u8mf2_b16(v4684, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4686 = __riscv_vadd_vx_i8mf2_mu(v4685, v4683, v4683, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4687 = __riscv_vwmacc_vx_i16m1(v4609, v4666, v4686, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4688 = __riscv_vsrl_vx_u8mf2(v4678, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4689 = __riscv_vand_vx_u8mf2(v4688, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4690 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4689);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4691 = __riscv_vand_vx_u8mf2(v4681, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4692 = __riscv_vmseq_vx_u8mf2_b16(v4691, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4693 = __riscv_vadd_vx_i8mf2_mu(v4692, v4690, v4690, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4694 = __riscv_vwmacc_vx_i16m1(v4616, v4669, v4693, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4695 = __riscv_vsrl_vx_u8mf2(v4678, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4696 = __riscv_vand_vx_u8mf2(v4695, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4697 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4696);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4698 = __riscv_vand_vx_u8mf2(v4681, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4699 = __riscv_vmseq_vx_u8mf2_b16(v4698, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4700 = __riscv_vadd_vx_i8mf2_mu(v4699, v4697, v4697, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4701 = __riscv_vwmacc_vx_i16m1(v4623, v4672, v4700, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4702 = __riscv_vsrl_vx_u8mf2(v4678, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4703 = __riscv_vand_vx_u8mf2(v4702, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4704 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4703);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4705 = __riscv_vand_vx_u8mf2(v4681, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4706 = __riscv_vmseq_vx_u8mf2_b16(v4705, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4707 = __riscv_vadd_vx_i8mf2_mu(v4706, v4704, v4704, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4708 = __riscv_vwmacc_vx_i16m1(v4630, v4675, v4707, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4709 = v19 + 1720;
      const uint8_t* v4710 = (const uint8_t*) v4709;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4711 = __riscv_vle8_v_u8mf2(v4710, 8);
      const uint8_t* v4712 = v19 + 696;
      const uint8_t* v4713 = (const uint8_t*) v4712;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4714 = __riscv_vle8_v_u8mf2(v4713, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4715 = __riscv_vand_vx_u8mf2(v4711, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4716 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4715);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4717 = __riscv_vand_vx_u8mf2(v4714, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4718 = __riscv_vmseq_vx_u8mf2_b16(v4717, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4719 = __riscv_vadd_vx_i8mf2_mu(v4718, v4716, v4716, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4720 = __riscv_vwmacc_vx_i16m1(v4642, v4666, v4719, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4721 = __riscv_vsrl_vx_u8mf2(v4711, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4722 = __riscv_vand_vx_u8mf2(v4721, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4723 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4722);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4724 = __riscv_vand_vx_u8mf2(v4714, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4725 = __riscv_vmseq_vx_u8mf2_b16(v4724, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4726 = __riscv_vadd_vx_i8mf2_mu(v4725, v4723, v4723, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4727 = __riscv_vwmacc_vx_i16m1(v4649, v4669, v4726, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4728 = __riscv_vsrl_vx_u8mf2(v4711, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4729 = __riscv_vand_vx_u8mf2(v4728, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4730 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4729);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4731 = __riscv_vand_vx_u8mf2(v4714, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4732 = __riscv_vmseq_vx_u8mf2_b16(v4731, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4733 = __riscv_vadd_vx_i8mf2_mu(v4732, v4730, v4730, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4734 = __riscv_vwmacc_vx_i16m1(v4656, v4672, v4733, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4735 = __riscv_vsrl_vx_u8mf2(v4711, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4736 = __riscv_vand_vx_u8mf2(v4735, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4737 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4736);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4738 = __riscv_vand_vx_u8mf2(v4714, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4739 = __riscv_vmseq_vx_u8mf2_b16(v4738, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4740 = __riscv_vadd_vx_i8mf2_mu(v4739, v4737, v4737, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4741 = __riscv_vwmacc_vx_i16m1(v4663, v4675, v4740, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4742 = v21 + 158;
      const int8_t* v4743 = (const int8_t*) v4742;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4744 = *(const int8_t *)(v4743);
      const uint8_t* v4745 = v21 + 190;
      const int8_t* v4746 = (const int8_t*) v4745;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4747 = *(const int8_t *)(v4746);
      const uint8_t* v4748 = v21 + 222;
      const int8_t* v4749 = (const int8_t*) v4748;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4750 = *(const int8_t *)(v4749);
      const uint8_t* v4751 = v21 + 254;
      const int8_t* v4752 = (const int8_t*) v4751;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4753 = *(const int8_t *)(v4752);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4754 = v19 + 1728;
      const uint8_t* v4755 = (const uint8_t*) v4754;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4756 = __riscv_vle8_v_u8mf2(v4755, 8);
      const uint8_t* v4757 = v19 + 704;
      const uint8_t* v4758 = (const uint8_t*) v4757;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4759 = __riscv_vle8_v_u8mf2(v4758, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4760 = __riscv_vand_vx_u8mf2(v4756, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4761 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4760);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4762 = __riscv_vand_vx_u8mf2(v4759, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4763 = __riscv_vmseq_vx_u8mf2_b16(v4762, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4764 = __riscv_vadd_vx_i8mf2_mu(v4763, v4761, v4761, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4765 = __riscv_vwmacc_vx_i16m1(v4687, v4744, v4764, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4766 = __riscv_vsrl_vx_u8mf2(v4756, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4767 = __riscv_vand_vx_u8mf2(v4766, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4768 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4767);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4769 = __riscv_vand_vx_u8mf2(v4759, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4770 = __riscv_vmseq_vx_u8mf2_b16(v4769, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4771 = __riscv_vadd_vx_i8mf2_mu(v4770, v4768, v4768, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4772 = __riscv_vwmacc_vx_i16m1(v4694, v4747, v4771, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4773 = __riscv_vsrl_vx_u8mf2(v4756, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4774 = __riscv_vand_vx_u8mf2(v4773, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4775 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4774);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4776 = __riscv_vand_vx_u8mf2(v4759, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4777 = __riscv_vmseq_vx_u8mf2_b16(v4776, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4778 = __riscv_vadd_vx_i8mf2_mu(v4777, v4775, v4775, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4779 = __riscv_vwmacc_vx_i16m1(v4701, v4750, v4778, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4780 = __riscv_vsrl_vx_u8mf2(v4756, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4781 = __riscv_vand_vx_u8mf2(v4780, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4782 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4781);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4783 = __riscv_vand_vx_u8mf2(v4759, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4784 = __riscv_vmseq_vx_u8mf2_b16(v4783, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4785 = __riscv_vadd_vx_i8mf2_mu(v4784, v4782, v4782, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4786 = __riscv_vwmacc_vx_i16m1(v4708, v4753, v4785, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4787 = v19 + 1736;
      const uint8_t* v4788 = (const uint8_t*) v4787;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4789 = __riscv_vle8_v_u8mf2(v4788, 8);
      const uint8_t* v4790 = v19 + 712;
      const uint8_t* v4791 = (const uint8_t*) v4790;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4792 = __riscv_vle8_v_u8mf2(v4791, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4793 = __riscv_vand_vx_u8mf2(v4789, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4794 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4793);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4795 = __riscv_vand_vx_u8mf2(v4792, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4796 = __riscv_vmseq_vx_u8mf2_b16(v4795, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4797 = __riscv_vadd_vx_i8mf2_mu(v4796, v4794, v4794, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4798 = __riscv_vwmacc_vx_i16m1(v4720, v4744, v4797, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4799 = __riscv_vsrl_vx_u8mf2(v4789, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4800 = __riscv_vand_vx_u8mf2(v4799, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4801 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4800);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4802 = __riscv_vand_vx_u8mf2(v4792, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4803 = __riscv_vmseq_vx_u8mf2_b16(v4802, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4804 = __riscv_vadd_vx_i8mf2_mu(v4803, v4801, v4801, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4805 = __riscv_vwmacc_vx_i16m1(v4727, v4747, v4804, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4806 = __riscv_vsrl_vx_u8mf2(v4789, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4807 = __riscv_vand_vx_u8mf2(v4806, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4808 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4807);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4809 = __riscv_vand_vx_u8mf2(v4792, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4810 = __riscv_vmseq_vx_u8mf2_b16(v4809, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4811 = __riscv_vadd_vx_i8mf2_mu(v4810, v4808, v4808, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4812 = __riscv_vwmacc_vx_i16m1(v4734, v4750, v4811, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4813 = __riscv_vsrl_vx_u8mf2(v4789, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4814 = __riscv_vand_vx_u8mf2(v4813, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4815 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4814);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4816 = __riscv_vand_vx_u8mf2(v4792, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4817 = __riscv_vmseq_vx_u8mf2_b16(v4816, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4818 = __riscv_vadd_vx_i8mf2_mu(v4817, v4815, v4815, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4819 = __riscv_vwmacc_vx_i16m1(v4741, v4753, v4818, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4820 = v21 + 159;
      const int8_t* v4821 = (const int8_t*) v4820;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4822 = *(const int8_t *)(v4821);
      const uint8_t* v4823 = v21 + 191;
      const int8_t* v4824 = (const int8_t*) v4823;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4825 = *(const int8_t *)(v4824);
      const uint8_t* v4826 = v21 + 223;
      const int8_t* v4827 = (const int8_t*) v4826;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4828 = *(const int8_t *)(v4827);
      const uint8_t* v4829 = v21 + 255;
      const int8_t* v4830 = (const int8_t*) v4829;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4831 = *(const int8_t *)(v4830);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4832 = v19 + 1744;
      const uint8_t* v4833 = (const uint8_t*) v4832;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4834 = __riscv_vle8_v_u8mf2(v4833, 8);
      const uint8_t* v4835 = v19 + 720;
      const uint8_t* v4836 = (const uint8_t*) v4835;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4837 = __riscv_vle8_v_u8mf2(v4836, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4838 = __riscv_vand_vx_u8mf2(v4834, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4839 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4838);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4840 = __riscv_vand_vx_u8mf2(v4837, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4841 = __riscv_vmseq_vx_u8mf2_b16(v4840, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4842 = __riscv_vadd_vx_i8mf2_mu(v4841, v4839, v4839, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4843 = __riscv_vwmacc_vx_i16m1(v4765, v4822, v4842, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4844 = __riscv_vsrl_vx_u8mf2(v4834, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4845 = __riscv_vand_vx_u8mf2(v4844, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4846 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4845);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4847 = __riscv_vand_vx_u8mf2(v4837, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4848 = __riscv_vmseq_vx_u8mf2_b16(v4847, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4849 = __riscv_vadd_vx_i8mf2_mu(v4848, v4846, v4846, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4850 = __riscv_vwmacc_vx_i16m1(v4772, v4825, v4849, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4851 = __riscv_vsrl_vx_u8mf2(v4834, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4852 = __riscv_vand_vx_u8mf2(v4851, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4853 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4852);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4854 = __riscv_vand_vx_u8mf2(v4837, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4855 = __riscv_vmseq_vx_u8mf2_b16(v4854, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4856 = __riscv_vadd_vx_i8mf2_mu(v4855, v4853, v4853, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4857 = __riscv_vwmacc_vx_i16m1(v4779, v4828, v4856, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4858 = __riscv_vsrl_vx_u8mf2(v4834, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4859 = __riscv_vand_vx_u8mf2(v4858, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4860 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4859);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4861 = __riscv_vand_vx_u8mf2(v4837, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4862 = __riscv_vmseq_vx_u8mf2_b16(v4861, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4863 = __riscv_vadd_vx_i8mf2_mu(v4862, v4860, v4860, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4864 = __riscv_vwmacc_vx_i16m1(v4786, v4831, v4863, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4865 = v19 + 1752;
      const uint8_t* v4866 = (const uint8_t*) v4865;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4867 = __riscv_vle8_v_u8mf2(v4866, 8);
      const uint8_t* v4868 = v19 + 728;
      const uint8_t* v4869 = (const uint8_t*) v4868;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4870 = __riscv_vle8_v_u8mf2(v4869, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4871 = __riscv_vand_vx_u8mf2(v4867, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4872 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4871);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4873 = __riscv_vand_vx_u8mf2(v4870, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4874 = __riscv_vmseq_vx_u8mf2_b16(v4873, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4875 = __riscv_vadd_vx_i8mf2_mu(v4874, v4872, v4872, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4876 = __riscv_vwmacc_vx_i16m1(v4798, v4822, v4875, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4877 = __riscv_vsrl_vx_u8mf2(v4867, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4878 = __riscv_vand_vx_u8mf2(v4877, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4879 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4878);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4880 = __riscv_vand_vx_u8mf2(v4870, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4881 = __riscv_vmseq_vx_u8mf2_b16(v4880, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4882 = __riscv_vadd_vx_i8mf2_mu(v4881, v4879, v4879, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4883 = __riscv_vwmacc_vx_i16m1(v4805, v4825, v4882, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4884 = __riscv_vsrl_vx_u8mf2(v4867, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4885 = __riscv_vand_vx_u8mf2(v4884, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4886 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4885);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4887 = __riscv_vand_vx_u8mf2(v4870, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4888 = __riscv_vmseq_vx_u8mf2_b16(v4887, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4889 = __riscv_vadd_vx_i8mf2_mu(v4888, v4886, v4886, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4890 = __riscv_vwmacc_vx_i16m1(v4812, v4828, v4889, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4891 = __riscv_vsrl_vx_u8mf2(v4867, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4892 = __riscv_vand_vx_u8mf2(v4891, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4893 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4892);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4894 = __riscv_vand_vx_u8mf2(v4870, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4895 = __riscv_vmseq_vx_u8mf2_b16(v4894, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4896 = __riscv_vadd_vx_i8mf2_mu(v4895, v4893, v4893, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4897 = __riscv_vwmacc_vx_i16m1(v4819, v4831, v4896, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4898 = v21 + 160;
      const int8_t* v4899 = (const int8_t*) v4898;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4900 = *(const int8_t *)(v4899);
      const uint8_t* v4901 = v21 + 192;
      const int8_t* v4902 = (const int8_t*) v4901;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4903 = *(const int8_t *)(v4902);
      const uint8_t* v4904 = v21 + 224;
      const int8_t* v4905 = (const int8_t*) v4904;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4906 = *(const int8_t *)(v4905);
      const uint8_t* v4907 = v21 + 256;
      const int8_t* v4908 = (const int8_t*) v4907;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4909 = *(const int8_t *)(v4908);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4910 = v19 + 1760;
      const uint8_t* v4911 = (const uint8_t*) v4910;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4912 = __riscv_vle8_v_u8mf2(v4911, 8);
      const uint8_t* v4913 = v19 + 736;
      const uint8_t* v4914 = (const uint8_t*) v4913;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4915 = __riscv_vle8_v_u8mf2(v4914, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4916 = __riscv_vand_vx_u8mf2(v4912, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4917 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4916);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4918 = __riscv_vand_vx_u8mf2(v4915, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4919 = __riscv_vmseq_vx_u8mf2_b16(v4918, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4920 = __riscv_vadd_vx_i8mf2_mu(v4919, v4917, v4917, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4921 = __riscv_vwmacc_vx_i16m1(v4843, v4900, v4920, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4922 = __riscv_vsrl_vx_u8mf2(v4912, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4923 = __riscv_vand_vx_u8mf2(v4922, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4924 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4923);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4925 = __riscv_vand_vx_u8mf2(v4915, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4926 = __riscv_vmseq_vx_u8mf2_b16(v4925, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4927 = __riscv_vadd_vx_i8mf2_mu(v4926, v4924, v4924, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4928 = __riscv_vwmacc_vx_i16m1(v4850, v4903, v4927, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4929 = __riscv_vsrl_vx_u8mf2(v4912, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4930 = __riscv_vand_vx_u8mf2(v4929, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4931 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4930);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4932 = __riscv_vand_vx_u8mf2(v4915, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4933 = __riscv_vmseq_vx_u8mf2_b16(v4932, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4934 = __riscv_vadd_vx_i8mf2_mu(v4933, v4931, v4931, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4935 = __riscv_vwmacc_vx_i16m1(v4857, v4906, v4934, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4936 = __riscv_vsrl_vx_u8mf2(v4912, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4937 = __riscv_vand_vx_u8mf2(v4936, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4938 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4937);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4939 = __riscv_vand_vx_u8mf2(v4915, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4940 = __riscv_vmseq_vx_u8mf2_b16(v4939, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4941 = __riscv_vadd_vx_i8mf2_mu(v4940, v4938, v4938, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4942 = __riscv_vwmacc_vx_i16m1(v4864, v4909, v4941, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4943 = v19 + 1768;
      const uint8_t* v4944 = (const uint8_t*) v4943;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4945 = __riscv_vle8_v_u8mf2(v4944, 8);
      const uint8_t* v4946 = v19 + 744;
      const uint8_t* v4947 = (const uint8_t*) v4946;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4948 = __riscv_vle8_v_u8mf2(v4947, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4949 = __riscv_vand_vx_u8mf2(v4945, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4950 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4949);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4951 = __riscv_vand_vx_u8mf2(v4948, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4952 = __riscv_vmseq_vx_u8mf2_b16(v4951, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4953 = __riscv_vadd_vx_i8mf2_mu(v4952, v4950, v4950, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4954 = __riscv_vwmacc_vx_i16m1(v4876, v4900, v4953, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4955 = __riscv_vsrl_vx_u8mf2(v4945, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4956 = __riscv_vand_vx_u8mf2(v4955, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4957 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4956);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4958 = __riscv_vand_vx_u8mf2(v4948, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4959 = __riscv_vmseq_vx_u8mf2_b16(v4958, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4960 = __riscv_vadd_vx_i8mf2_mu(v4959, v4957, v4957, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4961 = __riscv_vwmacc_vx_i16m1(v4883, v4903, v4960, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4962 = __riscv_vsrl_vx_u8mf2(v4945, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4963 = __riscv_vand_vx_u8mf2(v4962, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4964 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4963);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4965 = __riscv_vand_vx_u8mf2(v4948, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4966 = __riscv_vmseq_vx_u8mf2_b16(v4965, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4967 = __riscv_vadd_vx_i8mf2_mu(v4966, v4964, v4964, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4968 = __riscv_vwmacc_vx_i16m1(v4890, v4906, v4967, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4969 = __riscv_vsrl_vx_u8mf2(v4945, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4970 = __riscv_vand_vx_u8mf2(v4969, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4971 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4970);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4972 = __riscv_vand_vx_u8mf2(v4948, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4973 = __riscv_vmseq_vx_u8mf2_b16(v4972, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4974 = __riscv_vadd_vx_i8mf2_mu(v4973, v4971, v4971, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4975 = __riscv_vwmacc_vx_i16m1(v4897, v4909, v4974, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4976 = v21 + 161;
      const int8_t* v4977 = (const int8_t*) v4976;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4978 = *(const int8_t *)(v4977);
      const uint8_t* v4979 = v21 + 193;
      const int8_t* v4980 = (const int8_t*) v4979;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4981 = *(const int8_t *)(v4980);
      const uint8_t* v4982 = v21 + 225;
      const int8_t* v4983 = (const int8_t*) v4982;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4984 = *(const int8_t *)(v4983);
      const uint8_t* v4985 = v21 + 257;
      const int8_t* v4986 = (const int8_t*) v4985;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4987 = *(const int8_t *)(v4986);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v4988 = v19 + 1776;
      const uint8_t* v4989 = (const uint8_t*) v4988;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4990 = __riscv_vle8_v_u8mf2(v4989, 8);
      const uint8_t* v4991 = v19 + 752;
      const uint8_t* v4992 = (const uint8_t*) v4991;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4993 = __riscv_vle8_v_u8mf2(v4992, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4994 = __riscv_vand_vx_u8mf2(v4990, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4995 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4994);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4996 = __riscv_vand_vx_u8mf2(v4993, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v4997 = __riscv_vmseq_vx_u8mf2_b16(v4996, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v4998 = __riscv_vadd_vx_i8mf2_mu(v4997, v4995, v4995, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4999 = __riscv_vwmacc_vx_i16m1(v4921, v4978, v4998, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5000 = __riscv_vsrl_vx_u8mf2(v4990, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5001 = __riscv_vand_vx_u8mf2(v5000, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5002 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5001);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5003 = __riscv_vand_vx_u8mf2(v4993, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5004 = __riscv_vmseq_vx_u8mf2_b16(v5003, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5005 = __riscv_vadd_vx_i8mf2_mu(v5004, v5002, v5002, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5006 = __riscv_vwmacc_vx_i16m1(v4928, v4981, v5005, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5007 = __riscv_vsrl_vx_u8mf2(v4990, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5008 = __riscv_vand_vx_u8mf2(v5007, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5009 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5008);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5010 = __riscv_vand_vx_u8mf2(v4993, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5011 = __riscv_vmseq_vx_u8mf2_b16(v5010, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5012 = __riscv_vadd_vx_i8mf2_mu(v5011, v5009, v5009, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5013 = __riscv_vwmacc_vx_i16m1(v4935, v4984, v5012, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5014 = __riscv_vsrl_vx_u8mf2(v4990, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5015 = __riscv_vand_vx_u8mf2(v5014, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5016 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5015);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5017 = __riscv_vand_vx_u8mf2(v4993, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5018 = __riscv_vmseq_vx_u8mf2_b16(v5017, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5019 = __riscv_vadd_vx_i8mf2_mu(v5018, v5016, v5016, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5020 = __riscv_vwmacc_vx_i16m1(v4942, v4987, v5019, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v5021 = v19 + 1784;
      const uint8_t* v5022 = (const uint8_t*) v5021;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5023 = __riscv_vle8_v_u8mf2(v5022, 8);
      const uint8_t* v5024 = v19 + 760;
      const uint8_t* v5025 = (const uint8_t*) v5024;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5026 = __riscv_vle8_v_u8mf2(v5025, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5027 = __riscv_vand_vx_u8mf2(v5023, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5028 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5027);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5029 = __riscv_vand_vx_u8mf2(v5026, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5030 = __riscv_vmseq_vx_u8mf2_b16(v5029, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5031 = __riscv_vadd_vx_i8mf2_mu(v5030, v5028, v5028, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5032 = __riscv_vwmacc_vx_i16m1(v4954, v4978, v5031, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5033 = __riscv_vsrl_vx_u8mf2(v5023, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5034 = __riscv_vand_vx_u8mf2(v5033, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5035 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5034);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5036 = __riscv_vand_vx_u8mf2(v5026, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5037 = __riscv_vmseq_vx_u8mf2_b16(v5036, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5038 = __riscv_vadd_vx_i8mf2_mu(v5037, v5035, v5035, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5039 = __riscv_vwmacc_vx_i16m1(v4961, v4981, v5038, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5040 = __riscv_vsrl_vx_u8mf2(v5023, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5041 = __riscv_vand_vx_u8mf2(v5040, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5042 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5041);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5043 = __riscv_vand_vx_u8mf2(v5026, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5044 = __riscv_vmseq_vx_u8mf2_b16(v5043, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5045 = __riscv_vadd_vx_i8mf2_mu(v5044, v5042, v5042, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5046 = __riscv_vwmacc_vx_i16m1(v4968, v4984, v5045, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5047 = __riscv_vsrl_vx_u8mf2(v5023, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5048 = __riscv_vand_vx_u8mf2(v5047, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5049 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5048);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5050 = __riscv_vand_vx_u8mf2(v5026, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5051 = __riscv_vmseq_vx_u8mf2_b16(v5050, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5052 = __riscv_vadd_vx_i8mf2_mu(v5051, v5049, v5049, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5053 = __riscv_vwmacc_vx_i16m1(v4975, v4987, v5052, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5054 = v21 + 162;
      const int8_t* v5055 = (const int8_t*) v5054;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5056 = *(const int8_t *)(v5055);
      const uint8_t* v5057 = v21 + 194;
      const int8_t* v5058 = (const int8_t*) v5057;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5059 = *(const int8_t *)(v5058);
      const uint8_t* v5060 = v21 + 226;
      const int8_t* v5061 = (const int8_t*) v5060;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5062 = *(const int8_t *)(v5061);
      const uint8_t* v5063 = v21 + 258;
      const int8_t* v5064 = (const int8_t*) v5063;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5065 = *(const int8_t *)(v5064);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v5066 = v19 + 1792;
      const uint8_t* v5067 = (const uint8_t*) v5066;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5068 = __riscv_vle8_v_u8mf2(v5067, 8);
      const uint8_t* v5069 = v19 + 768;
      const uint8_t* v5070 = (const uint8_t*) v5069;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5071 = __riscv_vle8_v_u8mf2(v5070, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5072 = __riscv_vand_vx_u8mf2(v5068, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5073 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5072);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5074 = __riscv_vand_vx_u8mf2(v5071, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5075 = __riscv_vmseq_vx_u8mf2_b16(v5074, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5076 = __riscv_vadd_vx_i8mf2_mu(v5075, v5073, v5073, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5077 = __riscv_vwmacc_vx_i16m1(v4999, v5056, v5076, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5078 = __riscv_vsrl_vx_u8mf2(v5068, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5079 = __riscv_vand_vx_u8mf2(v5078, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5080 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5079);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5081 = __riscv_vand_vx_u8mf2(v5071, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5082 = __riscv_vmseq_vx_u8mf2_b16(v5081, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5083 = __riscv_vadd_vx_i8mf2_mu(v5082, v5080, v5080, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5084 = __riscv_vwmacc_vx_i16m1(v5006, v5059, v5083, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5085 = __riscv_vsrl_vx_u8mf2(v5068, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5086 = __riscv_vand_vx_u8mf2(v5085, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5087 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5086);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5088 = __riscv_vand_vx_u8mf2(v5071, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5089 = __riscv_vmseq_vx_u8mf2_b16(v5088, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5090 = __riscv_vadd_vx_i8mf2_mu(v5089, v5087, v5087, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5091 = __riscv_vwmacc_vx_i16m1(v5013, v5062, v5090, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5092 = __riscv_vsrl_vx_u8mf2(v5068, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5093 = __riscv_vand_vx_u8mf2(v5092, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5094 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5093);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5095 = __riscv_vand_vx_u8mf2(v5071, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5096 = __riscv_vmseq_vx_u8mf2_b16(v5095, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5097 = __riscv_vadd_vx_i8mf2_mu(v5096, v5094, v5094, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5098 = __riscv_vwmacc_vx_i16m1(v5020, v5065, v5097, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v5099 = v19 + 1800;
      const uint8_t* v5100 = (const uint8_t*) v5099;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5101 = __riscv_vle8_v_u8mf2(v5100, 8);
      const uint8_t* v5102 = v19 + 776;
      const uint8_t* v5103 = (const uint8_t*) v5102;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5104 = __riscv_vle8_v_u8mf2(v5103, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5105 = __riscv_vand_vx_u8mf2(v5101, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5106 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5105);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5107 = __riscv_vand_vx_u8mf2(v5104, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5108 = __riscv_vmseq_vx_u8mf2_b16(v5107, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5109 = __riscv_vadd_vx_i8mf2_mu(v5108, v5106, v5106, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5110 = __riscv_vwmacc_vx_i16m1(v5032, v5056, v5109, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5111 = __riscv_vsrl_vx_u8mf2(v5101, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5112 = __riscv_vand_vx_u8mf2(v5111, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5113 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5112);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5114 = __riscv_vand_vx_u8mf2(v5104, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5115 = __riscv_vmseq_vx_u8mf2_b16(v5114, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5116 = __riscv_vadd_vx_i8mf2_mu(v5115, v5113, v5113, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5117 = __riscv_vwmacc_vx_i16m1(v5039, v5059, v5116, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5118 = __riscv_vsrl_vx_u8mf2(v5101, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5119 = __riscv_vand_vx_u8mf2(v5118, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5120 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5119);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5121 = __riscv_vand_vx_u8mf2(v5104, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5122 = __riscv_vmseq_vx_u8mf2_b16(v5121, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5123 = __riscv_vadd_vx_i8mf2_mu(v5122, v5120, v5120, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5124 = __riscv_vwmacc_vx_i16m1(v5046, v5062, v5123, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5125 = __riscv_vsrl_vx_u8mf2(v5101, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5126 = __riscv_vand_vx_u8mf2(v5125, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5127 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5126);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5128 = __riscv_vand_vx_u8mf2(v5104, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5129 = __riscv_vmseq_vx_u8mf2_b16(v5128, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5130 = __riscv_vadd_vx_i8mf2_mu(v5129, v5127, v5127, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5131 = __riscv_vwmacc_vx_i16m1(v5053, v5065, v5130, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v5132 = v21 + 163;
      const int8_t* v5133 = (const int8_t*) v5132;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5134 = *(const int8_t *)(v5133);
      const uint8_t* v5135 = v21 + 195;
      const int8_t* v5136 = (const int8_t*) v5135;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5137 = *(const int8_t *)(v5136);
      const uint8_t* v5138 = v21 + 227;
      const int8_t* v5139 = (const int8_t*) v5138;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5140 = *(const int8_t *)(v5139);
      const uint8_t* v5141 = v21 + 259;
      const int8_t* v5142 = (const int8_t*) v5141;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v5143 = *(const int8_t *)(v5142);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v5144 = v19 + 1808;
      const uint8_t* v5145 = (const uint8_t*) v5144;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5146 = __riscv_vle8_v_u8mf2(v5145, 8);
      const uint8_t* v5147 = v19 + 784;
      const uint8_t* v5148 = (const uint8_t*) v5147;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5149 = __riscv_vle8_v_u8mf2(v5148, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5150 = __riscv_vand_vx_u8mf2(v5146, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5151 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5150);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5152 = __riscv_vand_vx_u8mf2(v5149, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5153 = __riscv_vmseq_vx_u8mf2_b16(v5152, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5154 = __riscv_vadd_vx_i8mf2_mu(v5153, v5151, v5151, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5155 = __riscv_vwmacc_vx_i16m1(v5077, v5134, v5154, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5156 = __riscv_vsrl_vx_u8mf2(v5146, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5157 = __riscv_vand_vx_u8mf2(v5156, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5158 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5157);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5159 = __riscv_vand_vx_u8mf2(v5149, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5160 = __riscv_vmseq_vx_u8mf2_b16(v5159, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5161 = __riscv_vadd_vx_i8mf2_mu(v5160, v5158, v5158, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5162 = __riscv_vwmacc_vx_i16m1(v5084, v5137, v5161, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5163 = __riscv_vsrl_vx_u8mf2(v5146, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5164 = __riscv_vand_vx_u8mf2(v5163, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5165 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5164);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5166 = __riscv_vand_vx_u8mf2(v5149, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5167 = __riscv_vmseq_vx_u8mf2_b16(v5166, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5168 = __riscv_vadd_vx_i8mf2_mu(v5167, v5165, v5165, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5169 = __riscv_vwmacc_vx_i16m1(v5091, v5140, v5168, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5170 = __riscv_vsrl_vx_u8mf2(v5146, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5171 = __riscv_vand_vx_u8mf2(v5170, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5172 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5171);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5173 = __riscv_vand_vx_u8mf2(v5149, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5174 = __riscv_vmseq_vx_u8mf2_b16(v5173, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5175 = __riscv_vadd_vx_i8mf2_mu(v5174, v5172, v5172, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5176 = __riscv_vwmacc_vx_i16m1(v5098, v5143, v5175, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
      const uint8_t* v5177 = v19 + 1816;
      const uint8_t* v5178 = (const uint8_t*) v5177;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5179 = __riscv_vle8_v_u8mf2(v5178, 8);
      const uint8_t* v5180 = v19 + 792;
      const uint8_t* v5181 = (const uint8_t*) v5180;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v5182 = __riscv_vle8_v_u8mf2(v5181, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5183 = __riscv_vand_vx_u8mf2(v5179, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5184 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5183);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5185 = __riscv_vand_vx_u8mf2(v5182, 16, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5186 = __riscv_vmseq_vx_u8mf2_b16(v5185, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5187 = __riscv_vadd_vx_i8mf2_mu(v5186, v5184, v5184, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5188 = __riscv_vwmacc_vx_i16m1(v5110, v5134, v5187, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5189 = __riscv_vsrl_vx_u8mf2(v5179, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5190 = __riscv_vand_vx_u8mf2(v5189, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5191 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5190);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5192 = __riscv_vand_vx_u8mf2(v5182, 32, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5193 = __riscv_vmseq_vx_u8mf2_b16(v5192, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5194 = __riscv_vadd_vx_i8mf2_mu(v5193, v5191, v5191, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5195 = __riscv_vwmacc_vx_i16m1(v5117, v5137, v5194, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5196 = __riscv_vsrl_vx_u8mf2(v5179, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5197 = __riscv_vand_vx_u8mf2(v5196, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5198 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5197);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5199 = __riscv_vand_vx_u8mf2(v5182, 64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5200 = __riscv_vmseq_vx_u8mf2_b16(v5199, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5201 = __riscv_vadd_vx_i8mf2_mu(v5200, v5198, v5198, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5202 = __riscv_vwmacc_vx_i16m1(v5124, v5140, v5201, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v5203 = __riscv_vsrl_vx_u8mf2(v5179, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5204 = __riscv_vand_vx_u8mf2(v5203, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v5205 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5204);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v5206 = __riscv_vand_vx_u8mf2(v5182, 128, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
      vbool16_t v5207 = __riscv_vmseq_vx_u8mf2_b16(v5206, 0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
      vint8mf2_t v5208 = __riscv_vadd_vx_i8mf2_mu(v5207, v5205, v5205, -4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v5209 = __riscv_vwmacc_vx_i16m1(v5131, v5143, v5208, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v5210 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v5211 = __riscv_vwmacc_vv_i32m2(v5210, v3925, v5155, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v5212 = __riscv_vwmacc_vv_i32m2(v5211, v3929, v5162, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v5213 = __riscv_vwmacc_vv_i32m2(v5212, v3933, v5169, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v5214 = __riscv_vwmacc_vv_i32m2(v5213, v3937, v5176, 8);
      v24 = v5214;
      vint32m2_t v5215 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v5216 = __riscv_vwmacc_vv_i32m2(v5215, v3941, v5188, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v5217 = __riscv_vwmacc_vv_i32m2(v5216, v3945, v5195, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v5218 = __riscv_vwmacc_vv_i32m2(v5217, v3949, v5202, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v5219 = __riscv_vwmacc_vv_i32m2(v5218, v3953, v5209, 8);
      v26 = v5219;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v5220 = (const _Float16*) v19;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v5221 = __riscv_vle16_v_f16m1(v5220, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v5222 = __riscv_vfwcvt_f_f_v_f32m2(v5221, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v5223 = __riscv_vfmul_vf_f32m2(v5222, v23, 8);
      vint32m2_t v5224 = v24;
      vfloat32m2_t v5225 = v13;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v5226 = __riscv_vfcvt_f_x_v_f32m2(v5224, 8);
      vfloat32m2_t v5227 = __riscv_vfmacc_vv_f32m2(v5225, v5226, v5223, 8);
      v13 = v5227;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v5228 = v19 + 16;
      const _Float16* v5229 = (const _Float16*) v5228;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v5230 = __riscv_vle16_v_f16m1(v5229, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v5231 = __riscv_vfwcvt_f_f_v_f32m2(v5230, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v5232 = __riscv_vfmul_vf_f32m2(v5231, v23, 8);
      vint32m2_t v5233 = v26;
      vfloat32m2_t v5234 = v15;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v5235 = __riscv_vfcvt_f_x_v_f32m2(v5233, 8);
      vfloat32m2_t v5236 = __riscv_vfmacc_vv_f32m2(v5234, v5235, v5232, 8);
      v15 = v5236;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v5237 = v9 * 16;
    float* v5238 = v2 + v5237;
    vfloat32m2_t v5239 = v13;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v5238, v5239, 8);
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v5240 = v9 * 16;
    size_t v5241 = v5240 + 8;
    float* v5242 = v2 + v5241;
    vfloat32m2_t v5243 = v15;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v5242, v5243, 8);
  }
  return;
}


