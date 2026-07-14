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
      vint16m1_t v60;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v61 = __riscv_vmv_v_x_i16m1(0, 8);
      v60 = v61;
      vint16m1_t v62;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v63 = __riscv_vmv_v_x_i16m1(0, 8);
      v62 = v63;
      vint16m1_t v64;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v65 = __riscv_vmv_v_x_i16m1(0, 8);
      v64 = v65;
      vint16m1_t v66;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v67 = __riscv_vmv_v_x_i16m1(0, 8);
      v66 = v67;
      vint16m1_t v68;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v69 = __riscv_vmv_v_x_i16m1(0, 8);
      v68 = v69;
      vint16m1_t v70;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v71 = __riscv_vmv_v_x_i16m1(0, 8);
      v70 = v71;
      vint16m1_t v72;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v73 = __riscv_vmv_v_x_i16m1(0, 8);
      v72 = v73;
      vint16m1_t v74;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v75 = __riscv_vmv_v_x_i16m1(0, 8);
      v74 = v75;
      for (size_t v76 = 0; v76 < 16; v76 += 1) {
        size_t v77 = v76 * 16;
        const uint8_t* v78 = v19 + v77;
        const uint8_t* v79 = v21 + v76;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v80 = v79 + 4;
        const int8_t* v81 = (const int8_t*) v80;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v82 = *(const int8_t *)(v81);
        const uint8_t* v83 = v79 + 36;
        const int8_t* v84 = (const int8_t*) v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v85 = *(const int8_t *)(v84);
        const uint8_t* v86 = v79 + 68;
        const int8_t* v87 = (const int8_t*) v86;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v88 = *(const int8_t *)(v87);
        const uint8_t* v89 = v79 + 100;
        const int8_t* v90 = (const int8_t*) v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v91 = *(const int8_t *)(v90);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
        const uint8_t* v92 = v78 + 800;
        const uint8_t* v93 = (const uint8_t*) v92;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v94 = __riscv_vle8_v_u8mf2(v93, 8);
        const uint8_t* v95 = v78 + 288;
        const uint8_t* v96 = (const uint8_t*) v95;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v97 = __riscv_vle8_v_u8mf2(v96, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v98 = __riscv_vand_vx_u8mf2(v94, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v99 = __riscv_vreinterpret_v_u8mf2_i8mf2(v98);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v100 = __riscv_vand_vx_u8mf2(v97, 1, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v101 = __riscv_vmseq_vx_u8mf2_b16(v100, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v102 = __riscv_vadd_vx_i8mf2_mu(v101, v99, v99, -4, 8);
        vint16m1_t v103 = v60;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v104 = __riscv_vwmacc_vx_i16m1(v103, v82, v102, 8);
        v60 = v104;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v105 = __riscv_vsrl_vx_u8mf2(v94, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v106 = __riscv_vand_vx_u8mf2(v105, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v107 = __riscv_vreinterpret_v_u8mf2_i8mf2(v106);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v108 = __riscv_vand_vx_u8mf2(v97, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v109 = __riscv_vmseq_vx_u8mf2_b16(v108, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v110 = __riscv_vadd_vx_i8mf2_mu(v109, v107, v107, -4, 8);
        vint16m1_t v111 = v62;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v112 = __riscv_vwmacc_vx_i16m1(v111, v85, v110, 8);
        v62 = v112;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v113 = __riscv_vsrl_vx_u8mf2(v94, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v114 = __riscv_vand_vx_u8mf2(v113, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v115 = __riscv_vreinterpret_v_u8mf2_i8mf2(v114);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v116 = __riscv_vand_vx_u8mf2(v97, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v117 = __riscv_vmseq_vx_u8mf2_b16(v116, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v118 = __riscv_vadd_vx_i8mf2_mu(v117, v115, v115, -4, 8);
        vint16m1_t v119 = v64;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v120 = __riscv_vwmacc_vx_i16m1(v119, v88, v118, 8);
        v64 = v120;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v121 = __riscv_vsrl_vx_u8mf2(v94, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v122 = __riscv_vand_vx_u8mf2(v121, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v123 = __riscv_vreinterpret_v_u8mf2_i8mf2(v122);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v124 = __riscv_vand_vx_u8mf2(v97, 8, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v125 = __riscv_vmseq_vx_u8mf2_b16(v124, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v126 = __riscv_vadd_vx_i8mf2_mu(v125, v123, v123, -4, 8);
        vint16m1_t v127 = v66;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v128 = __riscv_vwmacc_vx_i16m1(v127, v91, v126, 8);
        v66 = v128;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
        const uint8_t* v129 = v78 + 808;
        const uint8_t* v130 = (const uint8_t*) v129;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v131 = __riscv_vle8_v_u8mf2(v130, 8);
        const uint8_t* v132 = v78 + 296;
        const uint8_t* v133 = (const uint8_t*) v132;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v134 = __riscv_vle8_v_u8mf2(v133, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v135 = __riscv_vand_vx_u8mf2(v131, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v136 = __riscv_vreinterpret_v_u8mf2_i8mf2(v135);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v137 = __riscv_vand_vx_u8mf2(v134, 1, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v138 = __riscv_vmseq_vx_u8mf2_b16(v137, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v139 = __riscv_vadd_vx_i8mf2_mu(v138, v136, v136, -4, 8);
        vint16m1_t v140 = v68;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v141 = __riscv_vwmacc_vx_i16m1(v140, v82, v139, 8);
        v68 = v141;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v142 = __riscv_vsrl_vx_u8mf2(v131, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v143 = __riscv_vand_vx_u8mf2(v142, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v144 = __riscv_vreinterpret_v_u8mf2_i8mf2(v143);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v145 = __riscv_vand_vx_u8mf2(v134, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v146 = __riscv_vmseq_vx_u8mf2_b16(v145, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v147 = __riscv_vadd_vx_i8mf2_mu(v146, v144, v144, -4, 8);
        vint16m1_t v148 = v70;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v149 = __riscv_vwmacc_vx_i16m1(v148, v85, v147, 8);
        v70 = v149;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v150 = __riscv_vsrl_vx_u8mf2(v131, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v151 = __riscv_vand_vx_u8mf2(v150, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v152 = __riscv_vreinterpret_v_u8mf2_i8mf2(v151);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v153 = __riscv_vand_vx_u8mf2(v134, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v154 = __riscv_vmseq_vx_u8mf2_b16(v153, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v155 = __riscv_vadd_vx_i8mf2_mu(v154, v152, v152, -4, 8);
        vint16m1_t v156 = v72;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v157 = __riscv_vwmacc_vx_i16m1(v156, v88, v155, 8);
        v72 = v157;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v158 = __riscv_vsrl_vx_u8mf2(v131, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v159 = __riscv_vand_vx_u8mf2(v158, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v160 = __riscv_vreinterpret_v_u8mf2_i8mf2(v159);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v161 = __riscv_vand_vx_u8mf2(v134, 8, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v162 = __riscv_vmseq_vx_u8mf2_b16(v161, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v163 = __riscv_vadd_vx_i8mf2_mu(v162, v160, v160, -4, 8);
        vint16m1_t v164 = v74;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v165 = __riscv_vwmacc_vx_i16m1(v164, v91, v163, 8);
        v74 = v165;
      }
      vint16m1_t v166 = v60;
      vint16m1_t v167 = v62;
      vint16m1_t v168 = v64;
      vint16m1_t v169 = v66;
      vint16m1_t v170 = v68;
      vint16m1_t v171 = v70;
      vint16m1_t v172 = v72;
      vint16m1_t v173 = v74;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v174 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v175 = __riscv_vwmacc_vv_i32m2(v174, v31, v166, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v176 = __riscv_vwmacc_vv_i32m2(v175, v35, v167, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v177 = __riscv_vwmacc_vv_i32m2(v176, v39, v168, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v178 = __riscv_vwmacc_vv_i32m2(v177, v43, v169, 8);
      v24 = v178;
      vint32m2_t v179 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v180 = __riscv_vwmacc_vv_i32m2(v179, v47, v170, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v181 = __riscv_vwmacc_vv_i32m2(v180, v51, v171, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v182 = __riscv_vwmacc_vv_i32m2(v181, v55, v172, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v183 = __riscv_vwmacc_vv_i32m2(v182, v59, v173, 8);
      v26 = v183;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
      const uint8_t* v184 = v19 + 48;
      const int8_t* v185 = (const int8_t*) v184;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v186 = __riscv_vle8_v_i8mf2(v185, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v187 = __riscv_vsext_vf2_i16m1(v186, 8);
      const uint8_t* v188 = v19 + 80;
      const int8_t* v189 = (const int8_t*) v188;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v190 = __riscv_vle8_v_i8mf2(v189, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v191 = __riscv_vsext_vf2_i16m1(v190, 8);
      const uint8_t* v192 = v19 + 112;
      const int8_t* v193 = (const int8_t*) v192;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v194 = __riscv_vle8_v_i8mf2(v193, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v195 = __riscv_vsext_vf2_i16m1(v194, 8);
      const uint8_t* v196 = v19 + 144;
      const int8_t* v197 = (const int8_t*) v196;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v198 = __riscv_vle8_v_i8mf2(v197, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v199 = __riscv_vsext_vf2_i16m1(v198, 8);
      const uint8_t* v200 = v19 + 56;
      const int8_t* v201 = (const int8_t*) v200;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v202 = __riscv_vle8_v_i8mf2(v201, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v203 = __riscv_vsext_vf2_i16m1(v202, 8);
      const uint8_t* v204 = v19 + 88;
      const int8_t* v205 = (const int8_t*) v204;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v206 = __riscv_vle8_v_i8mf2(v205, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v207 = __riscv_vsext_vf2_i16m1(v206, 8);
      const uint8_t* v208 = v19 + 120;
      const int8_t* v209 = (const int8_t*) v208;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v210 = __riscv_vle8_v_i8mf2(v209, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v211 = __riscv_vsext_vf2_i16m1(v210, 8);
      const uint8_t* v212 = v19 + 152;
      const int8_t* v213 = (const int8_t*) v212;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v214 = __riscv_vle8_v_i8mf2(v213, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v215 = __riscv_vsext_vf2_i16m1(v214, 8);
      vint16m1_t v216;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v217 = __riscv_vmv_v_x_i16m1(0, 8);
      v216 = v217;
      vint16m1_t v218;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v219 = __riscv_vmv_v_x_i16m1(0, 8);
      v218 = v219;
      vint16m1_t v220;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v221 = __riscv_vmv_v_x_i16m1(0, 8);
      v220 = v221;
      vint16m1_t v222;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v223 = __riscv_vmv_v_x_i16m1(0, 8);
      v222 = v223;
      vint16m1_t v224;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v225 = __riscv_vmv_v_x_i16m1(0, 8);
      v224 = v225;
      vint16m1_t v226;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v227 = __riscv_vmv_v_x_i16m1(0, 8);
      v226 = v227;
      vint16m1_t v228;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v229 = __riscv_vmv_v_x_i16m1(0, 8);
      v228 = v229;
      vint16m1_t v230;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v231 = __riscv_vmv_v_x_i16m1(0, 8);
      v230 = v231;
      for (size_t v232 = 0; v232 < 16; v232 += 1) {
        size_t v233 = v232 * 16;
        const uint8_t* v234 = v19 + v233;
        const uint8_t* v235 = v21 + v232;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v236 = v235 + 20;
        const int8_t* v237 = (const int8_t*) v236;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v238 = *(const int8_t *)(v237);
        const uint8_t* v239 = v235 + 52;
        const int8_t* v240 = (const int8_t*) v239;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v241 = *(const int8_t *)(v240);
        const uint8_t* v242 = v235 + 84;
        const int8_t* v243 = (const int8_t*) v242;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v244 = *(const int8_t *)(v243);
        const uint8_t* v245 = v235 + 116;
        const int8_t* v246 = (const int8_t*) v245;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v247 = *(const int8_t *)(v246);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
        const uint8_t* v248 = v234 + 1056;
        const uint8_t* v249 = (const uint8_t*) v248;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v250 = __riscv_vle8_v_u8mf2(v249, 8);
        const uint8_t* v251 = v234 + 544;
        const uint8_t* v252 = (const uint8_t*) v251;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v253 = __riscv_vle8_v_u8mf2(v252, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v254 = __riscv_vand_vx_u8mf2(v250, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v255 = __riscv_vreinterpret_v_u8mf2_i8mf2(v254);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v256 = __riscv_vand_vx_u8mf2(v253, 1, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v257 = __riscv_vmseq_vx_u8mf2_b16(v256, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v258 = __riscv_vadd_vx_i8mf2_mu(v257, v255, v255, -4, 8);
        vint16m1_t v259 = v216;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v260 = __riscv_vwmacc_vx_i16m1(v259, v238, v258, 8);
        v216 = v260;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v261 = __riscv_vsrl_vx_u8mf2(v250, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v262 = __riscv_vand_vx_u8mf2(v261, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v263 = __riscv_vreinterpret_v_u8mf2_i8mf2(v262);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v264 = __riscv_vand_vx_u8mf2(v253, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v265 = __riscv_vmseq_vx_u8mf2_b16(v264, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v266 = __riscv_vadd_vx_i8mf2_mu(v265, v263, v263, -4, 8);
        vint16m1_t v267 = v218;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v268 = __riscv_vwmacc_vx_i16m1(v267, v241, v266, 8);
        v218 = v268;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v269 = __riscv_vsrl_vx_u8mf2(v250, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v270 = __riscv_vand_vx_u8mf2(v269, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v271 = __riscv_vreinterpret_v_u8mf2_i8mf2(v270);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v272 = __riscv_vand_vx_u8mf2(v253, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v273 = __riscv_vmseq_vx_u8mf2_b16(v272, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v274 = __riscv_vadd_vx_i8mf2_mu(v273, v271, v271, -4, 8);
        vint16m1_t v275 = v220;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v276 = __riscv_vwmacc_vx_i16m1(v275, v244, v274, 8);
        v220 = v276;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v277 = __riscv_vsrl_vx_u8mf2(v250, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v278 = __riscv_vand_vx_u8mf2(v277, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v279 = __riscv_vreinterpret_v_u8mf2_i8mf2(v278);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v280 = __riscv_vand_vx_u8mf2(v253, 8, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v281 = __riscv_vmseq_vx_u8mf2_b16(v280, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v282 = __riscv_vadd_vx_i8mf2_mu(v281, v279, v279, -4, 8);
        vint16m1_t v283 = v222;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v284 = __riscv_vwmacc_vx_i16m1(v283, v247, v282, 8);
        v222 = v284;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
        const uint8_t* v285 = v234 + 1064;
        const uint8_t* v286 = (const uint8_t*) v285;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v287 = __riscv_vle8_v_u8mf2(v286, 8);
        const uint8_t* v288 = v234 + 552;
        const uint8_t* v289 = (const uint8_t*) v288;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v290 = __riscv_vle8_v_u8mf2(v289, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v291 = __riscv_vand_vx_u8mf2(v287, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v292 = __riscv_vreinterpret_v_u8mf2_i8mf2(v291);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v293 = __riscv_vand_vx_u8mf2(v290, 1, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v294 = __riscv_vmseq_vx_u8mf2_b16(v293, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v295 = __riscv_vadd_vx_i8mf2_mu(v294, v292, v292, -4, 8);
        vint16m1_t v296 = v224;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v297 = __riscv_vwmacc_vx_i16m1(v296, v238, v295, 8);
        v224 = v297;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v298 = __riscv_vsrl_vx_u8mf2(v287, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v299 = __riscv_vand_vx_u8mf2(v298, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v300 = __riscv_vreinterpret_v_u8mf2_i8mf2(v299);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v301 = __riscv_vand_vx_u8mf2(v290, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v302 = __riscv_vmseq_vx_u8mf2_b16(v301, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v303 = __riscv_vadd_vx_i8mf2_mu(v302, v300, v300, -4, 8);
        vint16m1_t v304 = v226;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v305 = __riscv_vwmacc_vx_i16m1(v304, v241, v303, 8);
        v226 = v305;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v306 = __riscv_vsrl_vx_u8mf2(v287, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v307 = __riscv_vand_vx_u8mf2(v306, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v308 = __riscv_vreinterpret_v_u8mf2_i8mf2(v307);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v309 = __riscv_vand_vx_u8mf2(v290, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v310 = __riscv_vmseq_vx_u8mf2_b16(v309, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v311 = __riscv_vadd_vx_i8mf2_mu(v310, v308, v308, -4, 8);
        vint16m1_t v312 = v228;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v313 = __riscv_vwmacc_vx_i16m1(v312, v244, v311, 8);
        v228 = v313;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v314 = __riscv_vsrl_vx_u8mf2(v287, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v315 = __riscv_vand_vx_u8mf2(v314, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v316 = __riscv_vreinterpret_v_u8mf2_i8mf2(v315);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v317 = __riscv_vand_vx_u8mf2(v290, 8, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v318 = __riscv_vmseq_vx_u8mf2_b16(v317, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v319 = __riscv_vadd_vx_i8mf2_mu(v318, v316, v316, -4, 8);
        vint16m1_t v320 = v230;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v321 = __riscv_vwmacc_vx_i16m1(v320, v247, v319, 8);
        v230 = v321;
      }
      vint16m1_t v322 = v216;
      vint16m1_t v323 = v218;
      vint16m1_t v324 = v220;
      vint16m1_t v325 = v222;
      vint16m1_t v326 = v224;
      vint16m1_t v327 = v226;
      vint16m1_t v328 = v228;
      vint16m1_t v329 = v230;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v330 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v331 = __riscv_vwmacc_vv_i32m2(v330, v187, v322, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v332 = __riscv_vwmacc_vv_i32m2(v331, v191, v323, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v333 = __riscv_vwmacc_vv_i32m2(v332, v195, v324, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v334 = __riscv_vwmacc_vv_i32m2(v333, v199, v325, 8);
      v24 = v334;
      vint32m2_t v335 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v336 = __riscv_vwmacc_vv_i32m2(v335, v203, v326, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v337 = __riscv_vwmacc_vv_i32m2(v336, v207, v327, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v338 = __riscv_vwmacc_vv_i32m2(v337, v211, v328, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v339 = __riscv_vwmacc_vv_i32m2(v338, v215, v329, 8);
      v26 = v339;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
      const uint8_t* v340 = v19 + 160;
      const int8_t* v341 = (const int8_t*) v340;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v342 = __riscv_vle8_v_i8mf2(v341, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v343 = __riscv_vsext_vf2_i16m1(v342, 8);
      const uint8_t* v344 = v19 + 192;
      const int8_t* v345 = (const int8_t*) v344;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v346 = __riscv_vle8_v_i8mf2(v345, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v347 = __riscv_vsext_vf2_i16m1(v346, 8);
      const uint8_t* v348 = v19 + 224;
      const int8_t* v349 = (const int8_t*) v348;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v350 = __riscv_vle8_v_i8mf2(v349, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v351 = __riscv_vsext_vf2_i16m1(v350, 8);
      const uint8_t* v352 = v19 + 256;
      const int8_t* v353 = (const int8_t*) v352;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v354 = __riscv_vle8_v_i8mf2(v353, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v355 = __riscv_vsext_vf2_i16m1(v354, 8);
      const uint8_t* v356 = v19 + 168;
      const int8_t* v357 = (const int8_t*) v356;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v358 = __riscv_vle8_v_i8mf2(v357, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v359 = __riscv_vsext_vf2_i16m1(v358, 8);
      const uint8_t* v360 = v19 + 200;
      const int8_t* v361 = (const int8_t*) v360;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v362 = __riscv_vle8_v_i8mf2(v361, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v363 = __riscv_vsext_vf2_i16m1(v362, 8);
      const uint8_t* v364 = v19 + 232;
      const int8_t* v365 = (const int8_t*) v364;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v366 = __riscv_vle8_v_i8mf2(v365, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v367 = __riscv_vsext_vf2_i16m1(v366, 8);
      const uint8_t* v368 = v19 + 264;
      const int8_t* v369 = (const int8_t*) v368;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v370 = __riscv_vle8_v_i8mf2(v369, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v371 = __riscv_vsext_vf2_i16m1(v370, 8);
      vint16m1_t v372;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v373 = __riscv_vmv_v_x_i16m1(0, 8);
      v372 = v373;
      vint16m1_t v374;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v375 = __riscv_vmv_v_x_i16m1(0, 8);
      v374 = v375;
      vint16m1_t v376;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v377 = __riscv_vmv_v_x_i16m1(0, 8);
      v376 = v377;
      vint16m1_t v378;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v379 = __riscv_vmv_v_x_i16m1(0, 8);
      v378 = v379;
      vint16m1_t v380;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v381 = __riscv_vmv_v_x_i16m1(0, 8);
      v380 = v381;
      vint16m1_t v382;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v383 = __riscv_vmv_v_x_i16m1(0, 8);
      v382 = v383;
      vint16m1_t v384;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v385 = __riscv_vmv_v_x_i16m1(0, 8);
      v384 = v385;
      vint16m1_t v386;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v387 = __riscv_vmv_v_x_i16m1(0, 8);
      v386 = v387;
      for (size_t v388 = 0; v388 < 16; v388 += 1) {
        size_t v389 = v388 * 16;
        const uint8_t* v390 = v19 + v389;
        const uint8_t* v391 = v21 + v388;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v392 = v391 + 132;
        const int8_t* v393 = (const int8_t*) v392;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v394 = *(const int8_t *)(v393);
        const uint8_t* v395 = v391 + 164;
        const int8_t* v396 = (const int8_t*) v395;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v397 = *(const int8_t *)(v396);
        const uint8_t* v398 = v391 + 196;
        const int8_t* v399 = (const int8_t*) v398;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v400 = *(const int8_t *)(v399);
        const uint8_t* v401 = v391 + 228;
        const int8_t* v402 = (const int8_t*) v401;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v403 = *(const int8_t *)(v402);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
        const uint8_t* v404 = v390 + 1312;
        const uint8_t* v405 = (const uint8_t*) v404;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v406 = __riscv_vle8_v_u8mf2(v405, 8);
        const uint8_t* v407 = v390 + 288;
        const uint8_t* v408 = (const uint8_t*) v407;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v409 = __riscv_vle8_v_u8mf2(v408, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v410 = __riscv_vand_vx_u8mf2(v406, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v411 = __riscv_vreinterpret_v_u8mf2_i8mf2(v410);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v412 = __riscv_vand_vx_u8mf2(v409, 16, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v413 = __riscv_vmseq_vx_u8mf2_b16(v412, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v414 = __riscv_vadd_vx_i8mf2_mu(v413, v411, v411, -4, 8);
        vint16m1_t v415 = v372;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v416 = __riscv_vwmacc_vx_i16m1(v415, v394, v414, 8);
        v372 = v416;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v417 = __riscv_vsrl_vx_u8mf2(v406, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v418 = __riscv_vand_vx_u8mf2(v417, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v419 = __riscv_vreinterpret_v_u8mf2_i8mf2(v418);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v420 = __riscv_vand_vx_u8mf2(v409, 32, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v421 = __riscv_vmseq_vx_u8mf2_b16(v420, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v422 = __riscv_vadd_vx_i8mf2_mu(v421, v419, v419, -4, 8);
        vint16m1_t v423 = v374;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v424 = __riscv_vwmacc_vx_i16m1(v423, v397, v422, 8);
        v374 = v424;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v425 = __riscv_vsrl_vx_u8mf2(v406, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v426 = __riscv_vand_vx_u8mf2(v425, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v427 = __riscv_vreinterpret_v_u8mf2_i8mf2(v426);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v428 = __riscv_vand_vx_u8mf2(v409, 64, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v429 = __riscv_vmseq_vx_u8mf2_b16(v428, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v430 = __riscv_vadd_vx_i8mf2_mu(v429, v427, v427, -4, 8);
        vint16m1_t v431 = v376;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v432 = __riscv_vwmacc_vx_i16m1(v431, v400, v430, 8);
        v376 = v432;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v433 = __riscv_vsrl_vx_u8mf2(v406, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v434 = __riscv_vand_vx_u8mf2(v433, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v435 = __riscv_vreinterpret_v_u8mf2_i8mf2(v434);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v436 = __riscv_vand_vx_u8mf2(v409, 128, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v437 = __riscv_vmseq_vx_u8mf2_b16(v436, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v438 = __riscv_vadd_vx_i8mf2_mu(v437, v435, v435, -4, 8);
        vint16m1_t v439 = v378;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v440 = __riscv_vwmacc_vx_i16m1(v439, v403, v438, 8);
        v378 = v440;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
        const uint8_t* v441 = v390 + 1320;
        const uint8_t* v442 = (const uint8_t*) v441;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v443 = __riscv_vle8_v_u8mf2(v442, 8);
        const uint8_t* v444 = v390 + 296;
        const uint8_t* v445 = (const uint8_t*) v444;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v446 = __riscv_vle8_v_u8mf2(v445, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v447 = __riscv_vand_vx_u8mf2(v443, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v448 = __riscv_vreinterpret_v_u8mf2_i8mf2(v447);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v449 = __riscv_vand_vx_u8mf2(v446, 16, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v450 = __riscv_vmseq_vx_u8mf2_b16(v449, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v451 = __riscv_vadd_vx_i8mf2_mu(v450, v448, v448, -4, 8);
        vint16m1_t v452 = v380;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v453 = __riscv_vwmacc_vx_i16m1(v452, v394, v451, 8);
        v380 = v453;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v454 = __riscv_vsrl_vx_u8mf2(v443, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v455 = __riscv_vand_vx_u8mf2(v454, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v456 = __riscv_vreinterpret_v_u8mf2_i8mf2(v455);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v457 = __riscv_vand_vx_u8mf2(v446, 32, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v458 = __riscv_vmseq_vx_u8mf2_b16(v457, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v459 = __riscv_vadd_vx_i8mf2_mu(v458, v456, v456, -4, 8);
        vint16m1_t v460 = v382;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v461 = __riscv_vwmacc_vx_i16m1(v460, v397, v459, 8);
        v382 = v461;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v462 = __riscv_vsrl_vx_u8mf2(v443, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v463 = __riscv_vand_vx_u8mf2(v462, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v464 = __riscv_vreinterpret_v_u8mf2_i8mf2(v463);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v465 = __riscv_vand_vx_u8mf2(v446, 64, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v466 = __riscv_vmseq_vx_u8mf2_b16(v465, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v467 = __riscv_vadd_vx_i8mf2_mu(v466, v464, v464, -4, 8);
        vint16m1_t v468 = v384;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v469 = __riscv_vwmacc_vx_i16m1(v468, v400, v467, 8);
        v384 = v469;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v470 = __riscv_vsrl_vx_u8mf2(v443, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v471 = __riscv_vand_vx_u8mf2(v470, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v472 = __riscv_vreinterpret_v_u8mf2_i8mf2(v471);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v473 = __riscv_vand_vx_u8mf2(v446, 128, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v474 = __riscv_vmseq_vx_u8mf2_b16(v473, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v475 = __riscv_vadd_vx_i8mf2_mu(v474, v472, v472, -4, 8);
        vint16m1_t v476 = v386;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v477 = __riscv_vwmacc_vx_i16m1(v476, v403, v475, 8);
        v386 = v477;
      }
      vint16m1_t v478 = v372;
      vint16m1_t v479 = v374;
      vint16m1_t v480 = v376;
      vint16m1_t v481 = v378;
      vint16m1_t v482 = v380;
      vint16m1_t v483 = v382;
      vint16m1_t v484 = v384;
      vint16m1_t v485 = v386;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v486 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v487 = __riscv_vwmacc_vv_i32m2(v486, v343, v478, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v488 = __riscv_vwmacc_vv_i32m2(v487, v347, v479, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v489 = __riscv_vwmacc_vv_i32m2(v488, v351, v480, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v490 = __riscv_vwmacc_vv_i32m2(v489, v355, v481, 8);
      v24 = v490;
      vint32m2_t v491 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v492 = __riscv_vwmacc_vv_i32m2(v491, v359, v482, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v493 = __riscv_vwmacc_vv_i32m2(v492, v363, v483, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v494 = __riscv_vwmacc_vv_i32m2(v493, v367, v484, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v495 = __riscv_vwmacc_vv_i32m2(v494, v371, v485, 8);
      v26 = v495;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
      const uint8_t* v496 = v19 + 176;
      const int8_t* v497 = (const int8_t*) v496;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v498 = __riscv_vle8_v_i8mf2(v497, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v499 = __riscv_vsext_vf2_i16m1(v498, 8);
      const uint8_t* v500 = v19 + 208;
      const int8_t* v501 = (const int8_t*) v500;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v502 = __riscv_vle8_v_i8mf2(v501, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v503 = __riscv_vsext_vf2_i16m1(v502, 8);
      const uint8_t* v504 = v19 + 240;
      const int8_t* v505 = (const int8_t*) v504;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v506 = __riscv_vle8_v_i8mf2(v505, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v507 = __riscv_vsext_vf2_i16m1(v506, 8);
      const uint8_t* v508 = v19 + 272;
      const int8_t* v509 = (const int8_t*) v508;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v510 = __riscv_vle8_v_i8mf2(v509, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v511 = __riscv_vsext_vf2_i16m1(v510, 8);
      const uint8_t* v512 = v19 + 184;
      const int8_t* v513 = (const int8_t*) v512;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v514 = __riscv_vle8_v_i8mf2(v513, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v515 = __riscv_vsext_vf2_i16m1(v514, 8);
      const uint8_t* v516 = v19 + 216;
      const int8_t* v517 = (const int8_t*) v516;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v518 = __riscv_vle8_v_i8mf2(v517, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v519 = __riscv_vsext_vf2_i16m1(v518, 8);
      const uint8_t* v520 = v19 + 248;
      const int8_t* v521 = (const int8_t*) v520;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v522 = __riscv_vle8_v_i8mf2(v521, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v523 = __riscv_vsext_vf2_i16m1(v522, 8);
      const uint8_t* v524 = v19 + 280;
      const int8_t* v525 = (const int8_t*) v524;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v526 = __riscv_vle8_v_i8mf2(v525, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v527 = __riscv_vsext_vf2_i16m1(v526, 8);
      vint16m1_t v528;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v529 = __riscv_vmv_v_x_i16m1(0, 8);
      v528 = v529;
      vint16m1_t v530;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v531 = __riscv_vmv_v_x_i16m1(0, 8);
      v530 = v531;
      vint16m1_t v532;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v533 = __riscv_vmv_v_x_i16m1(0, 8);
      v532 = v533;
      vint16m1_t v534;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v535 = __riscv_vmv_v_x_i16m1(0, 8);
      v534 = v535;
      vint16m1_t v536;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v537 = __riscv_vmv_v_x_i16m1(0, 8);
      v536 = v537;
      vint16m1_t v538;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v539 = __riscv_vmv_v_x_i16m1(0, 8);
      v538 = v539;
      vint16m1_t v540;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v541 = __riscv_vmv_v_x_i16m1(0, 8);
      v540 = v541;
      vint16m1_t v542;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v543 = __riscv_vmv_v_x_i16m1(0, 8);
      v542 = v543;
      for (size_t v544 = 0; v544 < 16; v544 += 1) {
        size_t v545 = v544 * 16;
        const uint8_t* v546 = v19 + v545;
        const uint8_t* v547 = v21 + v544;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v548 = v547 + 148;
        const int8_t* v549 = (const int8_t*) v548;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v550 = *(const int8_t *)(v549);
        const uint8_t* v551 = v547 + 180;
        const int8_t* v552 = (const int8_t*) v551;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v553 = *(const int8_t *)(v552);
        const uint8_t* v554 = v547 + 212;
        const int8_t* v555 = (const int8_t*) v554;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v556 = *(const int8_t *)(v555);
        const uint8_t* v557 = v547 + 244;
        const int8_t* v558 = (const int8_t*) v557;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v559 = *(const int8_t *)(v558);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
        const uint8_t* v560 = v546 + 1568;
        const uint8_t* v561 = (const uint8_t*) v560;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v562 = __riscv_vle8_v_u8mf2(v561, 8);
        const uint8_t* v563 = v546 + 544;
        const uint8_t* v564 = (const uint8_t*) v563;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v565 = __riscv_vle8_v_u8mf2(v564, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v566 = __riscv_vand_vx_u8mf2(v562, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v567 = __riscv_vreinterpret_v_u8mf2_i8mf2(v566);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v568 = __riscv_vand_vx_u8mf2(v565, 16, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v569 = __riscv_vmseq_vx_u8mf2_b16(v568, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v570 = __riscv_vadd_vx_i8mf2_mu(v569, v567, v567, -4, 8);
        vint16m1_t v571 = v528;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v572 = __riscv_vwmacc_vx_i16m1(v571, v550, v570, 8);
        v528 = v572;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v573 = __riscv_vsrl_vx_u8mf2(v562, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v574 = __riscv_vand_vx_u8mf2(v573, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v575 = __riscv_vreinterpret_v_u8mf2_i8mf2(v574);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v576 = __riscv_vand_vx_u8mf2(v565, 32, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v577 = __riscv_vmseq_vx_u8mf2_b16(v576, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v578 = __riscv_vadd_vx_i8mf2_mu(v577, v575, v575, -4, 8);
        vint16m1_t v579 = v530;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v580 = __riscv_vwmacc_vx_i16m1(v579, v553, v578, 8);
        v530 = v580;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v581 = __riscv_vsrl_vx_u8mf2(v562, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v582 = __riscv_vand_vx_u8mf2(v581, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v583 = __riscv_vreinterpret_v_u8mf2_i8mf2(v582);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v584 = __riscv_vand_vx_u8mf2(v565, 64, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v585 = __riscv_vmseq_vx_u8mf2_b16(v584, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v586 = __riscv_vadd_vx_i8mf2_mu(v585, v583, v583, -4, 8);
        vint16m1_t v587 = v532;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v588 = __riscv_vwmacc_vx_i16m1(v587, v556, v586, 8);
        v532 = v588;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v589 = __riscv_vsrl_vx_u8mf2(v562, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v590 = __riscv_vand_vx_u8mf2(v589, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v591 = __riscv_vreinterpret_v_u8mf2_i8mf2(v590);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v592 = __riscv_vand_vx_u8mf2(v565, 128, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v593 = __riscv_vmseq_vx_u8mf2_b16(v592, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v594 = __riscv_vadd_vx_i8mf2_mu(v593, v591, v591, -4, 8);
        vint16m1_t v595 = v534;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v596 = __riscv_vwmacc_vx_i16m1(v595, v559, v594, 8);
        v534 = v596;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
        const uint8_t* v597 = v546 + 1576;
        const uint8_t* v598 = (const uint8_t*) v597;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v599 = __riscv_vle8_v_u8mf2(v598, 8);
        const uint8_t* v600 = v546 + 552;
        const uint8_t* v601 = (const uint8_t*) v600;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v602 = __riscv_vle8_v_u8mf2(v601, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v603 = __riscv_vand_vx_u8mf2(v599, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v604 = __riscv_vreinterpret_v_u8mf2_i8mf2(v603);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v605 = __riscv_vand_vx_u8mf2(v602, 16, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v606 = __riscv_vmseq_vx_u8mf2_b16(v605, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v607 = __riscv_vadd_vx_i8mf2_mu(v606, v604, v604, -4, 8);
        vint16m1_t v608 = v536;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v609 = __riscv_vwmacc_vx_i16m1(v608, v550, v607, 8);
        v536 = v609;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v610 = __riscv_vsrl_vx_u8mf2(v599, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v611 = __riscv_vand_vx_u8mf2(v610, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v612 = __riscv_vreinterpret_v_u8mf2_i8mf2(v611);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v613 = __riscv_vand_vx_u8mf2(v602, 32, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v614 = __riscv_vmseq_vx_u8mf2_b16(v613, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v615 = __riscv_vadd_vx_i8mf2_mu(v614, v612, v612, -4, 8);
        vint16m1_t v616 = v538;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v617 = __riscv_vwmacc_vx_i16m1(v616, v553, v615, 8);
        v538 = v617;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v618 = __riscv_vsrl_vx_u8mf2(v599, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v619 = __riscv_vand_vx_u8mf2(v618, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v620 = __riscv_vreinterpret_v_u8mf2_i8mf2(v619);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v621 = __riscv_vand_vx_u8mf2(v602, 64, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v622 = __riscv_vmseq_vx_u8mf2_b16(v621, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v623 = __riscv_vadd_vx_i8mf2_mu(v622, v620, v620, -4, 8);
        vint16m1_t v624 = v540;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v625 = __riscv_vwmacc_vx_i16m1(v624, v556, v623, 8);
        v540 = v625;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v626 = __riscv_vsrl_vx_u8mf2(v599, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v627 = __riscv_vand_vx_u8mf2(v626, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v628 = __riscv_vreinterpret_v_u8mf2_i8mf2(v627);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v629 = __riscv_vand_vx_u8mf2(v602, 128, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
        vbool16_t v630 = __riscv_vmseq_vx_u8mf2_b16(v629, 0, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
        vint8mf2_t v631 = __riscv_vadd_vx_i8mf2_mu(v630, v628, v628, -4, 8);
        vint16m1_t v632 = v542;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v633 = __riscv_vwmacc_vx_i16m1(v632, v559, v631, 8);
        v542 = v633;
      }
      vint16m1_t v634 = v528;
      vint16m1_t v635 = v530;
      vint16m1_t v636 = v532;
      vint16m1_t v637 = v534;
      vint16m1_t v638 = v536;
      vint16m1_t v639 = v538;
      vint16m1_t v640 = v540;
      vint16m1_t v641 = v542;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v642 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v643 = __riscv_vwmacc_vv_i32m2(v642, v499, v634, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v644 = __riscv_vwmacc_vv_i32m2(v643, v503, v635, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v645 = __riscv_vwmacc_vv_i32m2(v644, v507, v636, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v646 = __riscv_vwmacc_vv_i32m2(v645, v511, v637, 8);
      v24 = v646;
      vint32m2_t v647 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v648 = __riscv_vwmacc_vv_i32m2(v647, v515, v638, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v649 = __riscv_vwmacc_vv_i32m2(v648, v519, v639, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v650 = __riscv_vwmacc_vv_i32m2(v649, v523, v640, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v651 = __riscv_vwmacc_vv_i32m2(v650, v527, v641, 8);
      v26 = v651;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v652 = (const _Float16*) v19;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v653 = __riscv_vle16_v_f16m1(v652, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v654 = __riscv_vfwcvt_f_f_v_f32m2(v653, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v655 = __riscv_vfmul_vf_f32m2(v654, v23, 8);
      vint32m2_t v656 = v24;
      vfloat32m2_t v657 = v13;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v658 = __riscv_vfcvt_f_x_v_f32m2(v656, 8);
      vfloat32m2_t v659 = __riscv_vfmacc_vv_f32m2(v657, v658, v655, 8);
      v13 = v659;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v660 = v19 + 16;
      const _Float16* v661 = (const _Float16*) v660;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v662 = __riscv_vle16_v_f16m1(v661, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v663 = __riscv_vfwcvt_f_f_v_f32m2(v662, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v664 = __riscv_vfmul_vf_f32m2(v663, v23, 8);
      vint32m2_t v665 = v26;
      vfloat32m2_t v666 = v15;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v667 = __riscv_vfcvt_f_x_v_f32m2(v665, 8);
      vfloat32m2_t v668 = __riscv_vfmacc_vv_f32m2(v666, v667, v664, 8);
      v15 = v668;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v669 = v9 * 16;
    float* v670 = v2 + v669;
    vfloat32m2_t v671 = v13;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v670, v671, 8);
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v672 = v9 * 16;
    size_t v673 = v672 + 8;
    float* v674 = v2 + v673;
    vfloat32m2_t v675 = v15;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v674, v675, 8);
  }
  return;
}


