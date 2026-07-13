#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemv_q6_K_q8_K_kernel_ggml_repack_gemv_q6_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
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
    size_t v11 = v10 * 3360;
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
      size_t v18 = v17 * 3360;
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
      for (size_t v76 = 0; v76 < 8; v76 += 1) {
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
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v92 = v78 + 1312;
        const uint8_t* v93 = (const uint8_t*) v92;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v94 = __riscv_vle8_v_u8mf2(v93, 8);
        const uint8_t* v95 = v78 + 1824;
        const uint8_t* v96 = (const uint8_t*) v95;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v97 = __riscv_vle8_v_u8mf2(v96, 8);
        const uint8_t* v98 = v78 + 288;
        const uint8_t* v99 = (const uint8_t*) v98;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v100 = __riscv_vle8_v_u8mf2(v99, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v101 = __riscv_vand_vx_u8mf2(v94, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v102 = __riscv_vand_vx_u8mf2(v100, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v103 = __riscv_vsll_vx_u8mf2(v102, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v104 = __riscv_vor_vv_u8mf2(v101, v103, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v105 = __riscv_vreinterpret_v_u8mf2_i8mf2(v104);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v106 = __riscv_vsub_vx_i8mf2(v105, 32, 8);
        vint16m1_t v107 = v60;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v108 = __riscv_vwmacc_vx_i16m1(v107, v82, v106, 8);
        v60 = v108;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v109 = __riscv_vand_vx_u8mf2(v97, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v110 = __riscv_vsrl_vx_u8mf2(v100, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v111 = __riscv_vand_vx_u8mf2(v110, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v112 = __riscv_vsll_vx_u8mf2(v111, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v113 = __riscv_vor_vv_u8mf2(v109, v112, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v114 = __riscv_vreinterpret_v_u8mf2_i8mf2(v113);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v115 = __riscv_vsub_vx_i8mf2(v114, 32, 8);
        vint16m1_t v116 = v62;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v117 = __riscv_vwmacc_vx_i16m1(v116, v85, v115, 8);
        v62 = v117;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v118 = __riscv_vsrl_vx_u8mf2(v94, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v119 = __riscv_vsrl_vx_u8mf2(v100, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v120 = __riscv_vand_vx_u8mf2(v119, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v121 = __riscv_vsll_vx_u8mf2(v120, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v122 = __riscv_vor_vv_u8mf2(v118, v121, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v123 = __riscv_vreinterpret_v_u8mf2_i8mf2(v122);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v124 = __riscv_vsub_vx_i8mf2(v123, 32, 8);
        vint16m1_t v125 = v64;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v126 = __riscv_vwmacc_vx_i16m1(v125, v88, v124, 8);
        v64 = v126;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v127 = __riscv_vsrl_vx_u8mf2(v97, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v128 = __riscv_vsrl_vx_u8mf2(v100, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v129 = __riscv_vand_vx_u8mf2(v128, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v130 = __riscv_vsll_vx_u8mf2(v129, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v131 = __riscv_vor_vv_u8mf2(v127, v130, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v132 = __riscv_vreinterpret_v_u8mf2_i8mf2(v131);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v133 = __riscv_vsub_vx_i8mf2(v132, 32, 8);
        vint16m1_t v134 = v66;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v135 = __riscv_vwmacc_vx_i16m1(v134, v91, v133, 8);
        v66 = v135;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v136 = v78 + 1320;
        const uint8_t* v137 = (const uint8_t*) v136;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v138 = __riscv_vle8_v_u8mf2(v137, 8);
        const uint8_t* v139 = v78 + 1832;
        const uint8_t* v140 = (const uint8_t*) v139;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v141 = __riscv_vle8_v_u8mf2(v140, 8);
        const uint8_t* v142 = v78 + 296;
        const uint8_t* v143 = (const uint8_t*) v142;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v144 = __riscv_vle8_v_u8mf2(v143, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v145 = __riscv_vand_vx_u8mf2(v138, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v146 = __riscv_vand_vx_u8mf2(v144, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v147 = __riscv_vsll_vx_u8mf2(v146, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v148 = __riscv_vor_vv_u8mf2(v145, v147, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v149 = __riscv_vreinterpret_v_u8mf2_i8mf2(v148);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v150 = __riscv_vsub_vx_i8mf2(v149, 32, 8);
        vint16m1_t v151 = v68;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v152 = __riscv_vwmacc_vx_i16m1(v151, v82, v150, 8);
        v68 = v152;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v153 = __riscv_vand_vx_u8mf2(v141, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v154 = __riscv_vsrl_vx_u8mf2(v144, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v155 = __riscv_vand_vx_u8mf2(v154, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v156 = __riscv_vsll_vx_u8mf2(v155, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v157 = __riscv_vor_vv_u8mf2(v153, v156, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v158 = __riscv_vreinterpret_v_u8mf2_i8mf2(v157);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v159 = __riscv_vsub_vx_i8mf2(v158, 32, 8);
        vint16m1_t v160 = v70;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v161 = __riscv_vwmacc_vx_i16m1(v160, v85, v159, 8);
        v70 = v161;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v162 = __riscv_vsrl_vx_u8mf2(v138, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v163 = __riscv_vsrl_vx_u8mf2(v144, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v164 = __riscv_vand_vx_u8mf2(v163, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v165 = __riscv_vsll_vx_u8mf2(v164, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v166 = __riscv_vor_vv_u8mf2(v162, v165, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v167 = __riscv_vreinterpret_v_u8mf2_i8mf2(v166);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v168 = __riscv_vsub_vx_i8mf2(v167, 32, 8);
        vint16m1_t v169 = v72;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v170 = __riscv_vwmacc_vx_i16m1(v169, v88, v168, 8);
        v72 = v170;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v171 = __riscv_vsrl_vx_u8mf2(v141, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v172 = __riscv_vsrl_vx_u8mf2(v144, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v173 = __riscv_vand_vx_u8mf2(v172, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v174 = __riscv_vsll_vx_u8mf2(v173, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v175 = __riscv_vor_vv_u8mf2(v171, v174, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v176 = __riscv_vreinterpret_v_u8mf2_i8mf2(v175);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v177 = __riscv_vsub_vx_i8mf2(v176, 32, 8);
        vint16m1_t v178 = v74;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v179 = __riscv_vwmacc_vx_i16m1(v178, v91, v177, 8);
        v74 = v179;
      }
      vint16m1_t v180 = v60;
      vint16m1_t v181 = v62;
      vint16m1_t v182 = v64;
      vint16m1_t v183 = v66;
      vint16m1_t v184 = v68;
      vint16m1_t v185 = v70;
      vint16m1_t v186 = v72;
      vint16m1_t v187 = v74;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v188 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v189 = __riscv_vwmacc_vv_i32m2(v188, v31, v180, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v190 = __riscv_vwmacc_vv_i32m2(v189, v35, v181, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v191 = __riscv_vwmacc_vv_i32m2(v190, v39, v182, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v192 = __riscv_vwmacc_vv_i32m2(v191, v43, v183, 8);
      v24 = v192;
      vint32m2_t v193 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v194 = __riscv_vwmacc_vv_i32m2(v193, v47, v184, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v195 = __riscv_vwmacc_vv_i32m2(v194, v51, v185, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v196 = __riscv_vwmacc_vv_i32m2(v195, v55, v186, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v197 = __riscv_vwmacc_vv_i32m2(v196, v59, v187, 8);
      v26 = v197;
      vint16m1_t v198;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v199 = __riscv_vmv_v_x_i16m1(0, 8);
      v198 = v199;
      vint16m1_t v200;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v201 = __riscv_vmv_v_x_i16m1(0, 8);
      v200 = v201;
      vint16m1_t v202;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v203 = __riscv_vmv_v_x_i16m1(0, 8);
      v202 = v203;
      vint16m1_t v204;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v205 = __riscv_vmv_v_x_i16m1(0, 8);
      v204 = v205;
      vint16m1_t v206;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v207 = __riscv_vmv_v_x_i16m1(0, 8);
      v206 = v207;
      vint16m1_t v208;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v209 = __riscv_vmv_v_x_i16m1(0, 8);
      v208 = v209;
      vint16m1_t v210;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v211 = __riscv_vmv_v_x_i16m1(0, 8);
      v210 = v211;
      vint16m1_t v212;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v213 = __riscv_vmv_v_x_i16m1(0, 8);
      v212 = v213;
      for (size_t v214 = 0; v214 < 8; v214 += 1) {
        size_t v215 = v214 * 16;
        const uint8_t* v216 = v19 + v215;
        const uint8_t* v217 = v21 + v214;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v218 = v217 + 12;
        const int8_t* v219 = (const int8_t*) v218;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v220 = *(const int8_t *)(v219);
        const uint8_t* v221 = v217 + 44;
        const int8_t* v222 = (const int8_t*) v221;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v223 = *(const int8_t *)(v222);
        const uint8_t* v224 = v217 + 76;
        const int8_t* v225 = (const int8_t*) v224;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v226 = *(const int8_t *)(v225);
        const uint8_t* v227 = v217 + 108;
        const int8_t* v228 = (const int8_t*) v227;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v229 = *(const int8_t *)(v228);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v230 = v216 + 1440;
        const uint8_t* v231 = (const uint8_t*) v230;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v232 = __riscv_vle8_v_u8mf2(v231, 8);
        const uint8_t* v233 = v216 + 1952;
        const uint8_t* v234 = (const uint8_t*) v233;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v235 = __riscv_vle8_v_u8mf2(v234, 8);
        const uint8_t* v236 = v216 + 416;
        const uint8_t* v237 = (const uint8_t*) v236;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v238 = __riscv_vle8_v_u8mf2(v237, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v239 = __riscv_vand_vx_u8mf2(v232, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v240 = __riscv_vand_vx_u8mf2(v238, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v241 = __riscv_vsll_vx_u8mf2(v240, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v242 = __riscv_vor_vv_u8mf2(v239, v241, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v243 = __riscv_vreinterpret_v_u8mf2_i8mf2(v242);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v244 = __riscv_vsub_vx_i8mf2(v243, 32, 8);
        vint16m1_t v245 = v198;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v246 = __riscv_vwmacc_vx_i16m1(v245, v220, v244, 8);
        v198 = v246;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v247 = __riscv_vand_vx_u8mf2(v235, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v248 = __riscv_vsrl_vx_u8mf2(v238, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v249 = __riscv_vand_vx_u8mf2(v248, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v250 = __riscv_vsll_vx_u8mf2(v249, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v251 = __riscv_vor_vv_u8mf2(v247, v250, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v252 = __riscv_vreinterpret_v_u8mf2_i8mf2(v251);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v253 = __riscv_vsub_vx_i8mf2(v252, 32, 8);
        vint16m1_t v254 = v200;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v255 = __riscv_vwmacc_vx_i16m1(v254, v223, v253, 8);
        v200 = v255;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v256 = __riscv_vsrl_vx_u8mf2(v232, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v257 = __riscv_vsrl_vx_u8mf2(v238, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v258 = __riscv_vand_vx_u8mf2(v257, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v259 = __riscv_vsll_vx_u8mf2(v258, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v260 = __riscv_vor_vv_u8mf2(v256, v259, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v261 = __riscv_vreinterpret_v_u8mf2_i8mf2(v260);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v262 = __riscv_vsub_vx_i8mf2(v261, 32, 8);
        vint16m1_t v263 = v202;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v264 = __riscv_vwmacc_vx_i16m1(v263, v226, v262, 8);
        v202 = v264;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v265 = __riscv_vsrl_vx_u8mf2(v235, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v266 = __riscv_vsrl_vx_u8mf2(v238, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v267 = __riscv_vand_vx_u8mf2(v266, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v268 = __riscv_vsll_vx_u8mf2(v267, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v269 = __riscv_vor_vv_u8mf2(v265, v268, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v270 = __riscv_vreinterpret_v_u8mf2_i8mf2(v269);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v271 = __riscv_vsub_vx_i8mf2(v270, 32, 8);
        vint16m1_t v272 = v204;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v273 = __riscv_vwmacc_vx_i16m1(v272, v229, v271, 8);
        v204 = v273;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v274 = v216 + 1448;
        const uint8_t* v275 = (const uint8_t*) v274;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v276 = __riscv_vle8_v_u8mf2(v275, 8);
        const uint8_t* v277 = v216 + 1960;
        const uint8_t* v278 = (const uint8_t*) v277;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v279 = __riscv_vle8_v_u8mf2(v278, 8);
        const uint8_t* v280 = v216 + 424;
        const uint8_t* v281 = (const uint8_t*) v280;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v282 = __riscv_vle8_v_u8mf2(v281, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v283 = __riscv_vand_vx_u8mf2(v276, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v284 = __riscv_vand_vx_u8mf2(v282, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v285 = __riscv_vsll_vx_u8mf2(v284, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v286 = __riscv_vor_vv_u8mf2(v283, v285, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v287 = __riscv_vreinterpret_v_u8mf2_i8mf2(v286);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v288 = __riscv_vsub_vx_i8mf2(v287, 32, 8);
        vint16m1_t v289 = v206;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v290 = __riscv_vwmacc_vx_i16m1(v289, v220, v288, 8);
        v206 = v290;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v291 = __riscv_vand_vx_u8mf2(v279, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v292 = __riscv_vsrl_vx_u8mf2(v282, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v293 = __riscv_vand_vx_u8mf2(v292, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v294 = __riscv_vsll_vx_u8mf2(v293, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v295 = __riscv_vor_vv_u8mf2(v291, v294, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v296 = __riscv_vreinterpret_v_u8mf2_i8mf2(v295);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v297 = __riscv_vsub_vx_i8mf2(v296, 32, 8);
        vint16m1_t v298 = v208;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v299 = __riscv_vwmacc_vx_i16m1(v298, v223, v297, 8);
        v208 = v299;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v300 = __riscv_vsrl_vx_u8mf2(v276, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v301 = __riscv_vsrl_vx_u8mf2(v282, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v302 = __riscv_vand_vx_u8mf2(v301, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v303 = __riscv_vsll_vx_u8mf2(v302, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v304 = __riscv_vor_vv_u8mf2(v300, v303, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v305 = __riscv_vreinterpret_v_u8mf2_i8mf2(v304);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v306 = __riscv_vsub_vx_i8mf2(v305, 32, 8);
        vint16m1_t v307 = v210;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v308 = __riscv_vwmacc_vx_i16m1(v307, v226, v306, 8);
        v210 = v308;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v309 = __riscv_vsrl_vx_u8mf2(v279, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v310 = __riscv_vsrl_vx_u8mf2(v282, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v311 = __riscv_vand_vx_u8mf2(v310, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v312 = __riscv_vsll_vx_u8mf2(v311, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v313 = __riscv_vor_vv_u8mf2(v309, v312, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v314 = __riscv_vreinterpret_v_u8mf2_i8mf2(v313);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v315 = __riscv_vsub_vx_i8mf2(v314, 32, 8);
        vint16m1_t v316 = v212;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v317 = __riscv_vwmacc_vx_i16m1(v316, v229, v315, 8);
        v212 = v317;
      }
      vint16m1_t v318 = v198;
      vint16m1_t v319 = v200;
      vint16m1_t v320 = v202;
      vint16m1_t v321 = v204;
      vint16m1_t v322 = v206;
      vint16m1_t v323 = v208;
      vint16m1_t v324 = v210;
      vint16m1_t v325 = v212;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v326 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v327 = __riscv_vwmacc_vv_i32m2(v326, v31, v318, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v328 = __riscv_vwmacc_vv_i32m2(v327, v35, v319, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v329 = __riscv_vwmacc_vv_i32m2(v328, v39, v320, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v330 = __riscv_vwmacc_vv_i32m2(v329, v43, v321, 8);
      v24 = v330;
      vint32m2_t v331 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v332 = __riscv_vwmacc_vv_i32m2(v331, v47, v322, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v333 = __riscv_vwmacc_vv_i32m2(v332, v51, v323, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v334 = __riscv_vwmacc_vv_i32m2(v333, v55, v324, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v335 = __riscv_vwmacc_vv_i32m2(v334, v59, v325, 8);
      v26 = v335;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
      const uint8_t* v336 = v19 + 48;
      const int8_t* v337 = (const int8_t*) v336;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v338 = __riscv_vle8_v_i8mf2(v337, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v339 = __riscv_vsext_vf2_i16m1(v338, 8);
      const uint8_t* v340 = v19 + 80;
      const int8_t* v341 = (const int8_t*) v340;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v342 = __riscv_vle8_v_i8mf2(v341, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v343 = __riscv_vsext_vf2_i16m1(v342, 8);
      const uint8_t* v344 = v19 + 112;
      const int8_t* v345 = (const int8_t*) v344;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v346 = __riscv_vle8_v_i8mf2(v345, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v347 = __riscv_vsext_vf2_i16m1(v346, 8);
      const uint8_t* v348 = v19 + 144;
      const int8_t* v349 = (const int8_t*) v348;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v350 = __riscv_vle8_v_i8mf2(v349, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v351 = __riscv_vsext_vf2_i16m1(v350, 8);
      const uint8_t* v352 = v19 + 56;
      const int8_t* v353 = (const int8_t*) v352;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v354 = __riscv_vle8_v_i8mf2(v353, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v355 = __riscv_vsext_vf2_i16m1(v354, 8);
      const uint8_t* v356 = v19 + 88;
      const int8_t* v357 = (const int8_t*) v356;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v358 = __riscv_vle8_v_i8mf2(v357, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v359 = __riscv_vsext_vf2_i16m1(v358, 8);
      const uint8_t* v360 = v19 + 120;
      const int8_t* v361 = (const int8_t*) v360;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v362 = __riscv_vle8_v_i8mf2(v361, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v363 = __riscv_vsext_vf2_i16m1(v362, 8);
      const uint8_t* v364 = v19 + 152;
      const int8_t* v365 = (const int8_t*) v364;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v366 = __riscv_vle8_v_i8mf2(v365, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v367 = __riscv_vsext_vf2_i16m1(v366, 8);
      vint16m1_t v368;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v369 = __riscv_vmv_v_x_i16m1(0, 8);
      v368 = v369;
      vint16m1_t v370;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v371 = __riscv_vmv_v_x_i16m1(0, 8);
      v370 = v371;
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
      for (size_t v384 = 0; v384 < 8; v384 += 1) {
        size_t v385 = v384 * 16;
        const uint8_t* v386 = v19 + v385;
        const uint8_t* v387 = v21 + v384;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v388 = v387 + 20;
        const int8_t* v389 = (const int8_t*) v388;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v390 = *(const int8_t *)(v389);
        const uint8_t* v391 = v387 + 52;
        const int8_t* v392 = (const int8_t*) v391;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v393 = *(const int8_t *)(v392);
        const uint8_t* v394 = v387 + 84;
        const int8_t* v395 = (const int8_t*) v394;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v396 = *(const int8_t *)(v395);
        const uint8_t* v397 = v387 + 116;
        const int8_t* v398 = (const int8_t*) v397;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v399 = *(const int8_t *)(v398);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v400 = v386 + 1568;
        const uint8_t* v401 = (const uint8_t*) v400;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v402 = __riscv_vle8_v_u8mf2(v401, 8);
        const uint8_t* v403 = v386 + 2080;
        const uint8_t* v404 = (const uint8_t*) v403;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v405 = __riscv_vle8_v_u8mf2(v404, 8);
        const uint8_t* v406 = v386 + 544;
        const uint8_t* v407 = (const uint8_t*) v406;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v408 = __riscv_vle8_v_u8mf2(v407, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v409 = __riscv_vand_vx_u8mf2(v402, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v410 = __riscv_vand_vx_u8mf2(v408, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v411 = __riscv_vsll_vx_u8mf2(v410, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v412 = __riscv_vor_vv_u8mf2(v409, v411, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v413 = __riscv_vreinterpret_v_u8mf2_i8mf2(v412);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v414 = __riscv_vsub_vx_i8mf2(v413, 32, 8);
        vint16m1_t v415 = v368;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v416 = __riscv_vwmacc_vx_i16m1(v415, v390, v414, 8);
        v368 = v416;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v417 = __riscv_vand_vx_u8mf2(v405, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v418 = __riscv_vsrl_vx_u8mf2(v408, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v419 = __riscv_vand_vx_u8mf2(v418, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v420 = __riscv_vsll_vx_u8mf2(v419, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v421 = __riscv_vor_vv_u8mf2(v417, v420, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v422 = __riscv_vreinterpret_v_u8mf2_i8mf2(v421);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v423 = __riscv_vsub_vx_i8mf2(v422, 32, 8);
        vint16m1_t v424 = v370;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v425 = __riscv_vwmacc_vx_i16m1(v424, v393, v423, 8);
        v370 = v425;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v426 = __riscv_vsrl_vx_u8mf2(v402, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v427 = __riscv_vsrl_vx_u8mf2(v408, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v428 = __riscv_vand_vx_u8mf2(v427, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v429 = __riscv_vsll_vx_u8mf2(v428, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v430 = __riscv_vor_vv_u8mf2(v426, v429, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v431 = __riscv_vreinterpret_v_u8mf2_i8mf2(v430);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v432 = __riscv_vsub_vx_i8mf2(v431, 32, 8);
        vint16m1_t v433 = v372;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v434 = __riscv_vwmacc_vx_i16m1(v433, v396, v432, 8);
        v372 = v434;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v435 = __riscv_vsrl_vx_u8mf2(v405, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v436 = __riscv_vsrl_vx_u8mf2(v408, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v437 = __riscv_vand_vx_u8mf2(v436, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v438 = __riscv_vsll_vx_u8mf2(v437, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v439 = __riscv_vor_vv_u8mf2(v435, v438, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v440 = __riscv_vreinterpret_v_u8mf2_i8mf2(v439);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v441 = __riscv_vsub_vx_i8mf2(v440, 32, 8);
        vint16m1_t v442 = v374;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v443 = __riscv_vwmacc_vx_i16m1(v442, v399, v441, 8);
        v374 = v443;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v444 = v386 + 1576;
        const uint8_t* v445 = (const uint8_t*) v444;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v446 = __riscv_vle8_v_u8mf2(v445, 8);
        const uint8_t* v447 = v386 + 2088;
        const uint8_t* v448 = (const uint8_t*) v447;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v449 = __riscv_vle8_v_u8mf2(v448, 8);
        const uint8_t* v450 = v386 + 552;
        const uint8_t* v451 = (const uint8_t*) v450;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v452 = __riscv_vle8_v_u8mf2(v451, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v453 = __riscv_vand_vx_u8mf2(v446, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v454 = __riscv_vand_vx_u8mf2(v452, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v455 = __riscv_vsll_vx_u8mf2(v454, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v456 = __riscv_vor_vv_u8mf2(v453, v455, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v457 = __riscv_vreinterpret_v_u8mf2_i8mf2(v456);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v458 = __riscv_vsub_vx_i8mf2(v457, 32, 8);
        vint16m1_t v459 = v376;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v460 = __riscv_vwmacc_vx_i16m1(v459, v390, v458, 8);
        v376 = v460;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v461 = __riscv_vand_vx_u8mf2(v449, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v462 = __riscv_vsrl_vx_u8mf2(v452, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v463 = __riscv_vand_vx_u8mf2(v462, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v464 = __riscv_vsll_vx_u8mf2(v463, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v465 = __riscv_vor_vv_u8mf2(v461, v464, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v466 = __riscv_vreinterpret_v_u8mf2_i8mf2(v465);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v467 = __riscv_vsub_vx_i8mf2(v466, 32, 8);
        vint16m1_t v468 = v378;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v469 = __riscv_vwmacc_vx_i16m1(v468, v393, v467, 8);
        v378 = v469;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v470 = __riscv_vsrl_vx_u8mf2(v446, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v471 = __riscv_vsrl_vx_u8mf2(v452, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v472 = __riscv_vand_vx_u8mf2(v471, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v473 = __riscv_vsll_vx_u8mf2(v472, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v474 = __riscv_vor_vv_u8mf2(v470, v473, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v475 = __riscv_vreinterpret_v_u8mf2_i8mf2(v474);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v476 = __riscv_vsub_vx_i8mf2(v475, 32, 8);
        vint16m1_t v477 = v380;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v478 = __riscv_vwmacc_vx_i16m1(v477, v396, v476, 8);
        v380 = v478;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v479 = __riscv_vsrl_vx_u8mf2(v449, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v480 = __riscv_vsrl_vx_u8mf2(v452, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v481 = __riscv_vand_vx_u8mf2(v480, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v482 = __riscv_vsll_vx_u8mf2(v481, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v483 = __riscv_vor_vv_u8mf2(v479, v482, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v484 = __riscv_vreinterpret_v_u8mf2_i8mf2(v483);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v485 = __riscv_vsub_vx_i8mf2(v484, 32, 8);
        vint16m1_t v486 = v382;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v487 = __riscv_vwmacc_vx_i16m1(v486, v399, v485, 8);
        v382 = v487;
      }
      vint16m1_t v488 = v368;
      vint16m1_t v489 = v370;
      vint16m1_t v490 = v372;
      vint16m1_t v491 = v374;
      vint16m1_t v492 = v376;
      vint16m1_t v493 = v378;
      vint16m1_t v494 = v380;
      vint16m1_t v495 = v382;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v496 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v497 = __riscv_vwmacc_vv_i32m2(v496, v339, v488, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v498 = __riscv_vwmacc_vv_i32m2(v497, v343, v489, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v499 = __riscv_vwmacc_vv_i32m2(v498, v347, v490, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v500 = __riscv_vwmacc_vv_i32m2(v499, v351, v491, 8);
      v24 = v500;
      vint32m2_t v501 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v502 = __riscv_vwmacc_vv_i32m2(v501, v355, v492, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v503 = __riscv_vwmacc_vv_i32m2(v502, v359, v493, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v504 = __riscv_vwmacc_vv_i32m2(v503, v363, v494, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v505 = __riscv_vwmacc_vv_i32m2(v504, v367, v495, 8);
      v26 = v505;
      vint16m1_t v506;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v507 = __riscv_vmv_v_x_i16m1(0, 8);
      v506 = v507;
      vint16m1_t v508;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v509 = __riscv_vmv_v_x_i16m1(0, 8);
      v508 = v509;
      vint16m1_t v510;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v511 = __riscv_vmv_v_x_i16m1(0, 8);
      v510 = v511;
      vint16m1_t v512;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v513 = __riscv_vmv_v_x_i16m1(0, 8);
      v512 = v513;
      vint16m1_t v514;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v515 = __riscv_vmv_v_x_i16m1(0, 8);
      v514 = v515;
      vint16m1_t v516;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v517 = __riscv_vmv_v_x_i16m1(0, 8);
      v516 = v517;
      vint16m1_t v518;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v519 = __riscv_vmv_v_x_i16m1(0, 8);
      v518 = v519;
      vint16m1_t v520;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v521 = __riscv_vmv_v_x_i16m1(0, 8);
      v520 = v521;
      for (size_t v522 = 0; v522 < 8; v522 += 1) {
        size_t v523 = v522 * 16;
        const uint8_t* v524 = v19 + v523;
        const uint8_t* v525 = v21 + v522;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v526 = v525 + 28;
        const int8_t* v527 = (const int8_t*) v526;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v528 = *(const int8_t *)(v527);
        const uint8_t* v529 = v525 + 60;
        const int8_t* v530 = (const int8_t*) v529;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v531 = *(const int8_t *)(v530);
        const uint8_t* v532 = v525 + 92;
        const int8_t* v533 = (const int8_t*) v532;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v534 = *(const int8_t *)(v533);
        const uint8_t* v535 = v525 + 124;
        const int8_t* v536 = (const int8_t*) v535;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v537 = *(const int8_t *)(v536);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v538 = v524 + 1696;
        const uint8_t* v539 = (const uint8_t*) v538;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v540 = __riscv_vle8_v_u8mf2(v539, 8);
        const uint8_t* v541 = v524 + 2208;
        const uint8_t* v542 = (const uint8_t*) v541;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v543 = __riscv_vle8_v_u8mf2(v542, 8);
        const uint8_t* v544 = v524 + 672;
        const uint8_t* v545 = (const uint8_t*) v544;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v546 = __riscv_vle8_v_u8mf2(v545, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v547 = __riscv_vand_vx_u8mf2(v540, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v548 = __riscv_vand_vx_u8mf2(v546, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v549 = __riscv_vsll_vx_u8mf2(v548, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v550 = __riscv_vor_vv_u8mf2(v547, v549, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v551 = __riscv_vreinterpret_v_u8mf2_i8mf2(v550);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v552 = __riscv_vsub_vx_i8mf2(v551, 32, 8);
        vint16m1_t v553 = v506;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v554 = __riscv_vwmacc_vx_i16m1(v553, v528, v552, 8);
        v506 = v554;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v555 = __riscv_vand_vx_u8mf2(v543, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v556 = __riscv_vsrl_vx_u8mf2(v546, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v557 = __riscv_vand_vx_u8mf2(v556, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v558 = __riscv_vsll_vx_u8mf2(v557, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v559 = __riscv_vor_vv_u8mf2(v555, v558, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v560 = __riscv_vreinterpret_v_u8mf2_i8mf2(v559);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v561 = __riscv_vsub_vx_i8mf2(v560, 32, 8);
        vint16m1_t v562 = v508;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v563 = __riscv_vwmacc_vx_i16m1(v562, v531, v561, 8);
        v508 = v563;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v564 = __riscv_vsrl_vx_u8mf2(v540, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v565 = __riscv_vsrl_vx_u8mf2(v546, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v566 = __riscv_vand_vx_u8mf2(v565, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v567 = __riscv_vsll_vx_u8mf2(v566, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v568 = __riscv_vor_vv_u8mf2(v564, v567, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v569 = __riscv_vreinterpret_v_u8mf2_i8mf2(v568);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v570 = __riscv_vsub_vx_i8mf2(v569, 32, 8);
        vint16m1_t v571 = v510;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v572 = __riscv_vwmacc_vx_i16m1(v571, v534, v570, 8);
        v510 = v572;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v573 = __riscv_vsrl_vx_u8mf2(v543, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v574 = __riscv_vsrl_vx_u8mf2(v546, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v575 = __riscv_vand_vx_u8mf2(v574, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v576 = __riscv_vsll_vx_u8mf2(v575, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v577 = __riscv_vor_vv_u8mf2(v573, v576, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v578 = __riscv_vreinterpret_v_u8mf2_i8mf2(v577);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v579 = __riscv_vsub_vx_i8mf2(v578, 32, 8);
        vint16m1_t v580 = v512;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v581 = __riscv_vwmacc_vx_i16m1(v580, v537, v579, 8);
        v512 = v581;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v582 = v524 + 1704;
        const uint8_t* v583 = (const uint8_t*) v582;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v584 = __riscv_vle8_v_u8mf2(v583, 8);
        const uint8_t* v585 = v524 + 2216;
        const uint8_t* v586 = (const uint8_t*) v585;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v587 = __riscv_vle8_v_u8mf2(v586, 8);
        const uint8_t* v588 = v524 + 680;
        const uint8_t* v589 = (const uint8_t*) v588;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v590 = __riscv_vle8_v_u8mf2(v589, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v591 = __riscv_vand_vx_u8mf2(v584, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v592 = __riscv_vand_vx_u8mf2(v590, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v593 = __riscv_vsll_vx_u8mf2(v592, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v594 = __riscv_vor_vv_u8mf2(v591, v593, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v595 = __riscv_vreinterpret_v_u8mf2_i8mf2(v594);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v596 = __riscv_vsub_vx_i8mf2(v595, 32, 8);
        vint16m1_t v597 = v514;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v598 = __riscv_vwmacc_vx_i16m1(v597, v528, v596, 8);
        v514 = v598;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v599 = __riscv_vand_vx_u8mf2(v587, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v600 = __riscv_vsrl_vx_u8mf2(v590, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v601 = __riscv_vand_vx_u8mf2(v600, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v602 = __riscv_vsll_vx_u8mf2(v601, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v603 = __riscv_vor_vv_u8mf2(v599, v602, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v604 = __riscv_vreinterpret_v_u8mf2_i8mf2(v603);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v605 = __riscv_vsub_vx_i8mf2(v604, 32, 8);
        vint16m1_t v606 = v516;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v607 = __riscv_vwmacc_vx_i16m1(v606, v531, v605, 8);
        v516 = v607;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v608 = __riscv_vsrl_vx_u8mf2(v584, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v609 = __riscv_vsrl_vx_u8mf2(v590, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v610 = __riscv_vand_vx_u8mf2(v609, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v611 = __riscv_vsll_vx_u8mf2(v610, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v612 = __riscv_vor_vv_u8mf2(v608, v611, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v613 = __riscv_vreinterpret_v_u8mf2_i8mf2(v612);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v614 = __riscv_vsub_vx_i8mf2(v613, 32, 8);
        vint16m1_t v615 = v518;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v616 = __riscv_vwmacc_vx_i16m1(v615, v534, v614, 8);
        v518 = v616;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v617 = __riscv_vsrl_vx_u8mf2(v587, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v618 = __riscv_vsrl_vx_u8mf2(v590, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v619 = __riscv_vand_vx_u8mf2(v618, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v620 = __riscv_vsll_vx_u8mf2(v619, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v621 = __riscv_vor_vv_u8mf2(v617, v620, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v622 = __riscv_vreinterpret_v_u8mf2_i8mf2(v621);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v623 = __riscv_vsub_vx_i8mf2(v622, 32, 8);
        vint16m1_t v624 = v520;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v625 = __riscv_vwmacc_vx_i16m1(v624, v537, v623, 8);
        v520 = v625;
      }
      vint16m1_t v626 = v506;
      vint16m1_t v627 = v508;
      vint16m1_t v628 = v510;
      vint16m1_t v629 = v512;
      vint16m1_t v630 = v514;
      vint16m1_t v631 = v516;
      vint16m1_t v632 = v518;
      vint16m1_t v633 = v520;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v634 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v635 = __riscv_vwmacc_vv_i32m2(v634, v339, v626, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v636 = __riscv_vwmacc_vv_i32m2(v635, v343, v627, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v637 = __riscv_vwmacc_vv_i32m2(v636, v347, v628, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v638 = __riscv_vwmacc_vv_i32m2(v637, v351, v629, 8);
      v24 = v638;
      vint32m2_t v639 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v640 = __riscv_vwmacc_vv_i32m2(v639, v355, v630, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v641 = __riscv_vwmacc_vv_i32m2(v640, v359, v631, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v642 = __riscv_vwmacc_vv_i32m2(v641, v363, v632, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v643 = __riscv_vwmacc_vv_i32m2(v642, v367, v633, 8);
      v26 = v643;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
      const uint8_t* v644 = v19 + 160;
      const int8_t* v645 = (const int8_t*) v644;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v646 = __riscv_vle8_v_i8mf2(v645, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v647 = __riscv_vsext_vf2_i16m1(v646, 8);
      const uint8_t* v648 = v19 + 192;
      const int8_t* v649 = (const int8_t*) v648;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v650 = __riscv_vle8_v_i8mf2(v649, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v651 = __riscv_vsext_vf2_i16m1(v650, 8);
      const uint8_t* v652 = v19 + 224;
      const int8_t* v653 = (const int8_t*) v652;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v654 = __riscv_vle8_v_i8mf2(v653, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v655 = __riscv_vsext_vf2_i16m1(v654, 8);
      const uint8_t* v656 = v19 + 256;
      const int8_t* v657 = (const int8_t*) v656;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v658 = __riscv_vle8_v_i8mf2(v657, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v659 = __riscv_vsext_vf2_i16m1(v658, 8);
      const uint8_t* v660 = v19 + 168;
      const int8_t* v661 = (const int8_t*) v660;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v662 = __riscv_vle8_v_i8mf2(v661, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v663 = __riscv_vsext_vf2_i16m1(v662, 8);
      const uint8_t* v664 = v19 + 200;
      const int8_t* v665 = (const int8_t*) v664;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v666 = __riscv_vle8_v_i8mf2(v665, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v667 = __riscv_vsext_vf2_i16m1(v666, 8);
      const uint8_t* v668 = v19 + 232;
      const int8_t* v669 = (const int8_t*) v668;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v670 = __riscv_vle8_v_i8mf2(v669, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v671 = __riscv_vsext_vf2_i16m1(v670, 8);
      const uint8_t* v672 = v19 + 264;
      const int8_t* v673 = (const int8_t*) v672;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v674 = __riscv_vle8_v_i8mf2(v673, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v675 = __riscv_vsext_vf2_i16m1(v674, 8);
      vint16m1_t v676;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v677 = __riscv_vmv_v_x_i16m1(0, 8);
      v676 = v677;
      vint16m1_t v678;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v679 = __riscv_vmv_v_x_i16m1(0, 8);
      v678 = v679;
      vint16m1_t v680;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v681 = __riscv_vmv_v_x_i16m1(0, 8);
      v680 = v681;
      vint16m1_t v682;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v683 = __riscv_vmv_v_x_i16m1(0, 8);
      v682 = v683;
      vint16m1_t v684;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v685 = __riscv_vmv_v_x_i16m1(0, 8);
      v684 = v685;
      vint16m1_t v686;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v687 = __riscv_vmv_v_x_i16m1(0, 8);
      v686 = v687;
      vint16m1_t v688;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v689 = __riscv_vmv_v_x_i16m1(0, 8);
      v688 = v689;
      vint16m1_t v690;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v691 = __riscv_vmv_v_x_i16m1(0, 8);
      v690 = v691;
      for (size_t v692 = 0; v692 < 8; v692 += 1) {
        size_t v693 = v692 * 16;
        const uint8_t* v694 = v19 + v693;
        const uint8_t* v695 = v21 + v692;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v696 = v695 + 132;
        const int8_t* v697 = (const int8_t*) v696;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v698 = *(const int8_t *)(v697);
        const uint8_t* v699 = v695 + 164;
        const int8_t* v700 = (const int8_t*) v699;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v701 = *(const int8_t *)(v700);
        const uint8_t* v702 = v695 + 196;
        const int8_t* v703 = (const int8_t*) v702;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v704 = *(const int8_t *)(v703);
        const uint8_t* v705 = v695 + 228;
        const int8_t* v706 = (const int8_t*) v705;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v707 = *(const int8_t *)(v706);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v708 = v694 + 2336;
        const uint8_t* v709 = (const uint8_t*) v708;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v710 = __riscv_vle8_v_u8mf2(v709, 8);
        const uint8_t* v711 = v694 + 2848;
        const uint8_t* v712 = (const uint8_t*) v711;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v713 = __riscv_vle8_v_u8mf2(v712, 8);
        const uint8_t* v714 = v694 + 800;
        const uint8_t* v715 = (const uint8_t*) v714;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v716 = __riscv_vle8_v_u8mf2(v715, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v717 = __riscv_vand_vx_u8mf2(v710, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v718 = __riscv_vand_vx_u8mf2(v716, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v719 = __riscv_vsll_vx_u8mf2(v718, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v720 = __riscv_vor_vv_u8mf2(v717, v719, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v721 = __riscv_vreinterpret_v_u8mf2_i8mf2(v720);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v722 = __riscv_vsub_vx_i8mf2(v721, 32, 8);
        vint16m1_t v723 = v676;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v724 = __riscv_vwmacc_vx_i16m1(v723, v698, v722, 8);
        v676 = v724;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v725 = __riscv_vand_vx_u8mf2(v713, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v726 = __riscv_vsrl_vx_u8mf2(v716, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v727 = __riscv_vand_vx_u8mf2(v726, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v728 = __riscv_vsll_vx_u8mf2(v727, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v729 = __riscv_vor_vv_u8mf2(v725, v728, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v730 = __riscv_vreinterpret_v_u8mf2_i8mf2(v729);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v731 = __riscv_vsub_vx_i8mf2(v730, 32, 8);
        vint16m1_t v732 = v678;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v733 = __riscv_vwmacc_vx_i16m1(v732, v701, v731, 8);
        v678 = v733;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v734 = __riscv_vsrl_vx_u8mf2(v710, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v735 = __riscv_vsrl_vx_u8mf2(v716, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v736 = __riscv_vand_vx_u8mf2(v735, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v737 = __riscv_vsll_vx_u8mf2(v736, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v738 = __riscv_vor_vv_u8mf2(v734, v737, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v739 = __riscv_vreinterpret_v_u8mf2_i8mf2(v738);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v740 = __riscv_vsub_vx_i8mf2(v739, 32, 8);
        vint16m1_t v741 = v680;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v742 = __riscv_vwmacc_vx_i16m1(v741, v704, v740, 8);
        v680 = v742;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v743 = __riscv_vsrl_vx_u8mf2(v713, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v744 = __riscv_vsrl_vx_u8mf2(v716, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v745 = __riscv_vand_vx_u8mf2(v744, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v746 = __riscv_vsll_vx_u8mf2(v745, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v747 = __riscv_vor_vv_u8mf2(v743, v746, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v748 = __riscv_vreinterpret_v_u8mf2_i8mf2(v747);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v749 = __riscv_vsub_vx_i8mf2(v748, 32, 8);
        vint16m1_t v750 = v682;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v751 = __riscv_vwmacc_vx_i16m1(v750, v707, v749, 8);
        v682 = v751;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v752 = v694 + 2344;
        const uint8_t* v753 = (const uint8_t*) v752;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v754 = __riscv_vle8_v_u8mf2(v753, 8);
        const uint8_t* v755 = v694 + 2856;
        const uint8_t* v756 = (const uint8_t*) v755;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v757 = __riscv_vle8_v_u8mf2(v756, 8);
        const uint8_t* v758 = v694 + 808;
        const uint8_t* v759 = (const uint8_t*) v758;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v760 = __riscv_vle8_v_u8mf2(v759, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v761 = __riscv_vand_vx_u8mf2(v754, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v762 = __riscv_vand_vx_u8mf2(v760, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v763 = __riscv_vsll_vx_u8mf2(v762, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v764 = __riscv_vor_vv_u8mf2(v761, v763, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v765 = __riscv_vreinterpret_v_u8mf2_i8mf2(v764);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v766 = __riscv_vsub_vx_i8mf2(v765, 32, 8);
        vint16m1_t v767 = v684;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v768 = __riscv_vwmacc_vx_i16m1(v767, v698, v766, 8);
        v684 = v768;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v769 = __riscv_vand_vx_u8mf2(v757, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v770 = __riscv_vsrl_vx_u8mf2(v760, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v771 = __riscv_vand_vx_u8mf2(v770, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v772 = __riscv_vsll_vx_u8mf2(v771, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v773 = __riscv_vor_vv_u8mf2(v769, v772, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v774 = __riscv_vreinterpret_v_u8mf2_i8mf2(v773);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v775 = __riscv_vsub_vx_i8mf2(v774, 32, 8);
        vint16m1_t v776 = v686;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v777 = __riscv_vwmacc_vx_i16m1(v776, v701, v775, 8);
        v686 = v777;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v778 = __riscv_vsrl_vx_u8mf2(v754, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v779 = __riscv_vsrl_vx_u8mf2(v760, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v780 = __riscv_vand_vx_u8mf2(v779, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v781 = __riscv_vsll_vx_u8mf2(v780, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v782 = __riscv_vor_vv_u8mf2(v778, v781, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v783 = __riscv_vreinterpret_v_u8mf2_i8mf2(v782);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v784 = __riscv_vsub_vx_i8mf2(v783, 32, 8);
        vint16m1_t v785 = v688;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v786 = __riscv_vwmacc_vx_i16m1(v785, v704, v784, 8);
        v688 = v786;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v787 = __riscv_vsrl_vx_u8mf2(v757, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v788 = __riscv_vsrl_vx_u8mf2(v760, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v789 = __riscv_vand_vx_u8mf2(v788, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v790 = __riscv_vsll_vx_u8mf2(v789, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v791 = __riscv_vor_vv_u8mf2(v787, v790, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v792 = __riscv_vreinterpret_v_u8mf2_i8mf2(v791);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v793 = __riscv_vsub_vx_i8mf2(v792, 32, 8);
        vint16m1_t v794 = v690;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v795 = __riscv_vwmacc_vx_i16m1(v794, v707, v793, 8);
        v690 = v795;
      }
      vint16m1_t v796 = v676;
      vint16m1_t v797 = v678;
      vint16m1_t v798 = v680;
      vint16m1_t v799 = v682;
      vint16m1_t v800 = v684;
      vint16m1_t v801 = v686;
      vint16m1_t v802 = v688;
      vint16m1_t v803 = v690;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v804 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v805 = __riscv_vwmacc_vv_i32m2(v804, v647, v796, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v806 = __riscv_vwmacc_vv_i32m2(v805, v651, v797, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v807 = __riscv_vwmacc_vv_i32m2(v806, v655, v798, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v808 = __riscv_vwmacc_vv_i32m2(v807, v659, v799, 8);
      v24 = v808;
      vint32m2_t v809 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v810 = __riscv_vwmacc_vv_i32m2(v809, v663, v800, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v811 = __riscv_vwmacc_vv_i32m2(v810, v667, v801, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v812 = __riscv_vwmacc_vv_i32m2(v811, v671, v802, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v813 = __riscv_vwmacc_vv_i32m2(v812, v675, v803, 8);
      v26 = v813;
      vint16m1_t v814;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v815 = __riscv_vmv_v_x_i16m1(0, 8);
      v814 = v815;
      vint16m1_t v816;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v817 = __riscv_vmv_v_x_i16m1(0, 8);
      v816 = v817;
      vint16m1_t v818;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v819 = __riscv_vmv_v_x_i16m1(0, 8);
      v818 = v819;
      vint16m1_t v820;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v821 = __riscv_vmv_v_x_i16m1(0, 8);
      v820 = v821;
      vint16m1_t v822;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v823 = __riscv_vmv_v_x_i16m1(0, 8);
      v822 = v823;
      vint16m1_t v824;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v825 = __riscv_vmv_v_x_i16m1(0, 8);
      v824 = v825;
      vint16m1_t v826;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v827 = __riscv_vmv_v_x_i16m1(0, 8);
      v826 = v827;
      vint16m1_t v828;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v829 = __riscv_vmv_v_x_i16m1(0, 8);
      v828 = v829;
      for (size_t v830 = 0; v830 < 8; v830 += 1) {
        size_t v831 = v830 * 16;
        const uint8_t* v832 = v19 + v831;
        const uint8_t* v833 = v21 + v830;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v834 = v833 + 140;
        const int8_t* v835 = (const int8_t*) v834;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v836 = *(const int8_t *)(v835);
        const uint8_t* v837 = v833 + 172;
        const int8_t* v838 = (const int8_t*) v837;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v839 = *(const int8_t *)(v838);
        const uint8_t* v840 = v833 + 204;
        const int8_t* v841 = (const int8_t*) v840;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v842 = *(const int8_t *)(v841);
        const uint8_t* v843 = v833 + 236;
        const int8_t* v844 = (const int8_t*) v843;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v845 = *(const int8_t *)(v844);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v846 = v832 + 2464;
        const uint8_t* v847 = (const uint8_t*) v846;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v848 = __riscv_vle8_v_u8mf2(v847, 8);
        const uint8_t* v849 = v832 + 2976;
        const uint8_t* v850 = (const uint8_t*) v849;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v851 = __riscv_vle8_v_u8mf2(v850, 8);
        const uint8_t* v852 = v832 + 928;
        const uint8_t* v853 = (const uint8_t*) v852;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v854 = __riscv_vle8_v_u8mf2(v853, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v855 = __riscv_vand_vx_u8mf2(v848, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v856 = __riscv_vand_vx_u8mf2(v854, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v857 = __riscv_vsll_vx_u8mf2(v856, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v858 = __riscv_vor_vv_u8mf2(v855, v857, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v859 = __riscv_vreinterpret_v_u8mf2_i8mf2(v858);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v860 = __riscv_vsub_vx_i8mf2(v859, 32, 8);
        vint16m1_t v861 = v814;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v862 = __riscv_vwmacc_vx_i16m1(v861, v836, v860, 8);
        v814 = v862;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v863 = __riscv_vand_vx_u8mf2(v851, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v864 = __riscv_vsrl_vx_u8mf2(v854, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v865 = __riscv_vand_vx_u8mf2(v864, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v866 = __riscv_vsll_vx_u8mf2(v865, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v867 = __riscv_vor_vv_u8mf2(v863, v866, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v868 = __riscv_vreinterpret_v_u8mf2_i8mf2(v867);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v869 = __riscv_vsub_vx_i8mf2(v868, 32, 8);
        vint16m1_t v870 = v816;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v871 = __riscv_vwmacc_vx_i16m1(v870, v839, v869, 8);
        v816 = v871;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v872 = __riscv_vsrl_vx_u8mf2(v848, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v873 = __riscv_vsrl_vx_u8mf2(v854, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v874 = __riscv_vand_vx_u8mf2(v873, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v875 = __riscv_vsll_vx_u8mf2(v874, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v876 = __riscv_vor_vv_u8mf2(v872, v875, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v877 = __riscv_vreinterpret_v_u8mf2_i8mf2(v876);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v878 = __riscv_vsub_vx_i8mf2(v877, 32, 8);
        vint16m1_t v879 = v818;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v880 = __riscv_vwmacc_vx_i16m1(v879, v842, v878, 8);
        v818 = v880;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v881 = __riscv_vsrl_vx_u8mf2(v851, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v882 = __riscv_vsrl_vx_u8mf2(v854, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v883 = __riscv_vand_vx_u8mf2(v882, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v884 = __riscv_vsll_vx_u8mf2(v883, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v885 = __riscv_vor_vv_u8mf2(v881, v884, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v886 = __riscv_vreinterpret_v_u8mf2_i8mf2(v885);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v887 = __riscv_vsub_vx_i8mf2(v886, 32, 8);
        vint16m1_t v888 = v820;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v889 = __riscv_vwmacc_vx_i16m1(v888, v845, v887, 8);
        v820 = v889;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v890 = v832 + 2472;
        const uint8_t* v891 = (const uint8_t*) v890;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v892 = __riscv_vle8_v_u8mf2(v891, 8);
        const uint8_t* v893 = v832 + 2984;
        const uint8_t* v894 = (const uint8_t*) v893;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v895 = __riscv_vle8_v_u8mf2(v894, 8);
        const uint8_t* v896 = v832 + 936;
        const uint8_t* v897 = (const uint8_t*) v896;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v898 = __riscv_vle8_v_u8mf2(v897, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v899 = __riscv_vand_vx_u8mf2(v892, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v900 = __riscv_vand_vx_u8mf2(v898, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v901 = __riscv_vsll_vx_u8mf2(v900, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v902 = __riscv_vor_vv_u8mf2(v899, v901, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v903 = __riscv_vreinterpret_v_u8mf2_i8mf2(v902);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v904 = __riscv_vsub_vx_i8mf2(v903, 32, 8);
        vint16m1_t v905 = v822;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v906 = __riscv_vwmacc_vx_i16m1(v905, v836, v904, 8);
        v822 = v906;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v907 = __riscv_vand_vx_u8mf2(v895, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v908 = __riscv_vsrl_vx_u8mf2(v898, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v909 = __riscv_vand_vx_u8mf2(v908, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v910 = __riscv_vsll_vx_u8mf2(v909, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v911 = __riscv_vor_vv_u8mf2(v907, v910, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v912 = __riscv_vreinterpret_v_u8mf2_i8mf2(v911);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v913 = __riscv_vsub_vx_i8mf2(v912, 32, 8);
        vint16m1_t v914 = v824;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v915 = __riscv_vwmacc_vx_i16m1(v914, v839, v913, 8);
        v824 = v915;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v916 = __riscv_vsrl_vx_u8mf2(v892, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v917 = __riscv_vsrl_vx_u8mf2(v898, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v918 = __riscv_vand_vx_u8mf2(v917, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v919 = __riscv_vsll_vx_u8mf2(v918, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v920 = __riscv_vor_vv_u8mf2(v916, v919, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v921 = __riscv_vreinterpret_v_u8mf2_i8mf2(v920);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v922 = __riscv_vsub_vx_i8mf2(v921, 32, 8);
        vint16m1_t v923 = v826;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v924 = __riscv_vwmacc_vx_i16m1(v923, v842, v922, 8);
        v826 = v924;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v925 = __riscv_vsrl_vx_u8mf2(v895, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v926 = __riscv_vsrl_vx_u8mf2(v898, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v927 = __riscv_vand_vx_u8mf2(v926, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v928 = __riscv_vsll_vx_u8mf2(v927, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v929 = __riscv_vor_vv_u8mf2(v925, v928, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v930 = __riscv_vreinterpret_v_u8mf2_i8mf2(v929);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v931 = __riscv_vsub_vx_i8mf2(v930, 32, 8);
        vint16m1_t v932 = v828;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v933 = __riscv_vwmacc_vx_i16m1(v932, v845, v931, 8);
        v828 = v933;
      }
      vint16m1_t v934 = v814;
      vint16m1_t v935 = v816;
      vint16m1_t v936 = v818;
      vint16m1_t v937 = v820;
      vint16m1_t v938 = v822;
      vint16m1_t v939 = v824;
      vint16m1_t v940 = v826;
      vint16m1_t v941 = v828;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v942 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v943 = __riscv_vwmacc_vv_i32m2(v942, v647, v934, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v944 = __riscv_vwmacc_vv_i32m2(v943, v651, v935, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v945 = __riscv_vwmacc_vv_i32m2(v944, v655, v936, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v946 = __riscv_vwmacc_vv_i32m2(v945, v659, v937, 8);
      v24 = v946;
      vint32m2_t v947 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v948 = __riscv_vwmacc_vv_i32m2(v947, v663, v938, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v949 = __riscv_vwmacc_vv_i32m2(v948, v667, v939, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v950 = __riscv_vwmacc_vv_i32m2(v949, v671, v940, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v951 = __riscv_vwmacc_vv_i32m2(v950, v675, v941, 8);
      v26 = v951;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
      const uint8_t* v952 = v19 + 176;
      const int8_t* v953 = (const int8_t*) v952;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v954 = __riscv_vle8_v_i8mf2(v953, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v955 = __riscv_vsext_vf2_i16m1(v954, 8);
      const uint8_t* v956 = v19 + 208;
      const int8_t* v957 = (const int8_t*) v956;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v958 = __riscv_vle8_v_i8mf2(v957, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v959 = __riscv_vsext_vf2_i16m1(v958, 8);
      const uint8_t* v960 = v19 + 240;
      const int8_t* v961 = (const int8_t*) v960;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v962 = __riscv_vle8_v_i8mf2(v961, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v963 = __riscv_vsext_vf2_i16m1(v962, 8);
      const uint8_t* v964 = v19 + 272;
      const int8_t* v965 = (const int8_t*) v964;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v966 = __riscv_vle8_v_i8mf2(v965, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v967 = __riscv_vsext_vf2_i16m1(v966, 8);
      const uint8_t* v968 = v19 + 184;
      const int8_t* v969 = (const int8_t*) v968;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v970 = __riscv_vle8_v_i8mf2(v969, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v971 = __riscv_vsext_vf2_i16m1(v970, 8);
      const uint8_t* v972 = v19 + 216;
      const int8_t* v973 = (const int8_t*) v972;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v974 = __riscv_vle8_v_i8mf2(v973, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v975 = __riscv_vsext_vf2_i16m1(v974, 8);
      const uint8_t* v976 = v19 + 248;
      const int8_t* v977 = (const int8_t*) v976;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v978 = __riscv_vle8_v_i8mf2(v977, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v979 = __riscv_vsext_vf2_i16m1(v978, 8);
      const uint8_t* v980 = v19 + 280;
      const int8_t* v981 = (const int8_t*) v980;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v982 = __riscv_vle8_v_i8mf2(v981, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
      vint16m1_t v983 = __riscv_vsext_vf2_i16m1(v982, 8);
      vint16m1_t v984;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v985 = __riscv_vmv_v_x_i16m1(0, 8);
      v984 = v985;
      vint16m1_t v986;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v987 = __riscv_vmv_v_x_i16m1(0, 8);
      v986 = v987;
      vint16m1_t v988;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v989 = __riscv_vmv_v_x_i16m1(0, 8);
      v988 = v989;
      vint16m1_t v990;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v991 = __riscv_vmv_v_x_i16m1(0, 8);
      v990 = v991;
      vint16m1_t v992;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v993 = __riscv_vmv_v_x_i16m1(0, 8);
      v992 = v993;
      vint16m1_t v994;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v995 = __riscv_vmv_v_x_i16m1(0, 8);
      v994 = v995;
      vint16m1_t v996;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v997 = __riscv_vmv_v_x_i16m1(0, 8);
      v996 = v997;
      vint16m1_t v998;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v999 = __riscv_vmv_v_x_i16m1(0, 8);
      v998 = v999;
      for (size_t v1000 = 0; v1000 < 8; v1000 += 1) {
        size_t v1001 = v1000 * 16;
        const uint8_t* v1002 = v19 + v1001;
        const uint8_t* v1003 = v21 + v1000;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1004 = v1003 + 148;
        const int8_t* v1005 = (const int8_t*) v1004;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1006 = *(const int8_t *)(v1005);
        const uint8_t* v1007 = v1003 + 180;
        const int8_t* v1008 = (const int8_t*) v1007;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1009 = *(const int8_t *)(v1008);
        const uint8_t* v1010 = v1003 + 212;
        const int8_t* v1011 = (const int8_t*) v1010;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1012 = *(const int8_t *)(v1011);
        const uint8_t* v1013 = v1003 + 244;
        const int8_t* v1014 = (const int8_t*) v1013;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1015 = *(const int8_t *)(v1014);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v1016 = v1002 + 2592;
        const uint8_t* v1017 = (const uint8_t*) v1016;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1018 = __riscv_vle8_v_u8mf2(v1017, 8);
        const uint8_t* v1019 = v1002 + 3104;
        const uint8_t* v1020 = (const uint8_t*) v1019;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1021 = __riscv_vle8_v_u8mf2(v1020, 8);
        const uint8_t* v1022 = v1002 + 1056;
        const uint8_t* v1023 = (const uint8_t*) v1022;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1024 = __riscv_vle8_v_u8mf2(v1023, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1025 = __riscv_vand_vx_u8mf2(v1018, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1026 = __riscv_vand_vx_u8mf2(v1024, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1027 = __riscv_vsll_vx_u8mf2(v1026, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1028 = __riscv_vor_vv_u8mf2(v1025, v1027, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1029 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1028);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1030 = __riscv_vsub_vx_i8mf2(v1029, 32, 8);
        vint16m1_t v1031 = v984;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1032 = __riscv_vwmacc_vx_i16m1(v1031, v1006, v1030, 8);
        v984 = v1032;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1033 = __riscv_vand_vx_u8mf2(v1021, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1034 = __riscv_vsrl_vx_u8mf2(v1024, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1035 = __riscv_vand_vx_u8mf2(v1034, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1036 = __riscv_vsll_vx_u8mf2(v1035, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1037 = __riscv_vor_vv_u8mf2(v1033, v1036, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1038 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1037);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1039 = __riscv_vsub_vx_i8mf2(v1038, 32, 8);
        vint16m1_t v1040 = v986;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1041 = __riscv_vwmacc_vx_i16m1(v1040, v1009, v1039, 8);
        v986 = v1041;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1042 = __riscv_vsrl_vx_u8mf2(v1018, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1043 = __riscv_vsrl_vx_u8mf2(v1024, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1044 = __riscv_vand_vx_u8mf2(v1043, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1045 = __riscv_vsll_vx_u8mf2(v1044, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1046 = __riscv_vor_vv_u8mf2(v1042, v1045, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1047 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1046);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1048 = __riscv_vsub_vx_i8mf2(v1047, 32, 8);
        vint16m1_t v1049 = v988;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1050 = __riscv_vwmacc_vx_i16m1(v1049, v1012, v1048, 8);
        v988 = v1050;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1051 = __riscv_vsrl_vx_u8mf2(v1021, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1052 = __riscv_vsrl_vx_u8mf2(v1024, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1053 = __riscv_vand_vx_u8mf2(v1052, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1054 = __riscv_vsll_vx_u8mf2(v1053, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1055 = __riscv_vor_vv_u8mf2(v1051, v1054, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1056 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1055);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1057 = __riscv_vsub_vx_i8mf2(v1056, 32, 8);
        vint16m1_t v1058 = v990;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1059 = __riscv_vwmacc_vx_i16m1(v1058, v1015, v1057, 8);
        v990 = v1059;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v1060 = v1002 + 2600;
        const uint8_t* v1061 = (const uint8_t*) v1060;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1062 = __riscv_vle8_v_u8mf2(v1061, 8);
        const uint8_t* v1063 = v1002 + 3112;
        const uint8_t* v1064 = (const uint8_t*) v1063;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1065 = __riscv_vle8_v_u8mf2(v1064, 8);
        const uint8_t* v1066 = v1002 + 1064;
        const uint8_t* v1067 = (const uint8_t*) v1066;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1068 = __riscv_vle8_v_u8mf2(v1067, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1069 = __riscv_vand_vx_u8mf2(v1062, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1070 = __riscv_vand_vx_u8mf2(v1068, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1071 = __riscv_vsll_vx_u8mf2(v1070, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1072 = __riscv_vor_vv_u8mf2(v1069, v1071, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1073 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1072);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1074 = __riscv_vsub_vx_i8mf2(v1073, 32, 8);
        vint16m1_t v1075 = v992;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1076 = __riscv_vwmacc_vx_i16m1(v1075, v1006, v1074, 8);
        v992 = v1076;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1077 = __riscv_vand_vx_u8mf2(v1065, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1078 = __riscv_vsrl_vx_u8mf2(v1068, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1079 = __riscv_vand_vx_u8mf2(v1078, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1080 = __riscv_vsll_vx_u8mf2(v1079, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1081 = __riscv_vor_vv_u8mf2(v1077, v1080, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1082 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1081);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1083 = __riscv_vsub_vx_i8mf2(v1082, 32, 8);
        vint16m1_t v1084 = v994;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1085 = __riscv_vwmacc_vx_i16m1(v1084, v1009, v1083, 8);
        v994 = v1085;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1086 = __riscv_vsrl_vx_u8mf2(v1062, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1087 = __riscv_vsrl_vx_u8mf2(v1068, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1088 = __riscv_vand_vx_u8mf2(v1087, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1089 = __riscv_vsll_vx_u8mf2(v1088, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1090 = __riscv_vor_vv_u8mf2(v1086, v1089, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1091 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1090);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1092 = __riscv_vsub_vx_i8mf2(v1091, 32, 8);
        vint16m1_t v1093 = v996;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1094 = __riscv_vwmacc_vx_i16m1(v1093, v1012, v1092, 8);
        v996 = v1094;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1095 = __riscv_vsrl_vx_u8mf2(v1065, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1096 = __riscv_vsrl_vx_u8mf2(v1068, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1097 = __riscv_vand_vx_u8mf2(v1096, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1098 = __riscv_vsll_vx_u8mf2(v1097, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1099 = __riscv_vor_vv_u8mf2(v1095, v1098, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1100 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1099);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1101 = __riscv_vsub_vx_i8mf2(v1100, 32, 8);
        vint16m1_t v1102 = v998;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1103 = __riscv_vwmacc_vx_i16m1(v1102, v1015, v1101, 8);
        v998 = v1103;
      }
      vint16m1_t v1104 = v984;
      vint16m1_t v1105 = v986;
      vint16m1_t v1106 = v988;
      vint16m1_t v1107 = v990;
      vint16m1_t v1108 = v992;
      vint16m1_t v1109 = v994;
      vint16m1_t v1110 = v996;
      vint16m1_t v1111 = v998;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v1112 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1113 = __riscv_vwmacc_vv_i32m2(v1112, v955, v1104, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1114 = __riscv_vwmacc_vv_i32m2(v1113, v959, v1105, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1115 = __riscv_vwmacc_vv_i32m2(v1114, v963, v1106, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1116 = __riscv_vwmacc_vv_i32m2(v1115, v967, v1107, 8);
      v24 = v1116;
      vint32m2_t v1117 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1118 = __riscv_vwmacc_vv_i32m2(v1117, v971, v1108, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1119 = __riscv_vwmacc_vv_i32m2(v1118, v975, v1109, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1120 = __riscv_vwmacc_vv_i32m2(v1119, v979, v1110, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1121 = __riscv_vwmacc_vv_i32m2(v1120, v983, v1111, 8);
      v26 = v1121;
      vint16m1_t v1122;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1123 = __riscv_vmv_v_x_i16m1(0, 8);
      v1122 = v1123;
      vint16m1_t v1124;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1125 = __riscv_vmv_v_x_i16m1(0, 8);
      v1124 = v1125;
      vint16m1_t v1126;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1127 = __riscv_vmv_v_x_i16m1(0, 8);
      v1126 = v1127;
      vint16m1_t v1128;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1129 = __riscv_vmv_v_x_i16m1(0, 8);
      v1128 = v1129;
      vint16m1_t v1130;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1131 = __riscv_vmv_v_x_i16m1(0, 8);
      v1130 = v1131;
      vint16m1_t v1132;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1133 = __riscv_vmv_v_x_i16m1(0, 8);
      v1132 = v1133;
      vint16m1_t v1134;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1135 = __riscv_vmv_v_x_i16m1(0, 8);
      v1134 = v1135;
      vint16m1_t v1136;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1137 = __riscv_vmv_v_x_i16m1(0, 8);
      v1136 = v1137;
      for (size_t v1138 = 0; v1138 < 8; v1138 += 1) {
        size_t v1139 = v1138 * 16;
        const uint8_t* v1140 = v19 + v1139;
        const uint8_t* v1141 = v21 + v1138;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1142 = v1141 + 156;
        const int8_t* v1143 = (const int8_t*) v1142;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1144 = *(const int8_t *)(v1143);
        const uint8_t* v1145 = v1141 + 188;
        const int8_t* v1146 = (const int8_t*) v1145;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1147 = *(const int8_t *)(v1146);
        const uint8_t* v1148 = v1141 + 220;
        const int8_t* v1149 = (const int8_t*) v1148;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1150 = *(const int8_t *)(v1149);
        const uint8_t* v1151 = v1141 + 252;
        const int8_t* v1152 = (const int8_t*) v1151;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1153 = *(const int8_t *)(v1152);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v1154 = v1140 + 2720;
        const uint8_t* v1155 = (const uint8_t*) v1154;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1156 = __riscv_vle8_v_u8mf2(v1155, 8);
        const uint8_t* v1157 = v1140 + 3232;
        const uint8_t* v1158 = (const uint8_t*) v1157;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1159 = __riscv_vle8_v_u8mf2(v1158, 8);
        const uint8_t* v1160 = v1140 + 1184;
        const uint8_t* v1161 = (const uint8_t*) v1160;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1162 = __riscv_vle8_v_u8mf2(v1161, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1163 = __riscv_vand_vx_u8mf2(v1156, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1164 = __riscv_vand_vx_u8mf2(v1162, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1165 = __riscv_vsll_vx_u8mf2(v1164, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1166 = __riscv_vor_vv_u8mf2(v1163, v1165, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1167 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1166);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1168 = __riscv_vsub_vx_i8mf2(v1167, 32, 8);
        vint16m1_t v1169 = v1122;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1170 = __riscv_vwmacc_vx_i16m1(v1169, v1144, v1168, 8);
        v1122 = v1170;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1171 = __riscv_vand_vx_u8mf2(v1159, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1172 = __riscv_vsrl_vx_u8mf2(v1162, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1173 = __riscv_vand_vx_u8mf2(v1172, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1174 = __riscv_vsll_vx_u8mf2(v1173, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1175 = __riscv_vor_vv_u8mf2(v1171, v1174, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1176 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1175);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1177 = __riscv_vsub_vx_i8mf2(v1176, 32, 8);
        vint16m1_t v1178 = v1124;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1179 = __riscv_vwmacc_vx_i16m1(v1178, v1147, v1177, 8);
        v1124 = v1179;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1180 = __riscv_vsrl_vx_u8mf2(v1156, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1181 = __riscv_vsrl_vx_u8mf2(v1162, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1182 = __riscv_vand_vx_u8mf2(v1181, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1183 = __riscv_vsll_vx_u8mf2(v1182, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1184 = __riscv_vor_vv_u8mf2(v1180, v1183, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1185 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1184);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1186 = __riscv_vsub_vx_i8mf2(v1185, 32, 8);
        vint16m1_t v1187 = v1126;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1188 = __riscv_vwmacc_vx_i16m1(v1187, v1150, v1186, 8);
        v1126 = v1188;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1189 = __riscv_vsrl_vx_u8mf2(v1159, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1190 = __riscv_vsrl_vx_u8mf2(v1162, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1191 = __riscv_vand_vx_u8mf2(v1190, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1192 = __riscv_vsll_vx_u8mf2(v1191, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1193 = __riscv_vor_vv_u8mf2(v1189, v1192, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1194 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1193);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1195 = __riscv_vsub_vx_i8mf2(v1194, 32, 8);
        vint16m1_t v1196 = v1128;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1197 = __riscv_vwmacc_vx_i16m1(v1196, v1153, v1195, 8);
        v1128 = v1197;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
        const uint8_t* v1198 = v1140 + 2728;
        const uint8_t* v1199 = (const uint8_t*) v1198;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1200 = __riscv_vle8_v_u8mf2(v1199, 8);
        const uint8_t* v1201 = v1140 + 3240;
        const uint8_t* v1202 = (const uint8_t*) v1201;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1203 = __riscv_vle8_v_u8mf2(v1202, 8);
        const uint8_t* v1204 = v1140 + 1192;
        const uint8_t* v1205 = (const uint8_t*) v1204;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1206 = __riscv_vle8_v_u8mf2(v1205, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1207 = __riscv_vand_vx_u8mf2(v1200, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1208 = __riscv_vand_vx_u8mf2(v1206, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1209 = __riscv_vsll_vx_u8mf2(v1208, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1210 = __riscv_vor_vv_u8mf2(v1207, v1209, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1211 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1210);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1212 = __riscv_vsub_vx_i8mf2(v1211, 32, 8);
        vint16m1_t v1213 = v1130;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1214 = __riscv_vwmacc_vx_i16m1(v1213, v1144, v1212, 8);
        v1130 = v1214;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1215 = __riscv_vand_vx_u8mf2(v1203, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1216 = __riscv_vsrl_vx_u8mf2(v1206, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1217 = __riscv_vand_vx_u8mf2(v1216, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1218 = __riscv_vsll_vx_u8mf2(v1217, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1219 = __riscv_vor_vv_u8mf2(v1215, v1218, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1220 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1219);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1221 = __riscv_vsub_vx_i8mf2(v1220, 32, 8);
        vint16m1_t v1222 = v1132;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1223 = __riscv_vwmacc_vx_i16m1(v1222, v1147, v1221, 8);
        v1132 = v1223;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1224 = __riscv_vsrl_vx_u8mf2(v1200, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1225 = __riscv_vsrl_vx_u8mf2(v1206, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1226 = __riscv_vand_vx_u8mf2(v1225, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1227 = __riscv_vsll_vx_u8mf2(v1226, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1228 = __riscv_vor_vv_u8mf2(v1224, v1227, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1229 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1228);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1230 = __riscv_vsub_vx_i8mf2(v1229, 32, 8);
        vint16m1_t v1231 = v1134;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1232 = __riscv_vwmacc_vx_i16m1(v1231, v1150, v1230, 8);
        v1134 = v1232;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1233 = __riscv_vsrl_vx_u8mf2(v1203, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1234 = __riscv_vsrl_vx_u8mf2(v1206, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1235 = __riscv_vand_vx_u8mf2(v1234, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1236 = __riscv_vsll_vx_u8mf2(v1235, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1237 = __riscv_vor_vv_u8mf2(v1233, v1236, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1238 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1237);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
        vint8mf2_t v1239 = __riscv_vsub_vx_i8mf2(v1238, 32, 8);
        vint16m1_t v1240 = v1136;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1241 = __riscv_vwmacc_vx_i16m1(v1240, v1153, v1239, 8);
        v1136 = v1241;
      }
      vint16m1_t v1242 = v1122;
      vint16m1_t v1243 = v1124;
      vint16m1_t v1244 = v1126;
      vint16m1_t v1245 = v1128;
      vint16m1_t v1246 = v1130;
      vint16m1_t v1247 = v1132;
      vint16m1_t v1248 = v1134;
      vint16m1_t v1249 = v1136;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v1250 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1251 = __riscv_vwmacc_vv_i32m2(v1250, v955, v1242, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1252 = __riscv_vwmacc_vv_i32m2(v1251, v959, v1243, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1253 = __riscv_vwmacc_vv_i32m2(v1252, v963, v1244, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1254 = __riscv_vwmacc_vv_i32m2(v1253, v967, v1245, 8);
      v24 = v1254;
      vint32m2_t v1255 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1256 = __riscv_vwmacc_vv_i32m2(v1255, v971, v1246, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1257 = __riscv_vwmacc_vv_i32m2(v1256, v975, v1247, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1258 = __riscv_vwmacc_vv_i32m2(v1257, v979, v1248, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1259 = __riscv_vwmacc_vv_i32m2(v1258, v983, v1249, 8);
      v26 = v1259;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v1260 = (const _Float16*) v19;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v1261 = __riscv_vle16_v_f16m1(v1260, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v1262 = __riscv_vfwcvt_f_f_v_f32m2(v1261, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v1263 = __riscv_vfmul_vf_f32m2(v1262, v23, 8);
      vint32m2_t v1264 = v24;
      vfloat32m2_t v1265 = v13;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v1266 = __riscv_vfcvt_f_x_v_f32m2(v1264, 8);
      vfloat32m2_t v1267 = __riscv_vfmacc_vv_f32m2(v1265, v1266, v1263, 8);
      v13 = v1267;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v1268 = v19 + 16;
      const _Float16* v1269 = (const _Float16*) v1268;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v1270 = __riscv_vle16_v_f16m1(v1269, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v1271 = __riscv_vfwcvt_f_f_v_f32m2(v1270, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v1272 = __riscv_vfmul_vf_f32m2(v1271, v23, 8);
      vint32m2_t v1273 = v26;
      vfloat32m2_t v1274 = v15;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v1275 = __riscv_vfcvt_f_x_v_f32m2(v1273, 8);
      vfloat32m2_t v1276 = __riscv_vfmacc_vv_f32m2(v1274, v1275, v1272, 8);
      v15 = v1276;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v1277 = v9 * 16;
    float* v1278 = v2 + v1277;
    vfloat32m2_t v1279 = v13;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v1278, v1279, 8);
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v1280 = v9 * 16;
    size_t v1281 = v1280 + 8;
    float* v1282 = v2 + v1281;
    vfloat32m2_t v1283 = v15;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v1282, v1283, 8);
  }
  return;
}


