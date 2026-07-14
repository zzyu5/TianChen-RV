#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5, size_t v6, size_t v7) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v8 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v9 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=row_group_count
  size_t v10 = v5 / 4;
  // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=col_group_count
  size_t v11 = v6 / 16;
  for (size_t v12 = 0; v12 < v10; v12 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_group_base
    size_t v13 = v12 * v9;
    size_t v14 = v13 * 1168;
    const uint8_t* v15 = v4 + v14;
    for (size_t v16 = 0; v16 < v11; v16 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_group_base
      size_t v17 = v16 * v9;
      size_t v18 = v17 * 3360;
      const uint8_t* v19 = v3 + v18;
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
        size_t v29 = v28 * 3360;
        const uint8_t* v30 = v19 + v29;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
        size_t v31 = v28 * 1168;
        const uint8_t* v32 = v15 + v31;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const float* v33 = (const float*) v32;
        float v34 = *(const float *)(v33);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v35 = v32 + 4;
        const float* v36 = (const float*) v35;
        float v37 = *(const float *)(v36);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v38 = v32 + 8;
        const float* v39 = (const float*) v38;
        float v40 = *(const float *)(v39);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v41 = v32 + 12;
        const float* v42 = (const float*) v41;
        float v43 = *(const float *)(v42);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const _Float16* v44 = (const _Float16*) v30;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v45 = __riscv_vle16_v_f16m1(v44, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v46 = __riscv_vfwcvt_f_f_v_f32m2(v45, 16);
        vint32m2_t v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v48 = __riscv_vmv_v_x_i32m2(0, 16);
        v47 = v48;
        vint32m2_t v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v50 = __riscv_vmv_v_x_i32m2(0, 16);
        v49 = v50;
        vint32m2_t v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v52 = __riscv_vmv_v_x_i32m2(0, 16);
        v51 = v52;
        vint32m2_t v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v54 = __riscv_vmv_v_x_i32m2(0, 16);
        v53 = v54;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v55 = v30 + 32;
        const int8_t* v56 = (const int8_t*) v55;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v57 = __riscv_vle8_v_i8mf2(v56, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v58 = __riscv_vsext_vf2_i16m1(v57, 16);
        const uint8_t* v59 = v30 + 64;
        const int8_t* v60 = (const int8_t*) v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v61 = __riscv_vle8_v_i8mf2(v60, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v62 = __riscv_vsext_vf2_i16m1(v61, 16);
        const uint8_t* v63 = v30 + 96;
        const int8_t* v64 = (const int8_t*) v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v65 = __riscv_vle8_v_i8mf2(v64, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v66 = __riscv_vsext_vf2_i16m1(v65, 16);
        const uint8_t* v67 = v30 + 128;
        const int8_t* v68 = (const int8_t*) v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v69 = __riscv_vle8_v_i8mf2(v68, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v70 = __riscv_vsext_vf2_i16m1(v69, 16);
        vint16m1_t v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v72 = __riscv_vmv_v_x_i16m1(0, 16);
        v71 = v72;
        vint16m1_t v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v74 = __riscv_vmv_v_x_i16m1(0, 16);
        v73 = v74;
        vint16m1_t v75;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v76 = __riscv_vmv_v_x_i16m1(0, 16);
        v75 = v76;
        vint16m1_t v77;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v78 = __riscv_vmv_v_x_i16m1(0, 16);
        v77 = v78;
        vint16m1_t v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v80 = __riscv_vmv_v_x_i16m1(0, 16);
        v79 = v80;
        vint16m1_t v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v82 = __riscv_vmv_v_x_i16m1(0, 16);
        v81 = v82;
        vint16m1_t v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v84 = __riscv_vmv_v_x_i16m1(0, 16);
        v83 = v84;
        vint16m1_t v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v86 = __riscv_vmv_v_x_i16m1(0, 16);
        v85 = v86;
        vint16m1_t v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v88 = __riscv_vmv_v_x_i16m1(0, 16);
        v87 = v88;
        vint16m1_t v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v90 = __riscv_vmv_v_x_i16m1(0, 16);
        v89 = v90;
        vint16m1_t v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v92 = __riscv_vmv_v_x_i16m1(0, 16);
        v91 = v92;
        vint16m1_t v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v94 = __riscv_vmv_v_x_i16m1(0, 16);
        v93 = v94;
        vint16m1_t v95;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v96 = __riscv_vmv_v_x_i16m1(0, 16);
        v95 = v96;
        vint16m1_t v97;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v98 = __riscv_vmv_v_x_i16m1(0, 16);
        v97 = v98;
        vint16m1_t v99;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v100 = __riscv_vmv_v_x_i16m1(0, 16);
        v99 = v100;
        vint16m1_t v101;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v102 = __riscv_vmv_v_x_i16m1(0, 16);
        v101 = v102;
        for (size_t v103 = 0; v103 < 8; v103 += 1) {
          size_t v104 = v103 * 16;
          const uint8_t* v105 = v30 + v104;
          size_t v106 = v103 * 4;
          const uint8_t* v107 = v32 + v106;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v108 = v105 + 1312;
          const uint8_t* v109 = (const uint8_t*) v108;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v110 = __riscv_vle8_v_u8mf2(v109, 16);
          const uint8_t* v111 = v105 + 1824;
          const uint8_t* v112 = (const uint8_t*) v111;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v113 = __riscv_vle8_v_u8mf2(v112, 16);
          const uint8_t* v114 = v105 + 288;
          const uint8_t* v115 = (const uint8_t*) v114;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v116 = __riscv_vle8_v_u8mf2(v115, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v117 = __riscv_vand_vx_u8mf2(v110, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v118 = __riscv_vand_vx_u8mf2(v116, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v119 = __riscv_vsll_vx_u8mf2(v118, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v120 = __riscv_vor_vv_u8mf2(v117, v119, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v121 = __riscv_vreinterpret_v_u8mf2_i8mf2(v120);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v122 = __riscv_vsub_vx_i8mf2(v121, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v123 = __riscv_vand_vx_u8mf2(v113, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v124 = __riscv_vsrl_vx_u8mf2(v116, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v125 = __riscv_vand_vx_u8mf2(v124, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v126 = __riscv_vsll_vx_u8mf2(v125, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v127 = __riscv_vor_vv_u8mf2(v123, v126, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v128 = __riscv_vreinterpret_v_u8mf2_i8mf2(v127);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v129 = __riscv_vsub_vx_i8mf2(v128, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v130 = __riscv_vsrl_vx_u8mf2(v110, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v131 = __riscv_vsrl_vx_u8mf2(v116, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v132 = __riscv_vand_vx_u8mf2(v131, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v133 = __riscv_vsll_vx_u8mf2(v132, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v134 = __riscv_vor_vv_u8mf2(v130, v133, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v135 = __riscv_vreinterpret_v_u8mf2_i8mf2(v134);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v136 = __riscv_vsub_vx_i8mf2(v135, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v137 = __riscv_vsrl_vx_u8mf2(v113, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v138 = __riscv_vsrl_vx_u8mf2(v116, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v139 = __riscv_vand_vx_u8mf2(v138, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v140 = __riscv_vsll_vx_u8mf2(v139, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v141 = __riscv_vor_vv_u8mf2(v137, v140, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v142 = __riscv_vreinterpret_v_u8mf2_i8mf2(v141);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v143 = __riscv_vsub_vx_i8mf2(v142, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v144 = v107 + 16;
          const int8_t* v145 = (const int8_t*) v144;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v146 = *(const int8_t *)(v145);
          vint16m1_t v147 = v71;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v148 = __riscv_vwmacc_vx_i16m1(v147, v146, v122, 16);
          v71 = v148;
          const uint8_t* v149 = v107 + 144;
          const int8_t* v150 = (const int8_t*) v149;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v151 = *(const int8_t *)(v150);
          vint16m1_t v152 = v73;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v153 = __riscv_vwmacc_vx_i16m1(v152, v151, v129, 16);
          v73 = v153;
          const uint8_t* v154 = v107 + 272;
          const int8_t* v155 = (const int8_t*) v154;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v156 = *(const int8_t *)(v155);
          vint16m1_t v157 = v75;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v158 = __riscv_vwmacc_vx_i16m1(v157, v156, v136, 16);
          v75 = v158;
          const uint8_t* v159 = v107 + 400;
          const int8_t* v160 = (const int8_t*) v159;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v161 = *(const int8_t *)(v160);
          vint16m1_t v162 = v77;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v163 = __riscv_vwmacc_vx_i16m1(v162, v161, v143, 16);
          v77 = v163;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v164 = v107 + 17;
          const int8_t* v165 = (const int8_t*) v164;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v166 = *(const int8_t *)(v165);
          vint16m1_t v167 = v79;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v168 = __riscv_vwmacc_vx_i16m1(v167, v166, v122, 16);
          v79 = v168;
          const uint8_t* v169 = v107 + 145;
          const int8_t* v170 = (const int8_t*) v169;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v171 = *(const int8_t *)(v170);
          vint16m1_t v172 = v81;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v173 = __riscv_vwmacc_vx_i16m1(v172, v171, v129, 16);
          v81 = v173;
          const uint8_t* v174 = v107 + 273;
          const int8_t* v175 = (const int8_t*) v174;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v176 = *(const int8_t *)(v175);
          vint16m1_t v177 = v83;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v178 = __riscv_vwmacc_vx_i16m1(v177, v176, v136, 16);
          v83 = v178;
          const uint8_t* v179 = v107 + 401;
          const int8_t* v180 = (const int8_t*) v179;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v181 = *(const int8_t *)(v180);
          vint16m1_t v182 = v85;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v183 = __riscv_vwmacc_vx_i16m1(v182, v181, v143, 16);
          v85 = v183;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v184 = v107 + 18;
          const int8_t* v185 = (const int8_t*) v184;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v186 = *(const int8_t *)(v185);
          vint16m1_t v187 = v87;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v188 = __riscv_vwmacc_vx_i16m1(v187, v186, v122, 16);
          v87 = v188;
          const uint8_t* v189 = v107 + 146;
          const int8_t* v190 = (const int8_t*) v189;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v191 = *(const int8_t *)(v190);
          vint16m1_t v192 = v89;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v193 = __riscv_vwmacc_vx_i16m1(v192, v191, v129, 16);
          v89 = v193;
          const uint8_t* v194 = v107 + 274;
          const int8_t* v195 = (const int8_t*) v194;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v196 = *(const int8_t *)(v195);
          vint16m1_t v197 = v91;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v198 = __riscv_vwmacc_vx_i16m1(v197, v196, v136, 16);
          v91 = v198;
          const uint8_t* v199 = v107 + 402;
          const int8_t* v200 = (const int8_t*) v199;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v201 = *(const int8_t *)(v200);
          vint16m1_t v202 = v93;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v203 = __riscv_vwmacc_vx_i16m1(v202, v201, v143, 16);
          v93 = v203;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v204 = v107 + 19;
          const int8_t* v205 = (const int8_t*) v204;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v206 = *(const int8_t *)(v205);
          vint16m1_t v207 = v95;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v208 = __riscv_vwmacc_vx_i16m1(v207, v206, v122, 16);
          v95 = v208;
          const uint8_t* v209 = v107 + 147;
          const int8_t* v210 = (const int8_t*) v209;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v211 = *(const int8_t *)(v210);
          vint16m1_t v212 = v97;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v213 = __riscv_vwmacc_vx_i16m1(v212, v211, v129, 16);
          v97 = v213;
          const uint8_t* v214 = v107 + 275;
          const int8_t* v215 = (const int8_t*) v214;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v216 = *(const int8_t *)(v215);
          vint16m1_t v217 = v99;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v218 = __riscv_vwmacc_vx_i16m1(v217, v216, v136, 16);
          v99 = v218;
          const uint8_t* v219 = v107 + 403;
          const int8_t* v220 = (const int8_t*) v219;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v221 = *(const int8_t *)(v220);
          vint16m1_t v222 = v101;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v223 = __riscv_vwmacc_vx_i16m1(v222, v221, v143, 16);
          v101 = v223;
        }
        vint16m1_t v224 = v71;
        vint16m1_t v225 = v73;
        vint16m1_t v226 = v75;
        vint16m1_t v227 = v77;
        vint16m1_t v228 = v79;
        vint16m1_t v229 = v81;
        vint16m1_t v230 = v83;
        vint16m1_t v231 = v85;
        vint16m1_t v232 = v87;
        vint16m1_t v233 = v89;
        vint16m1_t v234 = v91;
        vint16m1_t v235 = v93;
        vint16m1_t v236 = v95;
        vint16m1_t v237 = v97;
        vint16m1_t v238 = v99;
        vint16m1_t v239 = v101;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v240 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v241 = __riscv_vwmacc_vv_i32m2(v240, v58, v224, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v242 = __riscv_vwmacc_vv_i32m2(v241, v62, v225, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v243 = __riscv_vwmacc_vv_i32m2(v242, v66, v226, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v244 = __riscv_vwmacc_vv_i32m2(v243, v70, v227, 16);
        v47 = v244;
        vint32m2_t v245 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v246 = __riscv_vwmacc_vv_i32m2(v245, v58, v228, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v247 = __riscv_vwmacc_vv_i32m2(v246, v62, v229, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v248 = __riscv_vwmacc_vv_i32m2(v247, v66, v230, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v249 = __riscv_vwmacc_vv_i32m2(v248, v70, v231, 16);
        v49 = v249;
        vint32m2_t v250 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v251 = __riscv_vwmacc_vv_i32m2(v250, v58, v232, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v252 = __riscv_vwmacc_vv_i32m2(v251, v62, v233, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v253 = __riscv_vwmacc_vv_i32m2(v252, v66, v234, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v254 = __riscv_vwmacc_vv_i32m2(v253, v70, v235, 16);
        v51 = v254;
        vint32m2_t v255 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v256 = __riscv_vwmacc_vv_i32m2(v255, v58, v236, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v257 = __riscv_vwmacc_vv_i32m2(v256, v62, v237, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v258 = __riscv_vwmacc_vv_i32m2(v257, v66, v238, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v259 = __riscv_vwmacc_vv_i32m2(v258, v70, v239, 16);
        v53 = v259;
        vint16m1_t v260;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v261 = __riscv_vmv_v_x_i16m1(0, 16);
        v260 = v261;
        vint16m1_t v262;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v263 = __riscv_vmv_v_x_i16m1(0, 16);
        v262 = v263;
        vint16m1_t v264;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v265 = __riscv_vmv_v_x_i16m1(0, 16);
        v264 = v265;
        vint16m1_t v266;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v267 = __riscv_vmv_v_x_i16m1(0, 16);
        v266 = v267;
        vint16m1_t v268;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v269 = __riscv_vmv_v_x_i16m1(0, 16);
        v268 = v269;
        vint16m1_t v270;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v271 = __riscv_vmv_v_x_i16m1(0, 16);
        v270 = v271;
        vint16m1_t v272;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v273 = __riscv_vmv_v_x_i16m1(0, 16);
        v272 = v273;
        vint16m1_t v274;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v275 = __riscv_vmv_v_x_i16m1(0, 16);
        v274 = v275;
        vint16m1_t v276;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v277 = __riscv_vmv_v_x_i16m1(0, 16);
        v276 = v277;
        vint16m1_t v278;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v279 = __riscv_vmv_v_x_i16m1(0, 16);
        v278 = v279;
        vint16m1_t v280;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v281 = __riscv_vmv_v_x_i16m1(0, 16);
        v280 = v281;
        vint16m1_t v282;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v283 = __riscv_vmv_v_x_i16m1(0, 16);
        v282 = v283;
        vint16m1_t v284;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v285 = __riscv_vmv_v_x_i16m1(0, 16);
        v284 = v285;
        vint16m1_t v286;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v287 = __riscv_vmv_v_x_i16m1(0, 16);
        v286 = v287;
        vint16m1_t v288;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v289 = __riscv_vmv_v_x_i16m1(0, 16);
        v288 = v289;
        vint16m1_t v290;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v291 = __riscv_vmv_v_x_i16m1(0, 16);
        v290 = v291;
        for (size_t v292 = 0; v292 < 8; v292 += 1) {
          size_t v293 = v292 * 16;
          const uint8_t* v294 = v30 + v293;
          size_t v295 = v292 * 4;
          const uint8_t* v296 = v32 + v295;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v297 = v294 + 1440;
          const uint8_t* v298 = (const uint8_t*) v297;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v299 = __riscv_vle8_v_u8mf2(v298, 16);
          const uint8_t* v300 = v294 + 1952;
          const uint8_t* v301 = (const uint8_t*) v300;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v302 = __riscv_vle8_v_u8mf2(v301, 16);
          const uint8_t* v303 = v294 + 416;
          const uint8_t* v304 = (const uint8_t*) v303;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v305 = __riscv_vle8_v_u8mf2(v304, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v306 = __riscv_vand_vx_u8mf2(v299, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v307 = __riscv_vand_vx_u8mf2(v305, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v308 = __riscv_vsll_vx_u8mf2(v307, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v309 = __riscv_vor_vv_u8mf2(v306, v308, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v310 = __riscv_vreinterpret_v_u8mf2_i8mf2(v309);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v311 = __riscv_vsub_vx_i8mf2(v310, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v312 = __riscv_vand_vx_u8mf2(v302, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v313 = __riscv_vsrl_vx_u8mf2(v305, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v314 = __riscv_vand_vx_u8mf2(v313, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v315 = __riscv_vsll_vx_u8mf2(v314, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v316 = __riscv_vor_vv_u8mf2(v312, v315, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v317 = __riscv_vreinterpret_v_u8mf2_i8mf2(v316);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v318 = __riscv_vsub_vx_i8mf2(v317, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v319 = __riscv_vsrl_vx_u8mf2(v299, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v320 = __riscv_vsrl_vx_u8mf2(v305, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v321 = __riscv_vand_vx_u8mf2(v320, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v322 = __riscv_vsll_vx_u8mf2(v321, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v323 = __riscv_vor_vv_u8mf2(v319, v322, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v324 = __riscv_vreinterpret_v_u8mf2_i8mf2(v323);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v325 = __riscv_vsub_vx_i8mf2(v324, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v326 = __riscv_vsrl_vx_u8mf2(v302, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v327 = __riscv_vsrl_vx_u8mf2(v305, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v328 = __riscv_vand_vx_u8mf2(v327, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v329 = __riscv_vsll_vx_u8mf2(v328, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v330 = __riscv_vor_vv_u8mf2(v326, v329, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v331 = __riscv_vreinterpret_v_u8mf2_i8mf2(v330);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v332 = __riscv_vsub_vx_i8mf2(v331, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v333 = v296 + 48;
          const int8_t* v334 = (const int8_t*) v333;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v335 = *(const int8_t *)(v334);
          vint16m1_t v336 = v260;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v337 = __riscv_vwmacc_vx_i16m1(v336, v335, v311, 16);
          v260 = v337;
          const uint8_t* v338 = v296 + 176;
          const int8_t* v339 = (const int8_t*) v338;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v340 = *(const int8_t *)(v339);
          vint16m1_t v341 = v262;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v342 = __riscv_vwmacc_vx_i16m1(v341, v340, v318, 16);
          v262 = v342;
          const uint8_t* v343 = v296 + 304;
          const int8_t* v344 = (const int8_t*) v343;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v345 = *(const int8_t *)(v344);
          vint16m1_t v346 = v264;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v347 = __riscv_vwmacc_vx_i16m1(v346, v345, v325, 16);
          v264 = v347;
          const uint8_t* v348 = v296 + 432;
          const int8_t* v349 = (const int8_t*) v348;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v350 = *(const int8_t *)(v349);
          vint16m1_t v351 = v266;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v352 = __riscv_vwmacc_vx_i16m1(v351, v350, v332, 16);
          v266 = v352;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v353 = v296 + 49;
          const int8_t* v354 = (const int8_t*) v353;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v355 = *(const int8_t *)(v354);
          vint16m1_t v356 = v268;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v357 = __riscv_vwmacc_vx_i16m1(v356, v355, v311, 16);
          v268 = v357;
          const uint8_t* v358 = v296 + 177;
          const int8_t* v359 = (const int8_t*) v358;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v360 = *(const int8_t *)(v359);
          vint16m1_t v361 = v270;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v362 = __riscv_vwmacc_vx_i16m1(v361, v360, v318, 16);
          v270 = v362;
          const uint8_t* v363 = v296 + 305;
          const int8_t* v364 = (const int8_t*) v363;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v365 = *(const int8_t *)(v364);
          vint16m1_t v366 = v272;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v367 = __riscv_vwmacc_vx_i16m1(v366, v365, v325, 16);
          v272 = v367;
          const uint8_t* v368 = v296 + 433;
          const int8_t* v369 = (const int8_t*) v368;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v370 = *(const int8_t *)(v369);
          vint16m1_t v371 = v274;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v372 = __riscv_vwmacc_vx_i16m1(v371, v370, v332, 16);
          v274 = v372;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v373 = v296 + 50;
          const int8_t* v374 = (const int8_t*) v373;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v375 = *(const int8_t *)(v374);
          vint16m1_t v376 = v276;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v377 = __riscv_vwmacc_vx_i16m1(v376, v375, v311, 16);
          v276 = v377;
          const uint8_t* v378 = v296 + 178;
          const int8_t* v379 = (const int8_t*) v378;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v380 = *(const int8_t *)(v379);
          vint16m1_t v381 = v278;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v382 = __riscv_vwmacc_vx_i16m1(v381, v380, v318, 16);
          v278 = v382;
          const uint8_t* v383 = v296 + 306;
          const int8_t* v384 = (const int8_t*) v383;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v385 = *(const int8_t *)(v384);
          vint16m1_t v386 = v280;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v387 = __riscv_vwmacc_vx_i16m1(v386, v385, v325, 16);
          v280 = v387;
          const uint8_t* v388 = v296 + 434;
          const int8_t* v389 = (const int8_t*) v388;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v390 = *(const int8_t *)(v389);
          vint16m1_t v391 = v282;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v392 = __riscv_vwmacc_vx_i16m1(v391, v390, v332, 16);
          v282 = v392;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v393 = v296 + 51;
          const int8_t* v394 = (const int8_t*) v393;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v395 = *(const int8_t *)(v394);
          vint16m1_t v396 = v284;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v397 = __riscv_vwmacc_vx_i16m1(v396, v395, v311, 16);
          v284 = v397;
          const uint8_t* v398 = v296 + 179;
          const int8_t* v399 = (const int8_t*) v398;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v400 = *(const int8_t *)(v399);
          vint16m1_t v401 = v286;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v402 = __riscv_vwmacc_vx_i16m1(v401, v400, v318, 16);
          v286 = v402;
          const uint8_t* v403 = v296 + 307;
          const int8_t* v404 = (const int8_t*) v403;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v405 = *(const int8_t *)(v404);
          vint16m1_t v406 = v288;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v407 = __riscv_vwmacc_vx_i16m1(v406, v405, v325, 16);
          v288 = v407;
          const uint8_t* v408 = v296 + 435;
          const int8_t* v409 = (const int8_t*) v408;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v410 = *(const int8_t *)(v409);
          vint16m1_t v411 = v290;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v412 = __riscv_vwmacc_vx_i16m1(v411, v410, v332, 16);
          v290 = v412;
        }
        vint16m1_t v413 = v260;
        vint16m1_t v414 = v262;
        vint16m1_t v415 = v264;
        vint16m1_t v416 = v266;
        vint16m1_t v417 = v268;
        vint16m1_t v418 = v270;
        vint16m1_t v419 = v272;
        vint16m1_t v420 = v274;
        vint16m1_t v421 = v276;
        vint16m1_t v422 = v278;
        vint16m1_t v423 = v280;
        vint16m1_t v424 = v282;
        vint16m1_t v425 = v284;
        vint16m1_t v426 = v286;
        vint16m1_t v427 = v288;
        vint16m1_t v428 = v290;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v429 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v430 = __riscv_vwmacc_vv_i32m2(v429, v58, v413, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v431 = __riscv_vwmacc_vv_i32m2(v430, v62, v414, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v432 = __riscv_vwmacc_vv_i32m2(v431, v66, v415, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v433 = __riscv_vwmacc_vv_i32m2(v432, v70, v416, 16);
        v47 = v433;
        vint32m2_t v434 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v435 = __riscv_vwmacc_vv_i32m2(v434, v58, v417, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v436 = __riscv_vwmacc_vv_i32m2(v435, v62, v418, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v437 = __riscv_vwmacc_vv_i32m2(v436, v66, v419, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v438 = __riscv_vwmacc_vv_i32m2(v437, v70, v420, 16);
        v49 = v438;
        vint32m2_t v439 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v440 = __riscv_vwmacc_vv_i32m2(v439, v58, v421, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v441 = __riscv_vwmacc_vv_i32m2(v440, v62, v422, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v442 = __riscv_vwmacc_vv_i32m2(v441, v66, v423, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v443 = __riscv_vwmacc_vv_i32m2(v442, v70, v424, 16);
        v51 = v443;
        vint32m2_t v444 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v445 = __riscv_vwmacc_vv_i32m2(v444, v58, v425, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v446 = __riscv_vwmacc_vv_i32m2(v445, v62, v426, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v447 = __riscv_vwmacc_vv_i32m2(v446, v66, v427, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v448 = __riscv_vwmacc_vv_i32m2(v447, v70, v428, 16);
        v53 = v448;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v449 = v30 + 48;
        const int8_t* v450 = (const int8_t*) v449;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v451 = __riscv_vle8_v_i8mf2(v450, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v452 = __riscv_vsext_vf2_i16m1(v451, 16);
        const uint8_t* v453 = v30 + 80;
        const int8_t* v454 = (const int8_t*) v453;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v455 = __riscv_vle8_v_i8mf2(v454, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v456 = __riscv_vsext_vf2_i16m1(v455, 16);
        const uint8_t* v457 = v30 + 112;
        const int8_t* v458 = (const int8_t*) v457;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v459 = __riscv_vle8_v_i8mf2(v458, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v460 = __riscv_vsext_vf2_i16m1(v459, 16);
        const uint8_t* v461 = v30 + 144;
        const int8_t* v462 = (const int8_t*) v461;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v463 = __riscv_vle8_v_i8mf2(v462, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v464 = __riscv_vsext_vf2_i16m1(v463, 16);
        vint16m1_t v465;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v466 = __riscv_vmv_v_x_i16m1(0, 16);
        v465 = v466;
        vint16m1_t v467;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v468 = __riscv_vmv_v_x_i16m1(0, 16);
        v467 = v468;
        vint16m1_t v469;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v470 = __riscv_vmv_v_x_i16m1(0, 16);
        v469 = v470;
        vint16m1_t v471;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v472 = __riscv_vmv_v_x_i16m1(0, 16);
        v471 = v472;
        vint16m1_t v473;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v474 = __riscv_vmv_v_x_i16m1(0, 16);
        v473 = v474;
        vint16m1_t v475;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v476 = __riscv_vmv_v_x_i16m1(0, 16);
        v475 = v476;
        vint16m1_t v477;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v478 = __riscv_vmv_v_x_i16m1(0, 16);
        v477 = v478;
        vint16m1_t v479;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v480 = __riscv_vmv_v_x_i16m1(0, 16);
        v479 = v480;
        vint16m1_t v481;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v482 = __riscv_vmv_v_x_i16m1(0, 16);
        v481 = v482;
        vint16m1_t v483;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v484 = __riscv_vmv_v_x_i16m1(0, 16);
        v483 = v484;
        vint16m1_t v485;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v486 = __riscv_vmv_v_x_i16m1(0, 16);
        v485 = v486;
        vint16m1_t v487;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v488 = __riscv_vmv_v_x_i16m1(0, 16);
        v487 = v488;
        vint16m1_t v489;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v490 = __riscv_vmv_v_x_i16m1(0, 16);
        v489 = v490;
        vint16m1_t v491;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v492 = __riscv_vmv_v_x_i16m1(0, 16);
        v491 = v492;
        vint16m1_t v493;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v494 = __riscv_vmv_v_x_i16m1(0, 16);
        v493 = v494;
        vint16m1_t v495;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v496 = __riscv_vmv_v_x_i16m1(0, 16);
        v495 = v496;
        for (size_t v497 = 0; v497 < 8; v497 += 1) {
          size_t v498 = v497 * 16;
          const uint8_t* v499 = v30 + v498;
          size_t v500 = v497 * 4;
          const uint8_t* v501 = v32 + v500;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v502 = v499 + 1568;
          const uint8_t* v503 = (const uint8_t*) v502;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v504 = __riscv_vle8_v_u8mf2(v503, 16);
          const uint8_t* v505 = v499 + 2080;
          const uint8_t* v506 = (const uint8_t*) v505;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v507 = __riscv_vle8_v_u8mf2(v506, 16);
          const uint8_t* v508 = v499 + 544;
          const uint8_t* v509 = (const uint8_t*) v508;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v510 = __riscv_vle8_v_u8mf2(v509, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v511 = __riscv_vand_vx_u8mf2(v504, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v512 = __riscv_vand_vx_u8mf2(v510, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v513 = __riscv_vsll_vx_u8mf2(v512, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v514 = __riscv_vor_vv_u8mf2(v511, v513, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v515 = __riscv_vreinterpret_v_u8mf2_i8mf2(v514);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v516 = __riscv_vsub_vx_i8mf2(v515, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v517 = __riscv_vand_vx_u8mf2(v507, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v518 = __riscv_vsrl_vx_u8mf2(v510, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v519 = __riscv_vand_vx_u8mf2(v518, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v520 = __riscv_vsll_vx_u8mf2(v519, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v521 = __riscv_vor_vv_u8mf2(v517, v520, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v522 = __riscv_vreinterpret_v_u8mf2_i8mf2(v521);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v523 = __riscv_vsub_vx_i8mf2(v522, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v524 = __riscv_vsrl_vx_u8mf2(v504, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v525 = __riscv_vsrl_vx_u8mf2(v510, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v526 = __riscv_vand_vx_u8mf2(v525, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v527 = __riscv_vsll_vx_u8mf2(v526, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v528 = __riscv_vor_vv_u8mf2(v524, v527, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v529 = __riscv_vreinterpret_v_u8mf2_i8mf2(v528);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v530 = __riscv_vsub_vx_i8mf2(v529, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v531 = __riscv_vsrl_vx_u8mf2(v507, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v532 = __riscv_vsrl_vx_u8mf2(v510, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v533 = __riscv_vand_vx_u8mf2(v532, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v534 = __riscv_vsll_vx_u8mf2(v533, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v535 = __riscv_vor_vv_u8mf2(v531, v534, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v536 = __riscv_vreinterpret_v_u8mf2_i8mf2(v535);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v537 = __riscv_vsub_vx_i8mf2(v536, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v538 = v501 + 80;
          const int8_t* v539 = (const int8_t*) v538;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v540 = *(const int8_t *)(v539);
          vint16m1_t v541 = v465;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v542 = __riscv_vwmacc_vx_i16m1(v541, v540, v516, 16);
          v465 = v542;
          const uint8_t* v543 = v501 + 208;
          const int8_t* v544 = (const int8_t*) v543;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v545 = *(const int8_t *)(v544);
          vint16m1_t v546 = v467;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v547 = __riscv_vwmacc_vx_i16m1(v546, v545, v523, 16);
          v467 = v547;
          const uint8_t* v548 = v501 + 336;
          const int8_t* v549 = (const int8_t*) v548;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v550 = *(const int8_t *)(v549);
          vint16m1_t v551 = v469;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v552 = __riscv_vwmacc_vx_i16m1(v551, v550, v530, 16);
          v469 = v552;
          const uint8_t* v553 = v501 + 464;
          const int8_t* v554 = (const int8_t*) v553;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v555 = *(const int8_t *)(v554);
          vint16m1_t v556 = v471;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v557 = __riscv_vwmacc_vx_i16m1(v556, v555, v537, 16);
          v471 = v557;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v558 = v501 + 81;
          const int8_t* v559 = (const int8_t*) v558;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v560 = *(const int8_t *)(v559);
          vint16m1_t v561 = v473;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v562 = __riscv_vwmacc_vx_i16m1(v561, v560, v516, 16);
          v473 = v562;
          const uint8_t* v563 = v501 + 209;
          const int8_t* v564 = (const int8_t*) v563;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v565 = *(const int8_t *)(v564);
          vint16m1_t v566 = v475;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v567 = __riscv_vwmacc_vx_i16m1(v566, v565, v523, 16);
          v475 = v567;
          const uint8_t* v568 = v501 + 337;
          const int8_t* v569 = (const int8_t*) v568;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v570 = *(const int8_t *)(v569);
          vint16m1_t v571 = v477;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v572 = __riscv_vwmacc_vx_i16m1(v571, v570, v530, 16);
          v477 = v572;
          const uint8_t* v573 = v501 + 465;
          const int8_t* v574 = (const int8_t*) v573;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v575 = *(const int8_t *)(v574);
          vint16m1_t v576 = v479;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v577 = __riscv_vwmacc_vx_i16m1(v576, v575, v537, 16);
          v479 = v577;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v578 = v501 + 82;
          const int8_t* v579 = (const int8_t*) v578;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v580 = *(const int8_t *)(v579);
          vint16m1_t v581 = v481;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v582 = __riscv_vwmacc_vx_i16m1(v581, v580, v516, 16);
          v481 = v582;
          const uint8_t* v583 = v501 + 210;
          const int8_t* v584 = (const int8_t*) v583;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v585 = *(const int8_t *)(v584);
          vint16m1_t v586 = v483;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v587 = __riscv_vwmacc_vx_i16m1(v586, v585, v523, 16);
          v483 = v587;
          const uint8_t* v588 = v501 + 338;
          const int8_t* v589 = (const int8_t*) v588;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v590 = *(const int8_t *)(v589);
          vint16m1_t v591 = v485;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v592 = __riscv_vwmacc_vx_i16m1(v591, v590, v530, 16);
          v485 = v592;
          const uint8_t* v593 = v501 + 466;
          const int8_t* v594 = (const int8_t*) v593;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v595 = *(const int8_t *)(v594);
          vint16m1_t v596 = v487;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v597 = __riscv_vwmacc_vx_i16m1(v596, v595, v537, 16);
          v487 = v597;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v598 = v501 + 83;
          const int8_t* v599 = (const int8_t*) v598;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v600 = *(const int8_t *)(v599);
          vint16m1_t v601 = v489;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v602 = __riscv_vwmacc_vx_i16m1(v601, v600, v516, 16);
          v489 = v602;
          const uint8_t* v603 = v501 + 211;
          const int8_t* v604 = (const int8_t*) v603;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v605 = *(const int8_t *)(v604);
          vint16m1_t v606 = v491;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v607 = __riscv_vwmacc_vx_i16m1(v606, v605, v523, 16);
          v491 = v607;
          const uint8_t* v608 = v501 + 339;
          const int8_t* v609 = (const int8_t*) v608;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v610 = *(const int8_t *)(v609);
          vint16m1_t v611 = v493;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v612 = __riscv_vwmacc_vx_i16m1(v611, v610, v530, 16);
          v493 = v612;
          const uint8_t* v613 = v501 + 467;
          const int8_t* v614 = (const int8_t*) v613;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v615 = *(const int8_t *)(v614);
          vint16m1_t v616 = v495;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v617 = __riscv_vwmacc_vx_i16m1(v616, v615, v537, 16);
          v495 = v617;
        }
        vint16m1_t v618 = v465;
        vint16m1_t v619 = v467;
        vint16m1_t v620 = v469;
        vint16m1_t v621 = v471;
        vint16m1_t v622 = v473;
        vint16m1_t v623 = v475;
        vint16m1_t v624 = v477;
        vint16m1_t v625 = v479;
        vint16m1_t v626 = v481;
        vint16m1_t v627 = v483;
        vint16m1_t v628 = v485;
        vint16m1_t v629 = v487;
        vint16m1_t v630 = v489;
        vint16m1_t v631 = v491;
        vint16m1_t v632 = v493;
        vint16m1_t v633 = v495;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v634 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v635 = __riscv_vwmacc_vv_i32m2(v634, v452, v618, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v636 = __riscv_vwmacc_vv_i32m2(v635, v456, v619, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v637 = __riscv_vwmacc_vv_i32m2(v636, v460, v620, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v638 = __riscv_vwmacc_vv_i32m2(v637, v464, v621, 16);
        v47 = v638;
        vint32m2_t v639 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v640 = __riscv_vwmacc_vv_i32m2(v639, v452, v622, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v641 = __riscv_vwmacc_vv_i32m2(v640, v456, v623, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v642 = __riscv_vwmacc_vv_i32m2(v641, v460, v624, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v643 = __riscv_vwmacc_vv_i32m2(v642, v464, v625, 16);
        v49 = v643;
        vint32m2_t v644 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v645 = __riscv_vwmacc_vv_i32m2(v644, v452, v626, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v646 = __riscv_vwmacc_vv_i32m2(v645, v456, v627, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v647 = __riscv_vwmacc_vv_i32m2(v646, v460, v628, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v648 = __riscv_vwmacc_vv_i32m2(v647, v464, v629, 16);
        v51 = v648;
        vint32m2_t v649 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v650 = __riscv_vwmacc_vv_i32m2(v649, v452, v630, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v651 = __riscv_vwmacc_vv_i32m2(v650, v456, v631, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v652 = __riscv_vwmacc_vv_i32m2(v651, v460, v632, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v653 = __riscv_vwmacc_vv_i32m2(v652, v464, v633, 16);
        v53 = v653;
        vint16m1_t v654;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v655 = __riscv_vmv_v_x_i16m1(0, 16);
        v654 = v655;
        vint16m1_t v656;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v657 = __riscv_vmv_v_x_i16m1(0, 16);
        v656 = v657;
        vint16m1_t v658;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v659 = __riscv_vmv_v_x_i16m1(0, 16);
        v658 = v659;
        vint16m1_t v660;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v661 = __riscv_vmv_v_x_i16m1(0, 16);
        v660 = v661;
        vint16m1_t v662;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v663 = __riscv_vmv_v_x_i16m1(0, 16);
        v662 = v663;
        vint16m1_t v664;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v665 = __riscv_vmv_v_x_i16m1(0, 16);
        v664 = v665;
        vint16m1_t v666;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v667 = __riscv_vmv_v_x_i16m1(0, 16);
        v666 = v667;
        vint16m1_t v668;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v669 = __riscv_vmv_v_x_i16m1(0, 16);
        v668 = v669;
        vint16m1_t v670;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v671 = __riscv_vmv_v_x_i16m1(0, 16);
        v670 = v671;
        vint16m1_t v672;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v673 = __riscv_vmv_v_x_i16m1(0, 16);
        v672 = v673;
        vint16m1_t v674;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v675 = __riscv_vmv_v_x_i16m1(0, 16);
        v674 = v675;
        vint16m1_t v676;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v677 = __riscv_vmv_v_x_i16m1(0, 16);
        v676 = v677;
        vint16m1_t v678;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v679 = __riscv_vmv_v_x_i16m1(0, 16);
        v678 = v679;
        vint16m1_t v680;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v681 = __riscv_vmv_v_x_i16m1(0, 16);
        v680 = v681;
        vint16m1_t v682;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v683 = __riscv_vmv_v_x_i16m1(0, 16);
        v682 = v683;
        vint16m1_t v684;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v685 = __riscv_vmv_v_x_i16m1(0, 16);
        v684 = v685;
        for (size_t v686 = 0; v686 < 8; v686 += 1) {
          size_t v687 = v686 * 16;
          const uint8_t* v688 = v30 + v687;
          size_t v689 = v686 * 4;
          const uint8_t* v690 = v32 + v689;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v691 = v688 + 1696;
          const uint8_t* v692 = (const uint8_t*) v691;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v693 = __riscv_vle8_v_u8mf2(v692, 16);
          const uint8_t* v694 = v688 + 2208;
          const uint8_t* v695 = (const uint8_t*) v694;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v696 = __riscv_vle8_v_u8mf2(v695, 16);
          const uint8_t* v697 = v688 + 672;
          const uint8_t* v698 = (const uint8_t*) v697;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v699 = __riscv_vle8_v_u8mf2(v698, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v700 = __riscv_vand_vx_u8mf2(v693, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v701 = __riscv_vand_vx_u8mf2(v699, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v702 = __riscv_vsll_vx_u8mf2(v701, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v703 = __riscv_vor_vv_u8mf2(v700, v702, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v704 = __riscv_vreinterpret_v_u8mf2_i8mf2(v703);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v705 = __riscv_vsub_vx_i8mf2(v704, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v706 = __riscv_vand_vx_u8mf2(v696, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v707 = __riscv_vsrl_vx_u8mf2(v699, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v708 = __riscv_vand_vx_u8mf2(v707, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v709 = __riscv_vsll_vx_u8mf2(v708, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v710 = __riscv_vor_vv_u8mf2(v706, v709, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v711 = __riscv_vreinterpret_v_u8mf2_i8mf2(v710);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v712 = __riscv_vsub_vx_i8mf2(v711, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v713 = __riscv_vsrl_vx_u8mf2(v693, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v714 = __riscv_vsrl_vx_u8mf2(v699, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v715 = __riscv_vand_vx_u8mf2(v714, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v716 = __riscv_vsll_vx_u8mf2(v715, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v717 = __riscv_vor_vv_u8mf2(v713, v716, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v718 = __riscv_vreinterpret_v_u8mf2_i8mf2(v717);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v719 = __riscv_vsub_vx_i8mf2(v718, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v720 = __riscv_vsrl_vx_u8mf2(v696, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v721 = __riscv_vsrl_vx_u8mf2(v699, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v722 = __riscv_vand_vx_u8mf2(v721, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v723 = __riscv_vsll_vx_u8mf2(v722, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v724 = __riscv_vor_vv_u8mf2(v720, v723, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v725 = __riscv_vreinterpret_v_u8mf2_i8mf2(v724);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v726 = __riscv_vsub_vx_i8mf2(v725, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v727 = v690 + 112;
          const int8_t* v728 = (const int8_t*) v727;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v729 = *(const int8_t *)(v728);
          vint16m1_t v730 = v654;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v731 = __riscv_vwmacc_vx_i16m1(v730, v729, v705, 16);
          v654 = v731;
          const uint8_t* v732 = v690 + 240;
          const int8_t* v733 = (const int8_t*) v732;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v734 = *(const int8_t *)(v733);
          vint16m1_t v735 = v656;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v736 = __riscv_vwmacc_vx_i16m1(v735, v734, v712, 16);
          v656 = v736;
          const uint8_t* v737 = v690 + 368;
          const int8_t* v738 = (const int8_t*) v737;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v739 = *(const int8_t *)(v738);
          vint16m1_t v740 = v658;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v741 = __riscv_vwmacc_vx_i16m1(v740, v739, v719, 16);
          v658 = v741;
          const uint8_t* v742 = v690 + 496;
          const int8_t* v743 = (const int8_t*) v742;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v744 = *(const int8_t *)(v743);
          vint16m1_t v745 = v660;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v746 = __riscv_vwmacc_vx_i16m1(v745, v744, v726, 16);
          v660 = v746;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v747 = v690 + 113;
          const int8_t* v748 = (const int8_t*) v747;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v749 = *(const int8_t *)(v748);
          vint16m1_t v750 = v662;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v751 = __riscv_vwmacc_vx_i16m1(v750, v749, v705, 16);
          v662 = v751;
          const uint8_t* v752 = v690 + 241;
          const int8_t* v753 = (const int8_t*) v752;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v754 = *(const int8_t *)(v753);
          vint16m1_t v755 = v664;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v756 = __riscv_vwmacc_vx_i16m1(v755, v754, v712, 16);
          v664 = v756;
          const uint8_t* v757 = v690 + 369;
          const int8_t* v758 = (const int8_t*) v757;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v759 = *(const int8_t *)(v758);
          vint16m1_t v760 = v666;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v761 = __riscv_vwmacc_vx_i16m1(v760, v759, v719, 16);
          v666 = v761;
          const uint8_t* v762 = v690 + 497;
          const int8_t* v763 = (const int8_t*) v762;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v764 = *(const int8_t *)(v763);
          vint16m1_t v765 = v668;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v766 = __riscv_vwmacc_vx_i16m1(v765, v764, v726, 16);
          v668 = v766;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v767 = v690 + 114;
          const int8_t* v768 = (const int8_t*) v767;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v769 = *(const int8_t *)(v768);
          vint16m1_t v770 = v670;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v771 = __riscv_vwmacc_vx_i16m1(v770, v769, v705, 16);
          v670 = v771;
          const uint8_t* v772 = v690 + 242;
          const int8_t* v773 = (const int8_t*) v772;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v774 = *(const int8_t *)(v773);
          vint16m1_t v775 = v672;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v776 = __riscv_vwmacc_vx_i16m1(v775, v774, v712, 16);
          v672 = v776;
          const uint8_t* v777 = v690 + 370;
          const int8_t* v778 = (const int8_t*) v777;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v779 = *(const int8_t *)(v778);
          vint16m1_t v780 = v674;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v781 = __riscv_vwmacc_vx_i16m1(v780, v779, v719, 16);
          v674 = v781;
          const uint8_t* v782 = v690 + 498;
          const int8_t* v783 = (const int8_t*) v782;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v784 = *(const int8_t *)(v783);
          vint16m1_t v785 = v676;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v786 = __riscv_vwmacc_vx_i16m1(v785, v784, v726, 16);
          v676 = v786;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v787 = v690 + 115;
          const int8_t* v788 = (const int8_t*) v787;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v789 = *(const int8_t *)(v788);
          vint16m1_t v790 = v678;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v791 = __riscv_vwmacc_vx_i16m1(v790, v789, v705, 16);
          v678 = v791;
          const uint8_t* v792 = v690 + 243;
          const int8_t* v793 = (const int8_t*) v792;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v794 = *(const int8_t *)(v793);
          vint16m1_t v795 = v680;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v796 = __riscv_vwmacc_vx_i16m1(v795, v794, v712, 16);
          v680 = v796;
          const uint8_t* v797 = v690 + 371;
          const int8_t* v798 = (const int8_t*) v797;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v799 = *(const int8_t *)(v798);
          vint16m1_t v800 = v682;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v801 = __riscv_vwmacc_vx_i16m1(v800, v799, v719, 16);
          v682 = v801;
          const uint8_t* v802 = v690 + 499;
          const int8_t* v803 = (const int8_t*) v802;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v804 = *(const int8_t *)(v803);
          vint16m1_t v805 = v684;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v806 = __riscv_vwmacc_vx_i16m1(v805, v804, v726, 16);
          v684 = v806;
        }
        vint16m1_t v807 = v654;
        vint16m1_t v808 = v656;
        vint16m1_t v809 = v658;
        vint16m1_t v810 = v660;
        vint16m1_t v811 = v662;
        vint16m1_t v812 = v664;
        vint16m1_t v813 = v666;
        vint16m1_t v814 = v668;
        vint16m1_t v815 = v670;
        vint16m1_t v816 = v672;
        vint16m1_t v817 = v674;
        vint16m1_t v818 = v676;
        vint16m1_t v819 = v678;
        vint16m1_t v820 = v680;
        vint16m1_t v821 = v682;
        vint16m1_t v822 = v684;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v823 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v824 = __riscv_vwmacc_vv_i32m2(v823, v452, v807, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v825 = __riscv_vwmacc_vv_i32m2(v824, v456, v808, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v826 = __riscv_vwmacc_vv_i32m2(v825, v460, v809, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v827 = __riscv_vwmacc_vv_i32m2(v826, v464, v810, 16);
        v47 = v827;
        vint32m2_t v828 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v829 = __riscv_vwmacc_vv_i32m2(v828, v452, v811, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v830 = __riscv_vwmacc_vv_i32m2(v829, v456, v812, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v831 = __riscv_vwmacc_vv_i32m2(v830, v460, v813, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v832 = __riscv_vwmacc_vv_i32m2(v831, v464, v814, 16);
        v49 = v832;
        vint32m2_t v833 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v834 = __riscv_vwmacc_vv_i32m2(v833, v452, v815, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v835 = __riscv_vwmacc_vv_i32m2(v834, v456, v816, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v836 = __riscv_vwmacc_vv_i32m2(v835, v460, v817, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v837 = __riscv_vwmacc_vv_i32m2(v836, v464, v818, 16);
        v51 = v837;
        vint32m2_t v838 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v839 = __riscv_vwmacc_vv_i32m2(v838, v452, v819, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v840 = __riscv_vwmacc_vv_i32m2(v839, v456, v820, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v841 = __riscv_vwmacc_vv_i32m2(v840, v460, v821, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v842 = __riscv_vwmacc_vv_i32m2(v841, v464, v822, 16);
        v53 = v842;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v843 = v30 + 160;
        const int8_t* v844 = (const int8_t*) v843;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v845 = __riscv_vle8_v_i8mf2(v844, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v846 = __riscv_vsext_vf2_i16m1(v845, 16);
        const uint8_t* v847 = v30 + 192;
        const int8_t* v848 = (const int8_t*) v847;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v849 = __riscv_vle8_v_i8mf2(v848, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v850 = __riscv_vsext_vf2_i16m1(v849, 16);
        const uint8_t* v851 = v30 + 224;
        const int8_t* v852 = (const int8_t*) v851;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v853 = __riscv_vle8_v_i8mf2(v852, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v854 = __riscv_vsext_vf2_i16m1(v853, 16);
        const uint8_t* v855 = v30 + 256;
        const int8_t* v856 = (const int8_t*) v855;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v857 = __riscv_vle8_v_i8mf2(v856, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v858 = __riscv_vsext_vf2_i16m1(v857, 16);
        vint16m1_t v859;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v860 = __riscv_vmv_v_x_i16m1(0, 16);
        v859 = v860;
        vint16m1_t v861;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v862 = __riscv_vmv_v_x_i16m1(0, 16);
        v861 = v862;
        vint16m1_t v863;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v864 = __riscv_vmv_v_x_i16m1(0, 16);
        v863 = v864;
        vint16m1_t v865;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v866 = __riscv_vmv_v_x_i16m1(0, 16);
        v865 = v866;
        vint16m1_t v867;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v868 = __riscv_vmv_v_x_i16m1(0, 16);
        v867 = v868;
        vint16m1_t v869;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v870 = __riscv_vmv_v_x_i16m1(0, 16);
        v869 = v870;
        vint16m1_t v871;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v872 = __riscv_vmv_v_x_i16m1(0, 16);
        v871 = v872;
        vint16m1_t v873;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v874 = __riscv_vmv_v_x_i16m1(0, 16);
        v873 = v874;
        vint16m1_t v875;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v876 = __riscv_vmv_v_x_i16m1(0, 16);
        v875 = v876;
        vint16m1_t v877;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v878 = __riscv_vmv_v_x_i16m1(0, 16);
        v877 = v878;
        vint16m1_t v879;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v880 = __riscv_vmv_v_x_i16m1(0, 16);
        v879 = v880;
        vint16m1_t v881;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v882 = __riscv_vmv_v_x_i16m1(0, 16);
        v881 = v882;
        vint16m1_t v883;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v884 = __riscv_vmv_v_x_i16m1(0, 16);
        v883 = v884;
        vint16m1_t v885;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v886 = __riscv_vmv_v_x_i16m1(0, 16);
        v885 = v886;
        vint16m1_t v887;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v888 = __riscv_vmv_v_x_i16m1(0, 16);
        v887 = v888;
        vint16m1_t v889;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v890 = __riscv_vmv_v_x_i16m1(0, 16);
        v889 = v890;
        for (size_t v891 = 0; v891 < 8; v891 += 1) {
          size_t v892 = v891 * 16;
          const uint8_t* v893 = v30 + v892;
          size_t v894 = v891 * 4;
          const uint8_t* v895 = v32 + v894;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v896 = v893 + 2336;
          const uint8_t* v897 = (const uint8_t*) v896;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v898 = __riscv_vle8_v_u8mf2(v897, 16);
          const uint8_t* v899 = v893 + 2848;
          const uint8_t* v900 = (const uint8_t*) v899;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v901 = __riscv_vle8_v_u8mf2(v900, 16);
          const uint8_t* v902 = v893 + 800;
          const uint8_t* v903 = (const uint8_t*) v902;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v904 = __riscv_vle8_v_u8mf2(v903, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v905 = __riscv_vand_vx_u8mf2(v898, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v906 = __riscv_vand_vx_u8mf2(v904, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v907 = __riscv_vsll_vx_u8mf2(v906, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v908 = __riscv_vor_vv_u8mf2(v905, v907, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v909 = __riscv_vreinterpret_v_u8mf2_i8mf2(v908);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v910 = __riscv_vsub_vx_i8mf2(v909, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v911 = __riscv_vand_vx_u8mf2(v901, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v912 = __riscv_vsrl_vx_u8mf2(v904, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v913 = __riscv_vand_vx_u8mf2(v912, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v914 = __riscv_vsll_vx_u8mf2(v913, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v915 = __riscv_vor_vv_u8mf2(v911, v914, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v916 = __riscv_vreinterpret_v_u8mf2_i8mf2(v915);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v917 = __riscv_vsub_vx_i8mf2(v916, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v918 = __riscv_vsrl_vx_u8mf2(v898, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v919 = __riscv_vsrl_vx_u8mf2(v904, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v920 = __riscv_vand_vx_u8mf2(v919, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v921 = __riscv_vsll_vx_u8mf2(v920, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v922 = __riscv_vor_vv_u8mf2(v918, v921, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v923 = __riscv_vreinterpret_v_u8mf2_i8mf2(v922);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v924 = __riscv_vsub_vx_i8mf2(v923, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v925 = __riscv_vsrl_vx_u8mf2(v901, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v926 = __riscv_vsrl_vx_u8mf2(v904, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v927 = __riscv_vand_vx_u8mf2(v926, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v928 = __riscv_vsll_vx_u8mf2(v927, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v929 = __riscv_vor_vv_u8mf2(v925, v928, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v930 = __riscv_vreinterpret_v_u8mf2_i8mf2(v929);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v931 = __riscv_vsub_vx_i8mf2(v930, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v932 = v895 + 528;
          const int8_t* v933 = (const int8_t*) v932;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v934 = *(const int8_t *)(v933);
          vint16m1_t v935 = v859;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v936 = __riscv_vwmacc_vx_i16m1(v935, v934, v910, 16);
          v859 = v936;
          const uint8_t* v937 = v895 + 656;
          const int8_t* v938 = (const int8_t*) v937;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v939 = *(const int8_t *)(v938);
          vint16m1_t v940 = v861;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v941 = __riscv_vwmacc_vx_i16m1(v940, v939, v917, 16);
          v861 = v941;
          const uint8_t* v942 = v895 + 784;
          const int8_t* v943 = (const int8_t*) v942;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v944 = *(const int8_t *)(v943);
          vint16m1_t v945 = v863;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v946 = __riscv_vwmacc_vx_i16m1(v945, v944, v924, 16);
          v863 = v946;
          const uint8_t* v947 = v895 + 912;
          const int8_t* v948 = (const int8_t*) v947;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v949 = *(const int8_t *)(v948);
          vint16m1_t v950 = v865;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v951 = __riscv_vwmacc_vx_i16m1(v950, v949, v931, 16);
          v865 = v951;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v952 = v895 + 529;
          const int8_t* v953 = (const int8_t*) v952;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v954 = *(const int8_t *)(v953);
          vint16m1_t v955 = v867;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v956 = __riscv_vwmacc_vx_i16m1(v955, v954, v910, 16);
          v867 = v956;
          const uint8_t* v957 = v895 + 657;
          const int8_t* v958 = (const int8_t*) v957;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v959 = *(const int8_t *)(v958);
          vint16m1_t v960 = v869;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v961 = __riscv_vwmacc_vx_i16m1(v960, v959, v917, 16);
          v869 = v961;
          const uint8_t* v962 = v895 + 785;
          const int8_t* v963 = (const int8_t*) v962;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v964 = *(const int8_t *)(v963);
          vint16m1_t v965 = v871;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v966 = __riscv_vwmacc_vx_i16m1(v965, v964, v924, 16);
          v871 = v966;
          const uint8_t* v967 = v895 + 913;
          const int8_t* v968 = (const int8_t*) v967;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v969 = *(const int8_t *)(v968);
          vint16m1_t v970 = v873;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v971 = __riscv_vwmacc_vx_i16m1(v970, v969, v931, 16);
          v873 = v971;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v972 = v895 + 530;
          const int8_t* v973 = (const int8_t*) v972;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v974 = *(const int8_t *)(v973);
          vint16m1_t v975 = v875;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v976 = __riscv_vwmacc_vx_i16m1(v975, v974, v910, 16);
          v875 = v976;
          const uint8_t* v977 = v895 + 658;
          const int8_t* v978 = (const int8_t*) v977;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v979 = *(const int8_t *)(v978);
          vint16m1_t v980 = v877;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v981 = __riscv_vwmacc_vx_i16m1(v980, v979, v917, 16);
          v877 = v981;
          const uint8_t* v982 = v895 + 786;
          const int8_t* v983 = (const int8_t*) v982;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v984 = *(const int8_t *)(v983);
          vint16m1_t v985 = v879;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v986 = __riscv_vwmacc_vx_i16m1(v985, v984, v924, 16);
          v879 = v986;
          const uint8_t* v987 = v895 + 914;
          const int8_t* v988 = (const int8_t*) v987;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v989 = *(const int8_t *)(v988);
          vint16m1_t v990 = v881;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v991 = __riscv_vwmacc_vx_i16m1(v990, v989, v931, 16);
          v881 = v991;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v992 = v895 + 531;
          const int8_t* v993 = (const int8_t*) v992;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v994 = *(const int8_t *)(v993);
          vint16m1_t v995 = v883;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v996 = __riscv_vwmacc_vx_i16m1(v995, v994, v910, 16);
          v883 = v996;
          const uint8_t* v997 = v895 + 659;
          const int8_t* v998 = (const int8_t*) v997;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v999 = *(const int8_t *)(v998);
          vint16m1_t v1000 = v885;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1001 = __riscv_vwmacc_vx_i16m1(v1000, v999, v917, 16);
          v885 = v1001;
          const uint8_t* v1002 = v895 + 787;
          const int8_t* v1003 = (const int8_t*) v1002;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1004 = *(const int8_t *)(v1003);
          vint16m1_t v1005 = v887;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1006 = __riscv_vwmacc_vx_i16m1(v1005, v1004, v924, 16);
          v887 = v1006;
          const uint8_t* v1007 = v895 + 915;
          const int8_t* v1008 = (const int8_t*) v1007;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1009 = *(const int8_t *)(v1008);
          vint16m1_t v1010 = v889;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1011 = __riscv_vwmacc_vx_i16m1(v1010, v1009, v931, 16);
          v889 = v1011;
        }
        vint16m1_t v1012 = v859;
        vint16m1_t v1013 = v861;
        vint16m1_t v1014 = v863;
        vint16m1_t v1015 = v865;
        vint16m1_t v1016 = v867;
        vint16m1_t v1017 = v869;
        vint16m1_t v1018 = v871;
        vint16m1_t v1019 = v873;
        vint16m1_t v1020 = v875;
        vint16m1_t v1021 = v877;
        vint16m1_t v1022 = v879;
        vint16m1_t v1023 = v881;
        vint16m1_t v1024 = v883;
        vint16m1_t v1025 = v885;
        vint16m1_t v1026 = v887;
        vint16m1_t v1027 = v889;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v1028 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1029 = __riscv_vwmacc_vv_i32m2(v1028, v846, v1012, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1030 = __riscv_vwmacc_vv_i32m2(v1029, v850, v1013, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1031 = __riscv_vwmacc_vv_i32m2(v1030, v854, v1014, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1032 = __riscv_vwmacc_vv_i32m2(v1031, v858, v1015, 16);
        v47 = v1032;
        vint32m2_t v1033 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1034 = __riscv_vwmacc_vv_i32m2(v1033, v846, v1016, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1035 = __riscv_vwmacc_vv_i32m2(v1034, v850, v1017, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1036 = __riscv_vwmacc_vv_i32m2(v1035, v854, v1018, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1037 = __riscv_vwmacc_vv_i32m2(v1036, v858, v1019, 16);
        v49 = v1037;
        vint32m2_t v1038 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1039 = __riscv_vwmacc_vv_i32m2(v1038, v846, v1020, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1040 = __riscv_vwmacc_vv_i32m2(v1039, v850, v1021, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1041 = __riscv_vwmacc_vv_i32m2(v1040, v854, v1022, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1042 = __riscv_vwmacc_vv_i32m2(v1041, v858, v1023, 16);
        v51 = v1042;
        vint32m2_t v1043 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1044 = __riscv_vwmacc_vv_i32m2(v1043, v846, v1024, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1045 = __riscv_vwmacc_vv_i32m2(v1044, v850, v1025, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1046 = __riscv_vwmacc_vv_i32m2(v1045, v854, v1026, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1047 = __riscv_vwmacc_vv_i32m2(v1046, v858, v1027, 16);
        v53 = v1047;
        vint16m1_t v1048;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1049 = __riscv_vmv_v_x_i16m1(0, 16);
        v1048 = v1049;
        vint16m1_t v1050;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1051 = __riscv_vmv_v_x_i16m1(0, 16);
        v1050 = v1051;
        vint16m1_t v1052;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1053 = __riscv_vmv_v_x_i16m1(0, 16);
        v1052 = v1053;
        vint16m1_t v1054;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1055 = __riscv_vmv_v_x_i16m1(0, 16);
        v1054 = v1055;
        vint16m1_t v1056;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1057 = __riscv_vmv_v_x_i16m1(0, 16);
        v1056 = v1057;
        vint16m1_t v1058;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1059 = __riscv_vmv_v_x_i16m1(0, 16);
        v1058 = v1059;
        vint16m1_t v1060;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1061 = __riscv_vmv_v_x_i16m1(0, 16);
        v1060 = v1061;
        vint16m1_t v1062;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1063 = __riscv_vmv_v_x_i16m1(0, 16);
        v1062 = v1063;
        vint16m1_t v1064;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1065 = __riscv_vmv_v_x_i16m1(0, 16);
        v1064 = v1065;
        vint16m1_t v1066;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1067 = __riscv_vmv_v_x_i16m1(0, 16);
        v1066 = v1067;
        vint16m1_t v1068;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1069 = __riscv_vmv_v_x_i16m1(0, 16);
        v1068 = v1069;
        vint16m1_t v1070;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1071 = __riscv_vmv_v_x_i16m1(0, 16);
        v1070 = v1071;
        vint16m1_t v1072;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1073 = __riscv_vmv_v_x_i16m1(0, 16);
        v1072 = v1073;
        vint16m1_t v1074;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1075 = __riscv_vmv_v_x_i16m1(0, 16);
        v1074 = v1075;
        vint16m1_t v1076;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1077 = __riscv_vmv_v_x_i16m1(0, 16);
        v1076 = v1077;
        vint16m1_t v1078;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1079 = __riscv_vmv_v_x_i16m1(0, 16);
        v1078 = v1079;
        for (size_t v1080 = 0; v1080 < 8; v1080 += 1) {
          size_t v1081 = v1080 * 16;
          const uint8_t* v1082 = v30 + v1081;
          size_t v1083 = v1080 * 4;
          const uint8_t* v1084 = v32 + v1083;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v1085 = v1082 + 2464;
          const uint8_t* v1086 = (const uint8_t*) v1085;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1087 = __riscv_vle8_v_u8mf2(v1086, 16);
          const uint8_t* v1088 = v1082 + 2976;
          const uint8_t* v1089 = (const uint8_t*) v1088;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1090 = __riscv_vle8_v_u8mf2(v1089, 16);
          const uint8_t* v1091 = v1082 + 928;
          const uint8_t* v1092 = (const uint8_t*) v1091;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1093 = __riscv_vle8_v_u8mf2(v1092, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1094 = __riscv_vand_vx_u8mf2(v1087, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1095 = __riscv_vand_vx_u8mf2(v1093, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1096 = __riscv_vsll_vx_u8mf2(v1095, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1097 = __riscv_vor_vv_u8mf2(v1094, v1096, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1098 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1097);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1099 = __riscv_vsub_vx_i8mf2(v1098, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1100 = __riscv_vand_vx_u8mf2(v1090, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1101 = __riscv_vsrl_vx_u8mf2(v1093, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1102 = __riscv_vand_vx_u8mf2(v1101, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1103 = __riscv_vsll_vx_u8mf2(v1102, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1104 = __riscv_vor_vv_u8mf2(v1100, v1103, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1105 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1104);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1106 = __riscv_vsub_vx_i8mf2(v1105, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1107 = __riscv_vsrl_vx_u8mf2(v1087, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1108 = __riscv_vsrl_vx_u8mf2(v1093, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1109 = __riscv_vand_vx_u8mf2(v1108, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1110 = __riscv_vsll_vx_u8mf2(v1109, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1111 = __riscv_vor_vv_u8mf2(v1107, v1110, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1112 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1111);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1113 = __riscv_vsub_vx_i8mf2(v1112, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1114 = __riscv_vsrl_vx_u8mf2(v1090, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1115 = __riscv_vsrl_vx_u8mf2(v1093, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1116 = __riscv_vand_vx_u8mf2(v1115, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1117 = __riscv_vsll_vx_u8mf2(v1116, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1118 = __riscv_vor_vv_u8mf2(v1114, v1117, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1119 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1118);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1120 = __riscv_vsub_vx_i8mf2(v1119, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1121 = v1084 + 560;
          const int8_t* v1122 = (const int8_t*) v1121;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1123 = *(const int8_t *)(v1122);
          vint16m1_t v1124 = v1048;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1125 = __riscv_vwmacc_vx_i16m1(v1124, v1123, v1099, 16);
          v1048 = v1125;
          const uint8_t* v1126 = v1084 + 688;
          const int8_t* v1127 = (const int8_t*) v1126;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1128 = *(const int8_t *)(v1127);
          vint16m1_t v1129 = v1050;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1130 = __riscv_vwmacc_vx_i16m1(v1129, v1128, v1106, 16);
          v1050 = v1130;
          const uint8_t* v1131 = v1084 + 816;
          const int8_t* v1132 = (const int8_t*) v1131;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1133 = *(const int8_t *)(v1132);
          vint16m1_t v1134 = v1052;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1135 = __riscv_vwmacc_vx_i16m1(v1134, v1133, v1113, 16);
          v1052 = v1135;
          const uint8_t* v1136 = v1084 + 944;
          const int8_t* v1137 = (const int8_t*) v1136;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1138 = *(const int8_t *)(v1137);
          vint16m1_t v1139 = v1054;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1140 = __riscv_vwmacc_vx_i16m1(v1139, v1138, v1120, 16);
          v1054 = v1140;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1141 = v1084 + 561;
          const int8_t* v1142 = (const int8_t*) v1141;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1143 = *(const int8_t *)(v1142);
          vint16m1_t v1144 = v1056;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1145 = __riscv_vwmacc_vx_i16m1(v1144, v1143, v1099, 16);
          v1056 = v1145;
          const uint8_t* v1146 = v1084 + 689;
          const int8_t* v1147 = (const int8_t*) v1146;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1148 = *(const int8_t *)(v1147);
          vint16m1_t v1149 = v1058;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1150 = __riscv_vwmacc_vx_i16m1(v1149, v1148, v1106, 16);
          v1058 = v1150;
          const uint8_t* v1151 = v1084 + 817;
          const int8_t* v1152 = (const int8_t*) v1151;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1153 = *(const int8_t *)(v1152);
          vint16m1_t v1154 = v1060;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1155 = __riscv_vwmacc_vx_i16m1(v1154, v1153, v1113, 16);
          v1060 = v1155;
          const uint8_t* v1156 = v1084 + 945;
          const int8_t* v1157 = (const int8_t*) v1156;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1158 = *(const int8_t *)(v1157);
          vint16m1_t v1159 = v1062;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1160 = __riscv_vwmacc_vx_i16m1(v1159, v1158, v1120, 16);
          v1062 = v1160;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1161 = v1084 + 562;
          const int8_t* v1162 = (const int8_t*) v1161;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1163 = *(const int8_t *)(v1162);
          vint16m1_t v1164 = v1064;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1165 = __riscv_vwmacc_vx_i16m1(v1164, v1163, v1099, 16);
          v1064 = v1165;
          const uint8_t* v1166 = v1084 + 690;
          const int8_t* v1167 = (const int8_t*) v1166;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1168 = *(const int8_t *)(v1167);
          vint16m1_t v1169 = v1066;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1170 = __riscv_vwmacc_vx_i16m1(v1169, v1168, v1106, 16);
          v1066 = v1170;
          const uint8_t* v1171 = v1084 + 818;
          const int8_t* v1172 = (const int8_t*) v1171;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1173 = *(const int8_t *)(v1172);
          vint16m1_t v1174 = v1068;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1175 = __riscv_vwmacc_vx_i16m1(v1174, v1173, v1113, 16);
          v1068 = v1175;
          const uint8_t* v1176 = v1084 + 946;
          const int8_t* v1177 = (const int8_t*) v1176;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1178 = *(const int8_t *)(v1177);
          vint16m1_t v1179 = v1070;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1180 = __riscv_vwmacc_vx_i16m1(v1179, v1178, v1120, 16);
          v1070 = v1180;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1181 = v1084 + 563;
          const int8_t* v1182 = (const int8_t*) v1181;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1183 = *(const int8_t *)(v1182);
          vint16m1_t v1184 = v1072;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1185 = __riscv_vwmacc_vx_i16m1(v1184, v1183, v1099, 16);
          v1072 = v1185;
          const uint8_t* v1186 = v1084 + 691;
          const int8_t* v1187 = (const int8_t*) v1186;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1188 = *(const int8_t *)(v1187);
          vint16m1_t v1189 = v1074;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1190 = __riscv_vwmacc_vx_i16m1(v1189, v1188, v1106, 16);
          v1074 = v1190;
          const uint8_t* v1191 = v1084 + 819;
          const int8_t* v1192 = (const int8_t*) v1191;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1193 = *(const int8_t *)(v1192);
          vint16m1_t v1194 = v1076;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1195 = __riscv_vwmacc_vx_i16m1(v1194, v1193, v1113, 16);
          v1076 = v1195;
          const uint8_t* v1196 = v1084 + 947;
          const int8_t* v1197 = (const int8_t*) v1196;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1198 = *(const int8_t *)(v1197);
          vint16m1_t v1199 = v1078;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1200 = __riscv_vwmacc_vx_i16m1(v1199, v1198, v1120, 16);
          v1078 = v1200;
        }
        vint16m1_t v1201 = v1048;
        vint16m1_t v1202 = v1050;
        vint16m1_t v1203 = v1052;
        vint16m1_t v1204 = v1054;
        vint16m1_t v1205 = v1056;
        vint16m1_t v1206 = v1058;
        vint16m1_t v1207 = v1060;
        vint16m1_t v1208 = v1062;
        vint16m1_t v1209 = v1064;
        vint16m1_t v1210 = v1066;
        vint16m1_t v1211 = v1068;
        vint16m1_t v1212 = v1070;
        vint16m1_t v1213 = v1072;
        vint16m1_t v1214 = v1074;
        vint16m1_t v1215 = v1076;
        vint16m1_t v1216 = v1078;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v1217 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1218 = __riscv_vwmacc_vv_i32m2(v1217, v846, v1201, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1219 = __riscv_vwmacc_vv_i32m2(v1218, v850, v1202, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1220 = __riscv_vwmacc_vv_i32m2(v1219, v854, v1203, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1221 = __riscv_vwmacc_vv_i32m2(v1220, v858, v1204, 16);
        v47 = v1221;
        vint32m2_t v1222 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1223 = __riscv_vwmacc_vv_i32m2(v1222, v846, v1205, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1224 = __riscv_vwmacc_vv_i32m2(v1223, v850, v1206, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1225 = __riscv_vwmacc_vv_i32m2(v1224, v854, v1207, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1226 = __riscv_vwmacc_vv_i32m2(v1225, v858, v1208, 16);
        v49 = v1226;
        vint32m2_t v1227 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1228 = __riscv_vwmacc_vv_i32m2(v1227, v846, v1209, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1229 = __riscv_vwmacc_vv_i32m2(v1228, v850, v1210, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1230 = __riscv_vwmacc_vv_i32m2(v1229, v854, v1211, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1231 = __riscv_vwmacc_vv_i32m2(v1230, v858, v1212, 16);
        v51 = v1231;
        vint32m2_t v1232 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1233 = __riscv_vwmacc_vv_i32m2(v1232, v846, v1213, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1234 = __riscv_vwmacc_vv_i32m2(v1233, v850, v1214, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1235 = __riscv_vwmacc_vv_i32m2(v1234, v854, v1215, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1236 = __riscv_vwmacc_vv_i32m2(v1235, v858, v1216, 16);
        v53 = v1236;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v1237 = v30 + 176;
        const int8_t* v1238 = (const int8_t*) v1237;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1239 = __riscv_vle8_v_i8mf2(v1238, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1240 = __riscv_vsext_vf2_i16m1(v1239, 16);
        const uint8_t* v1241 = v30 + 208;
        const int8_t* v1242 = (const int8_t*) v1241;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1243 = __riscv_vle8_v_i8mf2(v1242, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1244 = __riscv_vsext_vf2_i16m1(v1243, 16);
        const uint8_t* v1245 = v30 + 240;
        const int8_t* v1246 = (const int8_t*) v1245;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1247 = __riscv_vle8_v_i8mf2(v1246, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1248 = __riscv_vsext_vf2_i16m1(v1247, 16);
        const uint8_t* v1249 = v30 + 272;
        const int8_t* v1250 = (const int8_t*) v1249;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1251 = __riscv_vle8_v_i8mf2(v1250, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1252 = __riscv_vsext_vf2_i16m1(v1251, 16);
        vint16m1_t v1253;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1254 = __riscv_vmv_v_x_i16m1(0, 16);
        v1253 = v1254;
        vint16m1_t v1255;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1256 = __riscv_vmv_v_x_i16m1(0, 16);
        v1255 = v1256;
        vint16m1_t v1257;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1258 = __riscv_vmv_v_x_i16m1(0, 16);
        v1257 = v1258;
        vint16m1_t v1259;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1260 = __riscv_vmv_v_x_i16m1(0, 16);
        v1259 = v1260;
        vint16m1_t v1261;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1262 = __riscv_vmv_v_x_i16m1(0, 16);
        v1261 = v1262;
        vint16m1_t v1263;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1264 = __riscv_vmv_v_x_i16m1(0, 16);
        v1263 = v1264;
        vint16m1_t v1265;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1266 = __riscv_vmv_v_x_i16m1(0, 16);
        v1265 = v1266;
        vint16m1_t v1267;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1268 = __riscv_vmv_v_x_i16m1(0, 16);
        v1267 = v1268;
        vint16m1_t v1269;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1270 = __riscv_vmv_v_x_i16m1(0, 16);
        v1269 = v1270;
        vint16m1_t v1271;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1272 = __riscv_vmv_v_x_i16m1(0, 16);
        v1271 = v1272;
        vint16m1_t v1273;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1274 = __riscv_vmv_v_x_i16m1(0, 16);
        v1273 = v1274;
        vint16m1_t v1275;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1276 = __riscv_vmv_v_x_i16m1(0, 16);
        v1275 = v1276;
        vint16m1_t v1277;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1278 = __riscv_vmv_v_x_i16m1(0, 16);
        v1277 = v1278;
        vint16m1_t v1279;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1280 = __riscv_vmv_v_x_i16m1(0, 16);
        v1279 = v1280;
        vint16m1_t v1281;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1282 = __riscv_vmv_v_x_i16m1(0, 16);
        v1281 = v1282;
        vint16m1_t v1283;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1284 = __riscv_vmv_v_x_i16m1(0, 16);
        v1283 = v1284;
        for (size_t v1285 = 0; v1285 < 8; v1285 += 1) {
          size_t v1286 = v1285 * 16;
          const uint8_t* v1287 = v30 + v1286;
          size_t v1288 = v1285 * 4;
          const uint8_t* v1289 = v32 + v1288;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v1290 = v1287 + 2592;
          const uint8_t* v1291 = (const uint8_t*) v1290;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1292 = __riscv_vle8_v_u8mf2(v1291, 16);
          const uint8_t* v1293 = v1287 + 3104;
          const uint8_t* v1294 = (const uint8_t*) v1293;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1295 = __riscv_vle8_v_u8mf2(v1294, 16);
          const uint8_t* v1296 = v1287 + 1056;
          const uint8_t* v1297 = (const uint8_t*) v1296;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1298 = __riscv_vle8_v_u8mf2(v1297, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1299 = __riscv_vand_vx_u8mf2(v1292, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1300 = __riscv_vand_vx_u8mf2(v1298, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1301 = __riscv_vsll_vx_u8mf2(v1300, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1302 = __riscv_vor_vv_u8mf2(v1299, v1301, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1303 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1302);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1304 = __riscv_vsub_vx_i8mf2(v1303, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1305 = __riscv_vand_vx_u8mf2(v1295, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1306 = __riscv_vsrl_vx_u8mf2(v1298, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1307 = __riscv_vand_vx_u8mf2(v1306, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1308 = __riscv_vsll_vx_u8mf2(v1307, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1309 = __riscv_vor_vv_u8mf2(v1305, v1308, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1310 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1309);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1311 = __riscv_vsub_vx_i8mf2(v1310, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1312 = __riscv_vsrl_vx_u8mf2(v1292, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1313 = __riscv_vsrl_vx_u8mf2(v1298, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1314 = __riscv_vand_vx_u8mf2(v1313, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1315 = __riscv_vsll_vx_u8mf2(v1314, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1316 = __riscv_vor_vv_u8mf2(v1312, v1315, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1317 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1316);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1318 = __riscv_vsub_vx_i8mf2(v1317, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1319 = __riscv_vsrl_vx_u8mf2(v1295, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1320 = __riscv_vsrl_vx_u8mf2(v1298, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1321 = __riscv_vand_vx_u8mf2(v1320, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1322 = __riscv_vsll_vx_u8mf2(v1321, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1323 = __riscv_vor_vv_u8mf2(v1319, v1322, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1324 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1323);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1325 = __riscv_vsub_vx_i8mf2(v1324, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1326 = v1289 + 592;
          const int8_t* v1327 = (const int8_t*) v1326;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1328 = *(const int8_t *)(v1327);
          vint16m1_t v1329 = v1253;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1330 = __riscv_vwmacc_vx_i16m1(v1329, v1328, v1304, 16);
          v1253 = v1330;
          const uint8_t* v1331 = v1289 + 720;
          const int8_t* v1332 = (const int8_t*) v1331;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1333 = *(const int8_t *)(v1332);
          vint16m1_t v1334 = v1255;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1335 = __riscv_vwmacc_vx_i16m1(v1334, v1333, v1311, 16);
          v1255 = v1335;
          const uint8_t* v1336 = v1289 + 848;
          const int8_t* v1337 = (const int8_t*) v1336;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1338 = *(const int8_t *)(v1337);
          vint16m1_t v1339 = v1257;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1340 = __riscv_vwmacc_vx_i16m1(v1339, v1338, v1318, 16);
          v1257 = v1340;
          const uint8_t* v1341 = v1289 + 976;
          const int8_t* v1342 = (const int8_t*) v1341;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1343 = *(const int8_t *)(v1342);
          vint16m1_t v1344 = v1259;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1345 = __riscv_vwmacc_vx_i16m1(v1344, v1343, v1325, 16);
          v1259 = v1345;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1346 = v1289 + 593;
          const int8_t* v1347 = (const int8_t*) v1346;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1348 = *(const int8_t *)(v1347);
          vint16m1_t v1349 = v1261;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1350 = __riscv_vwmacc_vx_i16m1(v1349, v1348, v1304, 16);
          v1261 = v1350;
          const uint8_t* v1351 = v1289 + 721;
          const int8_t* v1352 = (const int8_t*) v1351;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1353 = *(const int8_t *)(v1352);
          vint16m1_t v1354 = v1263;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1355 = __riscv_vwmacc_vx_i16m1(v1354, v1353, v1311, 16);
          v1263 = v1355;
          const uint8_t* v1356 = v1289 + 849;
          const int8_t* v1357 = (const int8_t*) v1356;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1358 = *(const int8_t *)(v1357);
          vint16m1_t v1359 = v1265;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1360 = __riscv_vwmacc_vx_i16m1(v1359, v1358, v1318, 16);
          v1265 = v1360;
          const uint8_t* v1361 = v1289 + 977;
          const int8_t* v1362 = (const int8_t*) v1361;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1363 = *(const int8_t *)(v1362);
          vint16m1_t v1364 = v1267;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1365 = __riscv_vwmacc_vx_i16m1(v1364, v1363, v1325, 16);
          v1267 = v1365;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1366 = v1289 + 594;
          const int8_t* v1367 = (const int8_t*) v1366;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1368 = *(const int8_t *)(v1367);
          vint16m1_t v1369 = v1269;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1370 = __riscv_vwmacc_vx_i16m1(v1369, v1368, v1304, 16);
          v1269 = v1370;
          const uint8_t* v1371 = v1289 + 722;
          const int8_t* v1372 = (const int8_t*) v1371;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1373 = *(const int8_t *)(v1372);
          vint16m1_t v1374 = v1271;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1375 = __riscv_vwmacc_vx_i16m1(v1374, v1373, v1311, 16);
          v1271 = v1375;
          const uint8_t* v1376 = v1289 + 850;
          const int8_t* v1377 = (const int8_t*) v1376;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1378 = *(const int8_t *)(v1377);
          vint16m1_t v1379 = v1273;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1380 = __riscv_vwmacc_vx_i16m1(v1379, v1378, v1318, 16);
          v1273 = v1380;
          const uint8_t* v1381 = v1289 + 978;
          const int8_t* v1382 = (const int8_t*) v1381;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1383 = *(const int8_t *)(v1382);
          vint16m1_t v1384 = v1275;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1385 = __riscv_vwmacc_vx_i16m1(v1384, v1383, v1325, 16);
          v1275 = v1385;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1386 = v1289 + 595;
          const int8_t* v1387 = (const int8_t*) v1386;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1388 = *(const int8_t *)(v1387);
          vint16m1_t v1389 = v1277;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1390 = __riscv_vwmacc_vx_i16m1(v1389, v1388, v1304, 16);
          v1277 = v1390;
          const uint8_t* v1391 = v1289 + 723;
          const int8_t* v1392 = (const int8_t*) v1391;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1393 = *(const int8_t *)(v1392);
          vint16m1_t v1394 = v1279;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1395 = __riscv_vwmacc_vx_i16m1(v1394, v1393, v1311, 16);
          v1279 = v1395;
          const uint8_t* v1396 = v1289 + 851;
          const int8_t* v1397 = (const int8_t*) v1396;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1398 = *(const int8_t *)(v1397);
          vint16m1_t v1399 = v1281;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1400 = __riscv_vwmacc_vx_i16m1(v1399, v1398, v1318, 16);
          v1281 = v1400;
          const uint8_t* v1401 = v1289 + 979;
          const int8_t* v1402 = (const int8_t*) v1401;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1403 = *(const int8_t *)(v1402);
          vint16m1_t v1404 = v1283;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1405 = __riscv_vwmacc_vx_i16m1(v1404, v1403, v1325, 16);
          v1283 = v1405;
        }
        vint16m1_t v1406 = v1253;
        vint16m1_t v1407 = v1255;
        vint16m1_t v1408 = v1257;
        vint16m1_t v1409 = v1259;
        vint16m1_t v1410 = v1261;
        vint16m1_t v1411 = v1263;
        vint16m1_t v1412 = v1265;
        vint16m1_t v1413 = v1267;
        vint16m1_t v1414 = v1269;
        vint16m1_t v1415 = v1271;
        vint16m1_t v1416 = v1273;
        vint16m1_t v1417 = v1275;
        vint16m1_t v1418 = v1277;
        vint16m1_t v1419 = v1279;
        vint16m1_t v1420 = v1281;
        vint16m1_t v1421 = v1283;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v1422 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1423 = __riscv_vwmacc_vv_i32m2(v1422, v1240, v1406, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1424 = __riscv_vwmacc_vv_i32m2(v1423, v1244, v1407, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1425 = __riscv_vwmacc_vv_i32m2(v1424, v1248, v1408, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1426 = __riscv_vwmacc_vv_i32m2(v1425, v1252, v1409, 16);
        v47 = v1426;
        vint32m2_t v1427 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1428 = __riscv_vwmacc_vv_i32m2(v1427, v1240, v1410, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1429 = __riscv_vwmacc_vv_i32m2(v1428, v1244, v1411, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1430 = __riscv_vwmacc_vv_i32m2(v1429, v1248, v1412, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1431 = __riscv_vwmacc_vv_i32m2(v1430, v1252, v1413, 16);
        v49 = v1431;
        vint32m2_t v1432 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1433 = __riscv_vwmacc_vv_i32m2(v1432, v1240, v1414, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1434 = __riscv_vwmacc_vv_i32m2(v1433, v1244, v1415, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1435 = __riscv_vwmacc_vv_i32m2(v1434, v1248, v1416, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1436 = __riscv_vwmacc_vv_i32m2(v1435, v1252, v1417, 16);
        v51 = v1436;
        vint32m2_t v1437 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1438 = __riscv_vwmacc_vv_i32m2(v1437, v1240, v1418, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1439 = __riscv_vwmacc_vv_i32m2(v1438, v1244, v1419, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1440 = __riscv_vwmacc_vv_i32m2(v1439, v1248, v1420, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1441 = __riscv_vwmacc_vv_i32m2(v1440, v1252, v1421, 16);
        v53 = v1441;
        vint16m1_t v1442;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1443 = __riscv_vmv_v_x_i16m1(0, 16);
        v1442 = v1443;
        vint16m1_t v1444;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1445 = __riscv_vmv_v_x_i16m1(0, 16);
        v1444 = v1445;
        vint16m1_t v1446;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1447 = __riscv_vmv_v_x_i16m1(0, 16);
        v1446 = v1447;
        vint16m1_t v1448;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1449 = __riscv_vmv_v_x_i16m1(0, 16);
        v1448 = v1449;
        vint16m1_t v1450;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1451 = __riscv_vmv_v_x_i16m1(0, 16);
        v1450 = v1451;
        vint16m1_t v1452;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1453 = __riscv_vmv_v_x_i16m1(0, 16);
        v1452 = v1453;
        vint16m1_t v1454;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1455 = __riscv_vmv_v_x_i16m1(0, 16);
        v1454 = v1455;
        vint16m1_t v1456;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1457 = __riscv_vmv_v_x_i16m1(0, 16);
        v1456 = v1457;
        vint16m1_t v1458;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1459 = __riscv_vmv_v_x_i16m1(0, 16);
        v1458 = v1459;
        vint16m1_t v1460;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1461 = __riscv_vmv_v_x_i16m1(0, 16);
        v1460 = v1461;
        vint16m1_t v1462;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1463 = __riscv_vmv_v_x_i16m1(0, 16);
        v1462 = v1463;
        vint16m1_t v1464;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1465 = __riscv_vmv_v_x_i16m1(0, 16);
        v1464 = v1465;
        vint16m1_t v1466;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1467 = __riscv_vmv_v_x_i16m1(0, 16);
        v1466 = v1467;
        vint16m1_t v1468;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1469 = __riscv_vmv_v_x_i16m1(0, 16);
        v1468 = v1469;
        vint16m1_t v1470;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1471 = __riscv_vmv_v_x_i16m1(0, 16);
        v1470 = v1471;
        vint16m1_t v1472;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1473 = __riscv_vmv_v_x_i16m1(0, 16);
        v1472 = v1473;
        for (size_t v1474 = 0; v1474 < 8; v1474 += 1) {
          size_t v1475 = v1474 * 16;
          const uint8_t* v1476 = v30 + v1475;
          size_t v1477 = v1474 * 4;
          const uint8_t* v1478 = v32 + v1477;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v1479 = v1476 + 2720;
          const uint8_t* v1480 = (const uint8_t*) v1479;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1481 = __riscv_vle8_v_u8mf2(v1480, 16);
          const uint8_t* v1482 = v1476 + 3232;
          const uint8_t* v1483 = (const uint8_t*) v1482;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1484 = __riscv_vle8_v_u8mf2(v1483, 16);
          const uint8_t* v1485 = v1476 + 1184;
          const uint8_t* v1486 = (const uint8_t*) v1485;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1487 = __riscv_vle8_v_u8mf2(v1486, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1488 = __riscv_vand_vx_u8mf2(v1481, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1489 = __riscv_vand_vx_u8mf2(v1487, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1490 = __riscv_vsll_vx_u8mf2(v1489, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1491 = __riscv_vor_vv_u8mf2(v1488, v1490, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1492 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1491);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1493 = __riscv_vsub_vx_i8mf2(v1492, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1494 = __riscv_vand_vx_u8mf2(v1484, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1495 = __riscv_vsrl_vx_u8mf2(v1487, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1496 = __riscv_vand_vx_u8mf2(v1495, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1497 = __riscv_vsll_vx_u8mf2(v1496, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1498 = __riscv_vor_vv_u8mf2(v1494, v1497, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1499 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1498);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1500 = __riscv_vsub_vx_i8mf2(v1499, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1501 = __riscv_vsrl_vx_u8mf2(v1481, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1502 = __riscv_vsrl_vx_u8mf2(v1487, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1503 = __riscv_vand_vx_u8mf2(v1502, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1504 = __riscv_vsll_vx_u8mf2(v1503, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1505 = __riscv_vor_vv_u8mf2(v1501, v1504, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1506 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1505);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1507 = __riscv_vsub_vx_i8mf2(v1506, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1508 = __riscv_vsrl_vx_u8mf2(v1484, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1509 = __riscv_vsrl_vx_u8mf2(v1487, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1510 = __riscv_vand_vx_u8mf2(v1509, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1511 = __riscv_vsll_vx_u8mf2(v1510, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1512 = __riscv_vor_vv_u8mf2(v1508, v1511, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1513 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1512);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1514 = __riscv_vsub_vx_i8mf2(v1513, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1515 = v1478 + 624;
          const int8_t* v1516 = (const int8_t*) v1515;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1517 = *(const int8_t *)(v1516);
          vint16m1_t v1518 = v1442;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1519 = __riscv_vwmacc_vx_i16m1(v1518, v1517, v1493, 16);
          v1442 = v1519;
          const uint8_t* v1520 = v1478 + 752;
          const int8_t* v1521 = (const int8_t*) v1520;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1522 = *(const int8_t *)(v1521);
          vint16m1_t v1523 = v1444;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1524 = __riscv_vwmacc_vx_i16m1(v1523, v1522, v1500, 16);
          v1444 = v1524;
          const uint8_t* v1525 = v1478 + 880;
          const int8_t* v1526 = (const int8_t*) v1525;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1527 = *(const int8_t *)(v1526);
          vint16m1_t v1528 = v1446;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1529 = __riscv_vwmacc_vx_i16m1(v1528, v1527, v1507, 16);
          v1446 = v1529;
          const uint8_t* v1530 = v1478 + 1008;
          const int8_t* v1531 = (const int8_t*) v1530;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1532 = *(const int8_t *)(v1531);
          vint16m1_t v1533 = v1448;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1534 = __riscv_vwmacc_vx_i16m1(v1533, v1532, v1514, 16);
          v1448 = v1534;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1535 = v1478 + 625;
          const int8_t* v1536 = (const int8_t*) v1535;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1537 = *(const int8_t *)(v1536);
          vint16m1_t v1538 = v1450;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1539 = __riscv_vwmacc_vx_i16m1(v1538, v1537, v1493, 16);
          v1450 = v1539;
          const uint8_t* v1540 = v1478 + 753;
          const int8_t* v1541 = (const int8_t*) v1540;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1542 = *(const int8_t *)(v1541);
          vint16m1_t v1543 = v1452;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1544 = __riscv_vwmacc_vx_i16m1(v1543, v1542, v1500, 16);
          v1452 = v1544;
          const uint8_t* v1545 = v1478 + 881;
          const int8_t* v1546 = (const int8_t*) v1545;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1547 = *(const int8_t *)(v1546);
          vint16m1_t v1548 = v1454;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1549 = __riscv_vwmacc_vx_i16m1(v1548, v1547, v1507, 16);
          v1454 = v1549;
          const uint8_t* v1550 = v1478 + 1009;
          const int8_t* v1551 = (const int8_t*) v1550;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1552 = *(const int8_t *)(v1551);
          vint16m1_t v1553 = v1456;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1554 = __riscv_vwmacc_vx_i16m1(v1553, v1552, v1514, 16);
          v1456 = v1554;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1555 = v1478 + 626;
          const int8_t* v1556 = (const int8_t*) v1555;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1557 = *(const int8_t *)(v1556);
          vint16m1_t v1558 = v1458;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1559 = __riscv_vwmacc_vx_i16m1(v1558, v1557, v1493, 16);
          v1458 = v1559;
          const uint8_t* v1560 = v1478 + 754;
          const int8_t* v1561 = (const int8_t*) v1560;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1562 = *(const int8_t *)(v1561);
          vint16m1_t v1563 = v1460;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1564 = __riscv_vwmacc_vx_i16m1(v1563, v1562, v1500, 16);
          v1460 = v1564;
          const uint8_t* v1565 = v1478 + 882;
          const int8_t* v1566 = (const int8_t*) v1565;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1567 = *(const int8_t *)(v1566);
          vint16m1_t v1568 = v1462;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1569 = __riscv_vwmacc_vx_i16m1(v1568, v1567, v1507, 16);
          v1462 = v1569;
          const uint8_t* v1570 = v1478 + 1010;
          const int8_t* v1571 = (const int8_t*) v1570;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1572 = *(const int8_t *)(v1571);
          vint16m1_t v1573 = v1464;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1574 = __riscv_vwmacc_vx_i16m1(v1573, v1572, v1514, 16);
          v1464 = v1574;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1575 = v1478 + 627;
          const int8_t* v1576 = (const int8_t*) v1575;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1577 = *(const int8_t *)(v1576);
          vint16m1_t v1578 = v1466;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1579 = __riscv_vwmacc_vx_i16m1(v1578, v1577, v1493, 16);
          v1466 = v1579;
          const uint8_t* v1580 = v1478 + 755;
          const int8_t* v1581 = (const int8_t*) v1580;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1582 = *(const int8_t *)(v1581);
          vint16m1_t v1583 = v1468;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1584 = __riscv_vwmacc_vx_i16m1(v1583, v1582, v1500, 16);
          v1468 = v1584;
          const uint8_t* v1585 = v1478 + 883;
          const int8_t* v1586 = (const int8_t*) v1585;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1587 = *(const int8_t *)(v1586);
          vint16m1_t v1588 = v1470;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1589 = __riscv_vwmacc_vx_i16m1(v1588, v1587, v1507, 16);
          v1470 = v1589;
          const uint8_t* v1590 = v1478 + 1011;
          const int8_t* v1591 = (const int8_t*) v1590;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1592 = *(const int8_t *)(v1591);
          vint16m1_t v1593 = v1472;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1594 = __riscv_vwmacc_vx_i16m1(v1593, v1592, v1514, 16);
          v1472 = v1594;
        }
        vint16m1_t v1595 = v1442;
        vint16m1_t v1596 = v1444;
        vint16m1_t v1597 = v1446;
        vint16m1_t v1598 = v1448;
        vint16m1_t v1599 = v1450;
        vint16m1_t v1600 = v1452;
        vint16m1_t v1601 = v1454;
        vint16m1_t v1602 = v1456;
        vint16m1_t v1603 = v1458;
        vint16m1_t v1604 = v1460;
        vint16m1_t v1605 = v1462;
        vint16m1_t v1606 = v1464;
        vint16m1_t v1607 = v1466;
        vint16m1_t v1608 = v1468;
        vint16m1_t v1609 = v1470;
        vint16m1_t v1610 = v1472;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v1611 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1612 = __riscv_vwmacc_vv_i32m2(v1611, v1240, v1595, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1613 = __riscv_vwmacc_vv_i32m2(v1612, v1244, v1596, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1614 = __riscv_vwmacc_vv_i32m2(v1613, v1248, v1597, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1615 = __riscv_vwmacc_vv_i32m2(v1614, v1252, v1598, 16);
        v47 = v1615;
        vint32m2_t v1616 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1617 = __riscv_vwmacc_vv_i32m2(v1616, v1240, v1599, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1618 = __riscv_vwmacc_vv_i32m2(v1617, v1244, v1600, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1619 = __riscv_vwmacc_vv_i32m2(v1618, v1248, v1601, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1620 = __riscv_vwmacc_vv_i32m2(v1619, v1252, v1602, 16);
        v49 = v1620;
        vint32m2_t v1621 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1622 = __riscv_vwmacc_vv_i32m2(v1621, v1240, v1603, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1623 = __riscv_vwmacc_vv_i32m2(v1622, v1244, v1604, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1624 = __riscv_vwmacc_vv_i32m2(v1623, v1248, v1605, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1625 = __riscv_vwmacc_vv_i32m2(v1624, v1252, v1606, 16);
        v51 = v1625;
        vint32m2_t v1626 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1627 = __riscv_vwmacc_vv_i32m2(v1626, v1240, v1607, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1628 = __riscv_vwmacc_vv_i32m2(v1627, v1244, v1608, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1629 = __riscv_vwmacc_vv_i32m2(v1628, v1248, v1609, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1630 = __riscv_vwmacc_vv_i32m2(v1629, v1252, v1610, 16);
        v53 = v1630;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1631 = __riscv_vfmul_vf_f32m2(v46, v34, 16);
        vint32m2_t v1632 = v47;
        vfloat32m2_t v1633 = v20;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1634 = __riscv_vfcvt_f_x_v_f32m2(v1632, 16);
        vfloat32m2_t v1635 = __riscv_vfmacc_vv_f32m2(v1633, v1634, v1631, 16);
        v20 = v1635;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1636 = __riscv_vfmul_vf_f32m2(v46, v37, 16);
        vint32m2_t v1637 = v49;
        vfloat32m2_t v1638 = v22;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1639 = __riscv_vfcvt_f_x_v_f32m2(v1637, 16);
        vfloat32m2_t v1640 = __riscv_vfmacc_vv_f32m2(v1638, v1639, v1636, 16);
        v22 = v1640;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1641 = __riscv_vfmul_vf_f32m2(v46, v40, 16);
        vint32m2_t v1642 = v51;
        vfloat32m2_t v1643 = v24;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1644 = __riscv_vfcvt_f_x_v_f32m2(v1642, 16);
        vfloat32m2_t v1645 = __riscv_vfmacc_vv_f32m2(v1643, v1644, v1641, 16);
        v24 = v1645;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1646 = __riscv_vfmul_vf_f32m2(v46, v43, 16);
        vint32m2_t v1647 = v53;
        vfloat32m2_t v1648 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1649 = __riscv_vfcvt_f_x_v_f32m2(v1647, 16);
        vfloat32m2_t v1650 = __riscv_vfmacc_vv_f32m2(v1648, v1649, v1646, 16);
        v26 = v1650;
      }
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1651 = v12 * 4;
      size_t v1652 = v1651 + 0;
      size_t v1653 = v1652 * v7;
      size_t v1654 = v16 * 16;
      size_t v1655 = v1653 + v1654;
      float* v1656 = v2 + v1655;
      vfloat32m2_t v1657 = v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1656, v1657, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1658 = v12 * 4;
      size_t v1659 = v1658 + 1;
      size_t v1660 = v1659 * v7;
      size_t v1661 = v16 * 16;
      size_t v1662 = v1660 + v1661;
      float* v1663 = v2 + v1662;
      vfloat32m2_t v1664 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1663, v1664, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1665 = v12 * 4;
      size_t v1666 = v1665 + 2;
      size_t v1667 = v1666 * v7;
      size_t v1668 = v16 * 16;
      size_t v1669 = v1667 + v1668;
      float* v1670 = v2 + v1669;
      vfloat32m2_t v1671 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1670, v1671, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1672 = v12 * 4;
      size_t v1673 = v1672 + 3;
      size_t v1674 = v1673 * v7;
      size_t v1675 = v16 * 16;
      size_t v1676 = v1674 + v1675;
      float* v1677 = v2 + v1676;
      vfloat32m2_t v1678 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1677, v1678, 16);
    }
  }
  return;
}


