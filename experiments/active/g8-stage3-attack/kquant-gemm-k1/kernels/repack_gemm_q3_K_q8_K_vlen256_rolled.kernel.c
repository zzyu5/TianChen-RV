#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemm_q3_K_q8_K_kernel_ggml_repack_gemm_q3_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5, size_t v6, size_t v7) {
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
      size_t v18 = v17 * 1824;
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
        size_t v29 = v28 * 1824;
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
        for (size_t v103 = 0; v103 < 16; v103 += 1) {
          size_t v104 = v103 * 16;
          const uint8_t* v105 = v30 + v104;
          size_t v106 = v103 * 4;
          const uint8_t* v107 = v32 + v106;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
          const uint8_t* v108 = v105 + 800;
          const uint8_t* v109 = (const uint8_t*) v108;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v110 = __riscv_vle8_v_u8mf2(v109, 16);
          const uint8_t* v111 = v105 + 288;
          const uint8_t* v112 = (const uint8_t*) v111;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v113 = __riscv_vle8_v_u8mf2(v112, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v114 = __riscv_vand_vx_u8mf2(v110, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v115 = __riscv_vreinterpret_v_u8mf2_i8mf2(v114);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v116 = __riscv_vand_vx_u8mf2(v113, 1, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v117 = __riscv_vmseq_vx_u8mf2_b16(v116, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v118 = __riscv_vadd_vx_i8mf2_mu(v117, v115, v115, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v119 = __riscv_vsrl_vx_u8mf2(v110, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v120 = __riscv_vand_vx_u8mf2(v119, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v121 = __riscv_vreinterpret_v_u8mf2_i8mf2(v120);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v122 = __riscv_vand_vx_u8mf2(v113, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v123 = __riscv_vmseq_vx_u8mf2_b16(v122, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v124 = __riscv_vadd_vx_i8mf2_mu(v123, v121, v121, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v125 = __riscv_vsrl_vx_u8mf2(v110, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v126 = __riscv_vand_vx_u8mf2(v125, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v127 = __riscv_vreinterpret_v_u8mf2_i8mf2(v126);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v128 = __riscv_vand_vx_u8mf2(v113, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v129 = __riscv_vmseq_vx_u8mf2_b16(v128, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v130 = __riscv_vadd_vx_i8mf2_mu(v129, v127, v127, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v131 = __riscv_vsrl_vx_u8mf2(v110, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v132 = __riscv_vand_vx_u8mf2(v131, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v133 = __riscv_vreinterpret_v_u8mf2_i8mf2(v132);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v134 = __riscv_vand_vx_u8mf2(v113, 8, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v135 = __riscv_vmseq_vx_u8mf2_b16(v134, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v136 = __riscv_vadd_vx_i8mf2_mu(v135, v133, v133, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v137 = v107 + 16;
          const int8_t* v138 = (const int8_t*) v137;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v139 = *(const int8_t *)(v138);
          vint16m1_t v140 = v71;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v141 = __riscv_vwmacc_vx_i16m1(v140, v139, v118, 16);
          v71 = v141;
          const uint8_t* v142 = v107 + 144;
          const int8_t* v143 = (const int8_t*) v142;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v144 = *(const int8_t *)(v143);
          vint16m1_t v145 = v73;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v146 = __riscv_vwmacc_vx_i16m1(v145, v144, v124, 16);
          v73 = v146;
          const uint8_t* v147 = v107 + 272;
          const int8_t* v148 = (const int8_t*) v147;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v149 = *(const int8_t *)(v148);
          vint16m1_t v150 = v75;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v151 = __riscv_vwmacc_vx_i16m1(v150, v149, v130, 16);
          v75 = v151;
          const uint8_t* v152 = v107 + 400;
          const int8_t* v153 = (const int8_t*) v152;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v154 = *(const int8_t *)(v153);
          vint16m1_t v155 = v77;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v156 = __riscv_vwmacc_vx_i16m1(v155, v154, v136, 16);
          v77 = v156;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v157 = v107 + 17;
          const int8_t* v158 = (const int8_t*) v157;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v159 = *(const int8_t *)(v158);
          vint16m1_t v160 = v79;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v161 = __riscv_vwmacc_vx_i16m1(v160, v159, v118, 16);
          v79 = v161;
          const uint8_t* v162 = v107 + 145;
          const int8_t* v163 = (const int8_t*) v162;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v164 = *(const int8_t *)(v163);
          vint16m1_t v165 = v81;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v166 = __riscv_vwmacc_vx_i16m1(v165, v164, v124, 16);
          v81 = v166;
          const uint8_t* v167 = v107 + 273;
          const int8_t* v168 = (const int8_t*) v167;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v169 = *(const int8_t *)(v168);
          vint16m1_t v170 = v83;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v171 = __riscv_vwmacc_vx_i16m1(v170, v169, v130, 16);
          v83 = v171;
          const uint8_t* v172 = v107 + 401;
          const int8_t* v173 = (const int8_t*) v172;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v174 = *(const int8_t *)(v173);
          vint16m1_t v175 = v85;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v176 = __riscv_vwmacc_vx_i16m1(v175, v174, v136, 16);
          v85 = v176;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v177 = v107 + 18;
          const int8_t* v178 = (const int8_t*) v177;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v179 = *(const int8_t *)(v178);
          vint16m1_t v180 = v87;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v181 = __riscv_vwmacc_vx_i16m1(v180, v179, v118, 16);
          v87 = v181;
          const uint8_t* v182 = v107 + 146;
          const int8_t* v183 = (const int8_t*) v182;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v184 = *(const int8_t *)(v183);
          vint16m1_t v185 = v89;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v186 = __riscv_vwmacc_vx_i16m1(v185, v184, v124, 16);
          v89 = v186;
          const uint8_t* v187 = v107 + 274;
          const int8_t* v188 = (const int8_t*) v187;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v189 = *(const int8_t *)(v188);
          vint16m1_t v190 = v91;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v191 = __riscv_vwmacc_vx_i16m1(v190, v189, v130, 16);
          v91 = v191;
          const uint8_t* v192 = v107 + 402;
          const int8_t* v193 = (const int8_t*) v192;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v194 = *(const int8_t *)(v193);
          vint16m1_t v195 = v93;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v196 = __riscv_vwmacc_vx_i16m1(v195, v194, v136, 16);
          v93 = v196;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v197 = v107 + 19;
          const int8_t* v198 = (const int8_t*) v197;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v199 = *(const int8_t *)(v198);
          vint16m1_t v200 = v95;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v201 = __riscv_vwmacc_vx_i16m1(v200, v199, v118, 16);
          v95 = v201;
          const uint8_t* v202 = v107 + 147;
          const int8_t* v203 = (const int8_t*) v202;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v204 = *(const int8_t *)(v203);
          vint16m1_t v205 = v97;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v206 = __riscv_vwmacc_vx_i16m1(v205, v204, v124, 16);
          v97 = v206;
          const uint8_t* v207 = v107 + 275;
          const int8_t* v208 = (const int8_t*) v207;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v209 = *(const int8_t *)(v208);
          vint16m1_t v210 = v99;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v211 = __riscv_vwmacc_vx_i16m1(v210, v209, v130, 16);
          v99 = v211;
          const uint8_t* v212 = v107 + 403;
          const int8_t* v213 = (const int8_t*) v212;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v214 = *(const int8_t *)(v213);
          vint16m1_t v215 = v101;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v216 = __riscv_vwmacc_vx_i16m1(v215, v214, v136, 16);
          v101 = v216;
        }
        vint16m1_t v217 = v71;
        vint16m1_t v218 = v73;
        vint16m1_t v219 = v75;
        vint16m1_t v220 = v77;
        vint16m1_t v221 = v79;
        vint16m1_t v222 = v81;
        vint16m1_t v223 = v83;
        vint16m1_t v224 = v85;
        vint16m1_t v225 = v87;
        vint16m1_t v226 = v89;
        vint16m1_t v227 = v91;
        vint16m1_t v228 = v93;
        vint16m1_t v229 = v95;
        vint16m1_t v230 = v97;
        vint16m1_t v231 = v99;
        vint16m1_t v232 = v101;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v233 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v234 = __riscv_vwmacc_vv_i32m2(v233, v58, v217, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v235 = __riscv_vwmacc_vv_i32m2(v234, v62, v218, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v236 = __riscv_vwmacc_vv_i32m2(v235, v66, v219, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v237 = __riscv_vwmacc_vv_i32m2(v236, v70, v220, 16);
        v47 = v237;
        vint32m2_t v238 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v239 = __riscv_vwmacc_vv_i32m2(v238, v58, v221, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v240 = __riscv_vwmacc_vv_i32m2(v239, v62, v222, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v241 = __riscv_vwmacc_vv_i32m2(v240, v66, v223, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v242 = __riscv_vwmacc_vv_i32m2(v241, v70, v224, 16);
        v49 = v242;
        vint32m2_t v243 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v244 = __riscv_vwmacc_vv_i32m2(v243, v58, v225, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v245 = __riscv_vwmacc_vv_i32m2(v244, v62, v226, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v246 = __riscv_vwmacc_vv_i32m2(v245, v66, v227, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v247 = __riscv_vwmacc_vv_i32m2(v246, v70, v228, 16);
        v51 = v247;
        vint32m2_t v248 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v249 = __riscv_vwmacc_vv_i32m2(v248, v58, v229, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v250 = __riscv_vwmacc_vv_i32m2(v249, v62, v230, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v251 = __riscv_vwmacc_vv_i32m2(v250, v66, v231, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v252 = __riscv_vwmacc_vv_i32m2(v251, v70, v232, 16);
        v53 = v252;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v253 = v30 + 48;
        const int8_t* v254 = (const int8_t*) v253;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v255 = __riscv_vle8_v_i8mf2(v254, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v256 = __riscv_vsext_vf2_i16m1(v255, 16);
        const uint8_t* v257 = v30 + 80;
        const int8_t* v258 = (const int8_t*) v257;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v259 = __riscv_vle8_v_i8mf2(v258, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v260 = __riscv_vsext_vf2_i16m1(v259, 16);
        const uint8_t* v261 = v30 + 112;
        const int8_t* v262 = (const int8_t*) v261;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v263 = __riscv_vle8_v_i8mf2(v262, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v264 = __riscv_vsext_vf2_i16m1(v263, 16);
        const uint8_t* v265 = v30 + 144;
        const int8_t* v266 = (const int8_t*) v265;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v267 = __riscv_vle8_v_i8mf2(v266, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v268 = __riscv_vsext_vf2_i16m1(v267, 16);
        vint16m1_t v269;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v270 = __riscv_vmv_v_x_i16m1(0, 16);
        v269 = v270;
        vint16m1_t v271;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v272 = __riscv_vmv_v_x_i16m1(0, 16);
        v271 = v272;
        vint16m1_t v273;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v274 = __riscv_vmv_v_x_i16m1(0, 16);
        v273 = v274;
        vint16m1_t v275;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v276 = __riscv_vmv_v_x_i16m1(0, 16);
        v275 = v276;
        vint16m1_t v277;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v278 = __riscv_vmv_v_x_i16m1(0, 16);
        v277 = v278;
        vint16m1_t v279;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v280 = __riscv_vmv_v_x_i16m1(0, 16);
        v279 = v280;
        vint16m1_t v281;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v282 = __riscv_vmv_v_x_i16m1(0, 16);
        v281 = v282;
        vint16m1_t v283;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v284 = __riscv_vmv_v_x_i16m1(0, 16);
        v283 = v284;
        vint16m1_t v285;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v286 = __riscv_vmv_v_x_i16m1(0, 16);
        v285 = v286;
        vint16m1_t v287;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v288 = __riscv_vmv_v_x_i16m1(0, 16);
        v287 = v288;
        vint16m1_t v289;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v290 = __riscv_vmv_v_x_i16m1(0, 16);
        v289 = v290;
        vint16m1_t v291;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v292 = __riscv_vmv_v_x_i16m1(0, 16);
        v291 = v292;
        vint16m1_t v293;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v294 = __riscv_vmv_v_x_i16m1(0, 16);
        v293 = v294;
        vint16m1_t v295;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v296 = __riscv_vmv_v_x_i16m1(0, 16);
        v295 = v296;
        vint16m1_t v297;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v298 = __riscv_vmv_v_x_i16m1(0, 16);
        v297 = v298;
        vint16m1_t v299;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v300 = __riscv_vmv_v_x_i16m1(0, 16);
        v299 = v300;
        for (size_t v301 = 0; v301 < 16; v301 += 1) {
          size_t v302 = v301 * 16;
          const uint8_t* v303 = v30 + v302;
          size_t v304 = v301 * 4;
          const uint8_t* v305 = v32 + v304;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
          const uint8_t* v306 = v303 + 1056;
          const uint8_t* v307 = (const uint8_t*) v306;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v308 = __riscv_vle8_v_u8mf2(v307, 16);
          const uint8_t* v309 = v303 + 544;
          const uint8_t* v310 = (const uint8_t*) v309;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v311 = __riscv_vle8_v_u8mf2(v310, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v312 = __riscv_vand_vx_u8mf2(v308, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v313 = __riscv_vreinterpret_v_u8mf2_i8mf2(v312);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v314 = __riscv_vand_vx_u8mf2(v311, 1, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v315 = __riscv_vmseq_vx_u8mf2_b16(v314, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v316 = __riscv_vadd_vx_i8mf2_mu(v315, v313, v313, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v317 = __riscv_vsrl_vx_u8mf2(v308, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v318 = __riscv_vand_vx_u8mf2(v317, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v319 = __riscv_vreinterpret_v_u8mf2_i8mf2(v318);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v320 = __riscv_vand_vx_u8mf2(v311, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v321 = __riscv_vmseq_vx_u8mf2_b16(v320, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v322 = __riscv_vadd_vx_i8mf2_mu(v321, v319, v319, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v323 = __riscv_vsrl_vx_u8mf2(v308, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v324 = __riscv_vand_vx_u8mf2(v323, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v325 = __riscv_vreinterpret_v_u8mf2_i8mf2(v324);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v326 = __riscv_vand_vx_u8mf2(v311, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v327 = __riscv_vmseq_vx_u8mf2_b16(v326, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v328 = __riscv_vadd_vx_i8mf2_mu(v327, v325, v325, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v329 = __riscv_vsrl_vx_u8mf2(v308, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v330 = __riscv_vand_vx_u8mf2(v329, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v331 = __riscv_vreinterpret_v_u8mf2_i8mf2(v330);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v332 = __riscv_vand_vx_u8mf2(v311, 8, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v333 = __riscv_vmseq_vx_u8mf2_b16(v332, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v334 = __riscv_vadd_vx_i8mf2_mu(v333, v331, v331, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v335 = v305 + 80;
          const int8_t* v336 = (const int8_t*) v335;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v337 = *(const int8_t *)(v336);
          vint16m1_t v338 = v269;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v339 = __riscv_vwmacc_vx_i16m1(v338, v337, v316, 16);
          v269 = v339;
          const uint8_t* v340 = v305 + 208;
          const int8_t* v341 = (const int8_t*) v340;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v342 = *(const int8_t *)(v341);
          vint16m1_t v343 = v271;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v344 = __riscv_vwmacc_vx_i16m1(v343, v342, v322, 16);
          v271 = v344;
          const uint8_t* v345 = v305 + 336;
          const int8_t* v346 = (const int8_t*) v345;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v347 = *(const int8_t *)(v346);
          vint16m1_t v348 = v273;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v349 = __riscv_vwmacc_vx_i16m1(v348, v347, v328, 16);
          v273 = v349;
          const uint8_t* v350 = v305 + 464;
          const int8_t* v351 = (const int8_t*) v350;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v352 = *(const int8_t *)(v351);
          vint16m1_t v353 = v275;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v354 = __riscv_vwmacc_vx_i16m1(v353, v352, v334, 16);
          v275 = v354;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v355 = v305 + 81;
          const int8_t* v356 = (const int8_t*) v355;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v357 = *(const int8_t *)(v356);
          vint16m1_t v358 = v277;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v359 = __riscv_vwmacc_vx_i16m1(v358, v357, v316, 16);
          v277 = v359;
          const uint8_t* v360 = v305 + 209;
          const int8_t* v361 = (const int8_t*) v360;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v362 = *(const int8_t *)(v361);
          vint16m1_t v363 = v279;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v364 = __riscv_vwmacc_vx_i16m1(v363, v362, v322, 16);
          v279 = v364;
          const uint8_t* v365 = v305 + 337;
          const int8_t* v366 = (const int8_t*) v365;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v367 = *(const int8_t *)(v366);
          vint16m1_t v368 = v281;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v369 = __riscv_vwmacc_vx_i16m1(v368, v367, v328, 16);
          v281 = v369;
          const uint8_t* v370 = v305 + 465;
          const int8_t* v371 = (const int8_t*) v370;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v372 = *(const int8_t *)(v371);
          vint16m1_t v373 = v283;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v374 = __riscv_vwmacc_vx_i16m1(v373, v372, v334, 16);
          v283 = v374;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v375 = v305 + 82;
          const int8_t* v376 = (const int8_t*) v375;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v377 = *(const int8_t *)(v376);
          vint16m1_t v378 = v285;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v379 = __riscv_vwmacc_vx_i16m1(v378, v377, v316, 16);
          v285 = v379;
          const uint8_t* v380 = v305 + 210;
          const int8_t* v381 = (const int8_t*) v380;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v382 = *(const int8_t *)(v381);
          vint16m1_t v383 = v287;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v384 = __riscv_vwmacc_vx_i16m1(v383, v382, v322, 16);
          v287 = v384;
          const uint8_t* v385 = v305 + 338;
          const int8_t* v386 = (const int8_t*) v385;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v387 = *(const int8_t *)(v386);
          vint16m1_t v388 = v289;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v389 = __riscv_vwmacc_vx_i16m1(v388, v387, v328, 16);
          v289 = v389;
          const uint8_t* v390 = v305 + 466;
          const int8_t* v391 = (const int8_t*) v390;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v392 = *(const int8_t *)(v391);
          vint16m1_t v393 = v291;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v394 = __riscv_vwmacc_vx_i16m1(v393, v392, v334, 16);
          v291 = v394;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v395 = v305 + 83;
          const int8_t* v396 = (const int8_t*) v395;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v397 = *(const int8_t *)(v396);
          vint16m1_t v398 = v293;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v399 = __riscv_vwmacc_vx_i16m1(v398, v397, v316, 16);
          v293 = v399;
          const uint8_t* v400 = v305 + 211;
          const int8_t* v401 = (const int8_t*) v400;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v402 = *(const int8_t *)(v401);
          vint16m1_t v403 = v295;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v404 = __riscv_vwmacc_vx_i16m1(v403, v402, v322, 16);
          v295 = v404;
          const uint8_t* v405 = v305 + 339;
          const int8_t* v406 = (const int8_t*) v405;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v407 = *(const int8_t *)(v406);
          vint16m1_t v408 = v297;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v409 = __riscv_vwmacc_vx_i16m1(v408, v407, v328, 16);
          v297 = v409;
          const uint8_t* v410 = v305 + 467;
          const int8_t* v411 = (const int8_t*) v410;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v412 = *(const int8_t *)(v411);
          vint16m1_t v413 = v299;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v414 = __riscv_vwmacc_vx_i16m1(v413, v412, v334, 16);
          v299 = v414;
        }
        vint16m1_t v415 = v269;
        vint16m1_t v416 = v271;
        vint16m1_t v417 = v273;
        vint16m1_t v418 = v275;
        vint16m1_t v419 = v277;
        vint16m1_t v420 = v279;
        vint16m1_t v421 = v281;
        vint16m1_t v422 = v283;
        vint16m1_t v423 = v285;
        vint16m1_t v424 = v287;
        vint16m1_t v425 = v289;
        vint16m1_t v426 = v291;
        vint16m1_t v427 = v293;
        vint16m1_t v428 = v295;
        vint16m1_t v429 = v297;
        vint16m1_t v430 = v299;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v431 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v432 = __riscv_vwmacc_vv_i32m2(v431, v256, v415, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v433 = __riscv_vwmacc_vv_i32m2(v432, v260, v416, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v434 = __riscv_vwmacc_vv_i32m2(v433, v264, v417, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v435 = __riscv_vwmacc_vv_i32m2(v434, v268, v418, 16);
        v47 = v435;
        vint32m2_t v436 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v437 = __riscv_vwmacc_vv_i32m2(v436, v256, v419, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v438 = __riscv_vwmacc_vv_i32m2(v437, v260, v420, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v439 = __riscv_vwmacc_vv_i32m2(v438, v264, v421, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v440 = __riscv_vwmacc_vv_i32m2(v439, v268, v422, 16);
        v49 = v440;
        vint32m2_t v441 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v442 = __riscv_vwmacc_vv_i32m2(v441, v256, v423, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v443 = __riscv_vwmacc_vv_i32m2(v442, v260, v424, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v444 = __riscv_vwmacc_vv_i32m2(v443, v264, v425, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v445 = __riscv_vwmacc_vv_i32m2(v444, v268, v426, 16);
        v51 = v445;
        vint32m2_t v446 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v447 = __riscv_vwmacc_vv_i32m2(v446, v256, v427, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v448 = __riscv_vwmacc_vv_i32m2(v447, v260, v428, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v449 = __riscv_vwmacc_vv_i32m2(v448, v264, v429, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v450 = __riscv_vwmacc_vv_i32m2(v449, v268, v430, 16);
        v53 = v450;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v451 = v30 + 160;
        const int8_t* v452 = (const int8_t*) v451;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v453 = __riscv_vle8_v_i8mf2(v452, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v454 = __riscv_vsext_vf2_i16m1(v453, 16);
        const uint8_t* v455 = v30 + 192;
        const int8_t* v456 = (const int8_t*) v455;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v457 = __riscv_vle8_v_i8mf2(v456, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v458 = __riscv_vsext_vf2_i16m1(v457, 16);
        const uint8_t* v459 = v30 + 224;
        const int8_t* v460 = (const int8_t*) v459;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v461 = __riscv_vle8_v_i8mf2(v460, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v462 = __riscv_vsext_vf2_i16m1(v461, 16);
        const uint8_t* v463 = v30 + 256;
        const int8_t* v464 = (const int8_t*) v463;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v465 = __riscv_vle8_v_i8mf2(v464, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v466 = __riscv_vsext_vf2_i16m1(v465, 16);
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
        vint16m1_t v497;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v498 = __riscv_vmv_v_x_i16m1(0, 16);
        v497 = v498;
        for (size_t v499 = 0; v499 < 16; v499 += 1) {
          size_t v500 = v499 * 16;
          const uint8_t* v501 = v30 + v500;
          size_t v502 = v499 * 4;
          const uint8_t* v503 = v32 + v502;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
          const uint8_t* v504 = v501 + 1312;
          const uint8_t* v505 = (const uint8_t*) v504;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v506 = __riscv_vle8_v_u8mf2(v505, 16);
          const uint8_t* v507 = v501 + 288;
          const uint8_t* v508 = (const uint8_t*) v507;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v509 = __riscv_vle8_v_u8mf2(v508, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v510 = __riscv_vand_vx_u8mf2(v506, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v511 = __riscv_vreinterpret_v_u8mf2_i8mf2(v510);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v512 = __riscv_vand_vx_u8mf2(v509, 16, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v513 = __riscv_vmseq_vx_u8mf2_b16(v512, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v514 = __riscv_vadd_vx_i8mf2_mu(v513, v511, v511, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v515 = __riscv_vsrl_vx_u8mf2(v506, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v516 = __riscv_vand_vx_u8mf2(v515, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v517 = __riscv_vreinterpret_v_u8mf2_i8mf2(v516);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v518 = __riscv_vand_vx_u8mf2(v509, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v519 = __riscv_vmseq_vx_u8mf2_b16(v518, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v520 = __riscv_vadd_vx_i8mf2_mu(v519, v517, v517, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v521 = __riscv_vsrl_vx_u8mf2(v506, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v522 = __riscv_vand_vx_u8mf2(v521, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v523 = __riscv_vreinterpret_v_u8mf2_i8mf2(v522);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v524 = __riscv_vand_vx_u8mf2(v509, 64, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v525 = __riscv_vmseq_vx_u8mf2_b16(v524, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v526 = __riscv_vadd_vx_i8mf2_mu(v525, v523, v523, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v527 = __riscv_vsrl_vx_u8mf2(v506, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v528 = __riscv_vand_vx_u8mf2(v527, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v529 = __riscv_vreinterpret_v_u8mf2_i8mf2(v528);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v530 = __riscv_vand_vx_u8mf2(v509, 128, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v531 = __riscv_vmseq_vx_u8mf2_b16(v530, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v532 = __riscv_vadd_vx_i8mf2_mu(v531, v529, v529, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v533 = v503 + 528;
          const int8_t* v534 = (const int8_t*) v533;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v535 = *(const int8_t *)(v534);
          vint16m1_t v536 = v467;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v537 = __riscv_vwmacc_vx_i16m1(v536, v535, v514, 16);
          v467 = v537;
          const uint8_t* v538 = v503 + 656;
          const int8_t* v539 = (const int8_t*) v538;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v540 = *(const int8_t *)(v539);
          vint16m1_t v541 = v469;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v542 = __riscv_vwmacc_vx_i16m1(v541, v540, v520, 16);
          v469 = v542;
          const uint8_t* v543 = v503 + 784;
          const int8_t* v544 = (const int8_t*) v543;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v545 = *(const int8_t *)(v544);
          vint16m1_t v546 = v471;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v547 = __riscv_vwmacc_vx_i16m1(v546, v545, v526, 16);
          v471 = v547;
          const uint8_t* v548 = v503 + 912;
          const int8_t* v549 = (const int8_t*) v548;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v550 = *(const int8_t *)(v549);
          vint16m1_t v551 = v473;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v552 = __riscv_vwmacc_vx_i16m1(v551, v550, v532, 16);
          v473 = v552;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v553 = v503 + 529;
          const int8_t* v554 = (const int8_t*) v553;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v555 = *(const int8_t *)(v554);
          vint16m1_t v556 = v475;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v557 = __riscv_vwmacc_vx_i16m1(v556, v555, v514, 16);
          v475 = v557;
          const uint8_t* v558 = v503 + 657;
          const int8_t* v559 = (const int8_t*) v558;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v560 = *(const int8_t *)(v559);
          vint16m1_t v561 = v477;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v562 = __riscv_vwmacc_vx_i16m1(v561, v560, v520, 16);
          v477 = v562;
          const uint8_t* v563 = v503 + 785;
          const int8_t* v564 = (const int8_t*) v563;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v565 = *(const int8_t *)(v564);
          vint16m1_t v566 = v479;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v567 = __riscv_vwmacc_vx_i16m1(v566, v565, v526, 16);
          v479 = v567;
          const uint8_t* v568 = v503 + 913;
          const int8_t* v569 = (const int8_t*) v568;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v570 = *(const int8_t *)(v569);
          vint16m1_t v571 = v481;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v572 = __riscv_vwmacc_vx_i16m1(v571, v570, v532, 16);
          v481 = v572;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v573 = v503 + 530;
          const int8_t* v574 = (const int8_t*) v573;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v575 = *(const int8_t *)(v574);
          vint16m1_t v576 = v483;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v577 = __riscv_vwmacc_vx_i16m1(v576, v575, v514, 16);
          v483 = v577;
          const uint8_t* v578 = v503 + 658;
          const int8_t* v579 = (const int8_t*) v578;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v580 = *(const int8_t *)(v579);
          vint16m1_t v581 = v485;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v582 = __riscv_vwmacc_vx_i16m1(v581, v580, v520, 16);
          v485 = v582;
          const uint8_t* v583 = v503 + 786;
          const int8_t* v584 = (const int8_t*) v583;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v585 = *(const int8_t *)(v584);
          vint16m1_t v586 = v487;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v587 = __riscv_vwmacc_vx_i16m1(v586, v585, v526, 16);
          v487 = v587;
          const uint8_t* v588 = v503 + 914;
          const int8_t* v589 = (const int8_t*) v588;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v590 = *(const int8_t *)(v589);
          vint16m1_t v591 = v489;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v592 = __riscv_vwmacc_vx_i16m1(v591, v590, v532, 16);
          v489 = v592;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v593 = v503 + 531;
          const int8_t* v594 = (const int8_t*) v593;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v595 = *(const int8_t *)(v594);
          vint16m1_t v596 = v491;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v597 = __riscv_vwmacc_vx_i16m1(v596, v595, v514, 16);
          v491 = v597;
          const uint8_t* v598 = v503 + 659;
          const int8_t* v599 = (const int8_t*) v598;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v600 = *(const int8_t *)(v599);
          vint16m1_t v601 = v493;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v602 = __riscv_vwmacc_vx_i16m1(v601, v600, v520, 16);
          v493 = v602;
          const uint8_t* v603 = v503 + 787;
          const int8_t* v604 = (const int8_t*) v603;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v605 = *(const int8_t *)(v604);
          vint16m1_t v606 = v495;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v607 = __riscv_vwmacc_vx_i16m1(v606, v605, v526, 16);
          v495 = v607;
          const uint8_t* v608 = v503 + 915;
          const int8_t* v609 = (const int8_t*) v608;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v610 = *(const int8_t *)(v609);
          vint16m1_t v611 = v497;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v612 = __riscv_vwmacc_vx_i16m1(v611, v610, v532, 16);
          v497 = v612;
        }
        vint16m1_t v613 = v467;
        vint16m1_t v614 = v469;
        vint16m1_t v615 = v471;
        vint16m1_t v616 = v473;
        vint16m1_t v617 = v475;
        vint16m1_t v618 = v477;
        vint16m1_t v619 = v479;
        vint16m1_t v620 = v481;
        vint16m1_t v621 = v483;
        vint16m1_t v622 = v485;
        vint16m1_t v623 = v487;
        vint16m1_t v624 = v489;
        vint16m1_t v625 = v491;
        vint16m1_t v626 = v493;
        vint16m1_t v627 = v495;
        vint16m1_t v628 = v497;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v629 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v630 = __riscv_vwmacc_vv_i32m2(v629, v454, v613, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v631 = __riscv_vwmacc_vv_i32m2(v630, v458, v614, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v632 = __riscv_vwmacc_vv_i32m2(v631, v462, v615, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v633 = __riscv_vwmacc_vv_i32m2(v632, v466, v616, 16);
        v47 = v633;
        vint32m2_t v634 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v635 = __riscv_vwmacc_vv_i32m2(v634, v454, v617, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v636 = __riscv_vwmacc_vv_i32m2(v635, v458, v618, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v637 = __riscv_vwmacc_vv_i32m2(v636, v462, v619, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v638 = __riscv_vwmacc_vv_i32m2(v637, v466, v620, 16);
        v49 = v638;
        vint32m2_t v639 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v640 = __riscv_vwmacc_vv_i32m2(v639, v454, v621, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v641 = __riscv_vwmacc_vv_i32m2(v640, v458, v622, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v642 = __riscv_vwmacc_vv_i32m2(v641, v462, v623, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v643 = __riscv_vwmacc_vv_i32m2(v642, v466, v624, 16);
        v51 = v643;
        vint32m2_t v644 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v645 = __riscv_vwmacc_vv_i32m2(v644, v454, v625, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v646 = __riscv_vwmacc_vv_i32m2(v645, v458, v626, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v647 = __riscv_vwmacc_vv_i32m2(v646, v462, v627, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v648 = __riscv_vwmacc_vv_i32m2(v647, v466, v628, 16);
        v53 = v648;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v649 = v30 + 176;
        const int8_t* v650 = (const int8_t*) v649;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v651 = __riscv_vle8_v_i8mf2(v650, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v652 = __riscv_vsext_vf2_i16m1(v651, 16);
        const uint8_t* v653 = v30 + 208;
        const int8_t* v654 = (const int8_t*) v653;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v655 = __riscv_vle8_v_i8mf2(v654, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v656 = __riscv_vsext_vf2_i16m1(v655, 16);
        const uint8_t* v657 = v30 + 240;
        const int8_t* v658 = (const int8_t*) v657;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v659 = __riscv_vle8_v_i8mf2(v658, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v660 = __riscv_vsext_vf2_i16m1(v659, 16);
        const uint8_t* v661 = v30 + 272;
        const int8_t* v662 = (const int8_t*) v661;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v663 = __riscv_vle8_v_i8mf2(v662, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v664 = __riscv_vsext_vf2_i16m1(v663, 16);
        vint16m1_t v665;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v666 = __riscv_vmv_v_x_i16m1(0, 16);
        v665 = v666;
        vint16m1_t v667;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v668 = __riscv_vmv_v_x_i16m1(0, 16);
        v667 = v668;
        vint16m1_t v669;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v670 = __riscv_vmv_v_x_i16m1(0, 16);
        v669 = v670;
        vint16m1_t v671;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v672 = __riscv_vmv_v_x_i16m1(0, 16);
        v671 = v672;
        vint16m1_t v673;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v674 = __riscv_vmv_v_x_i16m1(0, 16);
        v673 = v674;
        vint16m1_t v675;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v676 = __riscv_vmv_v_x_i16m1(0, 16);
        v675 = v676;
        vint16m1_t v677;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v678 = __riscv_vmv_v_x_i16m1(0, 16);
        v677 = v678;
        vint16m1_t v679;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v680 = __riscv_vmv_v_x_i16m1(0, 16);
        v679 = v680;
        vint16m1_t v681;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v682 = __riscv_vmv_v_x_i16m1(0, 16);
        v681 = v682;
        vint16m1_t v683;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v684 = __riscv_vmv_v_x_i16m1(0, 16);
        v683 = v684;
        vint16m1_t v685;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v686 = __riscv_vmv_v_x_i16m1(0, 16);
        v685 = v686;
        vint16m1_t v687;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v688 = __riscv_vmv_v_x_i16m1(0, 16);
        v687 = v688;
        vint16m1_t v689;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v690 = __riscv_vmv_v_x_i16m1(0, 16);
        v689 = v690;
        vint16m1_t v691;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v692 = __riscv_vmv_v_x_i16m1(0, 16);
        v691 = v692;
        vint16m1_t v693;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v694 = __riscv_vmv_v_x_i16m1(0, 16);
        v693 = v694;
        vint16m1_t v695;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v696 = __riscv_vmv_v_x_i16m1(0, 16);
        v695 = v696;
        for (size_t v697 = 0; v697 < 16; v697 += 1) {
          size_t v698 = v697 * 16;
          const uint8_t* v699 = v30 + v698;
          size_t v700 = v697 * 4;
          const uint8_t* v701 = v32 + v700;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
          const uint8_t* v702 = v699 + 1568;
          const uint8_t* v703 = (const uint8_t*) v702;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v704 = __riscv_vle8_v_u8mf2(v703, 16);
          const uint8_t* v705 = v699 + 544;
          const uint8_t* v706 = (const uint8_t*) v705;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v707 = __riscv_vle8_v_u8mf2(v706, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v708 = __riscv_vand_vx_u8mf2(v704, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v709 = __riscv_vreinterpret_v_u8mf2_i8mf2(v708);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v710 = __riscv_vand_vx_u8mf2(v707, 16, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v711 = __riscv_vmseq_vx_u8mf2_b16(v710, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v712 = __riscv_vadd_vx_i8mf2_mu(v711, v709, v709, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v713 = __riscv_vsrl_vx_u8mf2(v704, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v714 = __riscv_vand_vx_u8mf2(v713, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v715 = __riscv_vreinterpret_v_u8mf2_i8mf2(v714);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v716 = __riscv_vand_vx_u8mf2(v707, 32, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v717 = __riscv_vmseq_vx_u8mf2_b16(v716, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v718 = __riscv_vadd_vx_i8mf2_mu(v717, v715, v715, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v719 = __riscv_vsrl_vx_u8mf2(v704, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v720 = __riscv_vand_vx_u8mf2(v719, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v721 = __riscv_vreinterpret_v_u8mf2_i8mf2(v720);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v722 = __riscv_vand_vx_u8mf2(v707, 64, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v723 = __riscv_vmseq_vx_u8mf2_b16(v722, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v724 = __riscv_vadd_vx_i8mf2_mu(v723, v721, v721, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v725 = __riscv_vsrl_vx_u8mf2(v704, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v726 = __riscv_vand_vx_u8mf2(v725, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v727 = __riscv_vreinterpret_v_u8mf2_i8mf2(v726);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v728 = __riscv_vand_vx_u8mf2(v707, 128, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v729 = __riscv_vmseq_vx_u8mf2_b16(v728, 0, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v730 = __riscv_vadd_vx_i8mf2_mu(v729, v727, v727, -4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v731 = v701 + 592;
          const int8_t* v732 = (const int8_t*) v731;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v733 = *(const int8_t *)(v732);
          vint16m1_t v734 = v665;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v735 = __riscv_vwmacc_vx_i16m1(v734, v733, v712, 16);
          v665 = v735;
          const uint8_t* v736 = v701 + 720;
          const int8_t* v737 = (const int8_t*) v736;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v738 = *(const int8_t *)(v737);
          vint16m1_t v739 = v667;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v740 = __riscv_vwmacc_vx_i16m1(v739, v738, v718, 16);
          v667 = v740;
          const uint8_t* v741 = v701 + 848;
          const int8_t* v742 = (const int8_t*) v741;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v743 = *(const int8_t *)(v742);
          vint16m1_t v744 = v669;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v745 = __riscv_vwmacc_vx_i16m1(v744, v743, v724, 16);
          v669 = v745;
          const uint8_t* v746 = v701 + 976;
          const int8_t* v747 = (const int8_t*) v746;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v748 = *(const int8_t *)(v747);
          vint16m1_t v749 = v671;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v750 = __riscv_vwmacc_vx_i16m1(v749, v748, v730, 16);
          v671 = v750;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v751 = v701 + 593;
          const int8_t* v752 = (const int8_t*) v751;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v753 = *(const int8_t *)(v752);
          vint16m1_t v754 = v673;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v755 = __riscv_vwmacc_vx_i16m1(v754, v753, v712, 16);
          v673 = v755;
          const uint8_t* v756 = v701 + 721;
          const int8_t* v757 = (const int8_t*) v756;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v758 = *(const int8_t *)(v757);
          vint16m1_t v759 = v675;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v760 = __riscv_vwmacc_vx_i16m1(v759, v758, v718, 16);
          v675 = v760;
          const uint8_t* v761 = v701 + 849;
          const int8_t* v762 = (const int8_t*) v761;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v763 = *(const int8_t *)(v762);
          vint16m1_t v764 = v677;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v765 = __riscv_vwmacc_vx_i16m1(v764, v763, v724, 16);
          v677 = v765;
          const uint8_t* v766 = v701 + 977;
          const int8_t* v767 = (const int8_t*) v766;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v768 = *(const int8_t *)(v767);
          vint16m1_t v769 = v679;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v770 = __riscv_vwmacc_vx_i16m1(v769, v768, v730, 16);
          v679 = v770;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v771 = v701 + 594;
          const int8_t* v772 = (const int8_t*) v771;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v773 = *(const int8_t *)(v772);
          vint16m1_t v774 = v681;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v775 = __riscv_vwmacc_vx_i16m1(v774, v773, v712, 16);
          v681 = v775;
          const uint8_t* v776 = v701 + 722;
          const int8_t* v777 = (const int8_t*) v776;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v778 = *(const int8_t *)(v777);
          vint16m1_t v779 = v683;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v780 = __riscv_vwmacc_vx_i16m1(v779, v778, v718, 16);
          v683 = v780;
          const uint8_t* v781 = v701 + 850;
          const int8_t* v782 = (const int8_t*) v781;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v783 = *(const int8_t *)(v782);
          vint16m1_t v784 = v685;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v785 = __riscv_vwmacc_vx_i16m1(v784, v783, v724, 16);
          v685 = v785;
          const uint8_t* v786 = v701 + 978;
          const int8_t* v787 = (const int8_t*) v786;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v788 = *(const int8_t *)(v787);
          vint16m1_t v789 = v687;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v790 = __riscv_vwmacc_vx_i16m1(v789, v788, v730, 16);
          v687 = v790;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v791 = v701 + 595;
          const int8_t* v792 = (const int8_t*) v791;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v793 = *(const int8_t *)(v792);
          vint16m1_t v794 = v689;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v795 = __riscv_vwmacc_vx_i16m1(v794, v793, v712, 16);
          v689 = v795;
          const uint8_t* v796 = v701 + 723;
          const int8_t* v797 = (const int8_t*) v796;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v798 = *(const int8_t *)(v797);
          vint16m1_t v799 = v691;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v800 = __riscv_vwmacc_vx_i16m1(v799, v798, v718, 16);
          v691 = v800;
          const uint8_t* v801 = v701 + 851;
          const int8_t* v802 = (const int8_t*) v801;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v803 = *(const int8_t *)(v802);
          vint16m1_t v804 = v693;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v805 = __riscv_vwmacc_vx_i16m1(v804, v803, v724, 16);
          v693 = v805;
          const uint8_t* v806 = v701 + 979;
          const int8_t* v807 = (const int8_t*) v806;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v808 = *(const int8_t *)(v807);
          vint16m1_t v809 = v695;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v810 = __riscv_vwmacc_vx_i16m1(v809, v808, v730, 16);
          v695 = v810;
        }
        vint16m1_t v811 = v665;
        vint16m1_t v812 = v667;
        vint16m1_t v813 = v669;
        vint16m1_t v814 = v671;
        vint16m1_t v815 = v673;
        vint16m1_t v816 = v675;
        vint16m1_t v817 = v677;
        vint16m1_t v818 = v679;
        vint16m1_t v819 = v681;
        vint16m1_t v820 = v683;
        vint16m1_t v821 = v685;
        vint16m1_t v822 = v687;
        vint16m1_t v823 = v689;
        vint16m1_t v824 = v691;
        vint16m1_t v825 = v693;
        vint16m1_t v826 = v695;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v827 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v828 = __riscv_vwmacc_vv_i32m2(v827, v652, v811, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v829 = __riscv_vwmacc_vv_i32m2(v828, v656, v812, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v830 = __riscv_vwmacc_vv_i32m2(v829, v660, v813, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v831 = __riscv_vwmacc_vv_i32m2(v830, v664, v814, 16);
        v47 = v831;
        vint32m2_t v832 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v833 = __riscv_vwmacc_vv_i32m2(v832, v652, v815, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v834 = __riscv_vwmacc_vv_i32m2(v833, v656, v816, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v835 = __riscv_vwmacc_vv_i32m2(v834, v660, v817, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v836 = __riscv_vwmacc_vv_i32m2(v835, v664, v818, 16);
        v49 = v836;
        vint32m2_t v837 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v838 = __riscv_vwmacc_vv_i32m2(v837, v652, v819, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v839 = __riscv_vwmacc_vv_i32m2(v838, v656, v820, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v840 = __riscv_vwmacc_vv_i32m2(v839, v660, v821, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v841 = __riscv_vwmacc_vv_i32m2(v840, v664, v822, 16);
        v51 = v841;
        vint32m2_t v842 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v843 = __riscv_vwmacc_vv_i32m2(v842, v652, v823, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v844 = __riscv_vwmacc_vv_i32m2(v843, v656, v824, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v845 = __riscv_vwmacc_vv_i32m2(v844, v660, v825, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v846 = __riscv_vwmacc_vv_i32m2(v845, v664, v826, 16);
        v53 = v846;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v847 = __riscv_vfmul_vf_f32m2(v46, v34, 16);
        vint32m2_t v848 = v47;
        vfloat32m2_t v849 = v20;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v850 = __riscv_vfcvt_f_x_v_f32m2(v848, 16);
        vfloat32m2_t v851 = __riscv_vfmacc_vv_f32m2(v849, v850, v847, 16);
        v20 = v851;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v852 = __riscv_vfmul_vf_f32m2(v46, v37, 16);
        vint32m2_t v853 = v49;
        vfloat32m2_t v854 = v22;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v855 = __riscv_vfcvt_f_x_v_f32m2(v853, 16);
        vfloat32m2_t v856 = __riscv_vfmacc_vv_f32m2(v854, v855, v852, 16);
        v22 = v856;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v857 = __riscv_vfmul_vf_f32m2(v46, v40, 16);
        vint32m2_t v858 = v51;
        vfloat32m2_t v859 = v24;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v860 = __riscv_vfcvt_f_x_v_f32m2(v858, 16);
        vfloat32m2_t v861 = __riscv_vfmacc_vv_f32m2(v859, v860, v857, 16);
        v24 = v861;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v862 = __riscv_vfmul_vf_f32m2(v46, v43, 16);
        vint32m2_t v863 = v53;
        vfloat32m2_t v864 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v865 = __riscv_vfcvt_f_x_v_f32m2(v863, 16);
        vfloat32m2_t v866 = __riscv_vfmacc_vv_f32m2(v864, v865, v862, 16);
        v26 = v866;
      }
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v867 = v12 * 4;
      size_t v868 = v867 + 0;
      size_t v869 = v868 * v7;
      size_t v870 = v16 * 16;
      size_t v871 = v869 + v870;
      float* v872 = v2 + v871;
      vfloat32m2_t v873 = v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v872, v873, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v874 = v12 * 4;
      size_t v875 = v874 + 1;
      size_t v876 = v875 * v7;
      size_t v877 = v16 * 16;
      size_t v878 = v876 + v877;
      float* v879 = v2 + v878;
      vfloat32m2_t v880 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v879, v880, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v881 = v12 * 4;
      size_t v882 = v881 + 2;
      size_t v883 = v882 * v7;
      size_t v884 = v16 * 16;
      size_t v885 = v883 + v884;
      float* v886 = v2 + v885;
      vfloat32m2_t v887 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v886, v887, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v888 = v12 * 4;
      size_t v889 = v888 + 3;
      size_t v890 = v889 * v7;
      size_t v891 = v16 * 16;
      size_t v892 = v890 + v891;
      float* v893 = v2 + v892;
      vfloat32m2_t v894 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v893, v894, 16);
    }
  }
  return;
}


