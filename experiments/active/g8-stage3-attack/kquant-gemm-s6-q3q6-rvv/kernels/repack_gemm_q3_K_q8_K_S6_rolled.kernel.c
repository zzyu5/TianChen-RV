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
        vfloat16m1_t v45 = __riscv_vle16_v_f16m1(v44, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v46 = __riscv_vfwcvt_f_f_v_f32m2(v45, 8);
        vint32m2_t v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v48 = __riscv_vmv_v_x_i32m2(0, 8);
        v47 = v48;
        vint32m2_t v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v50 = __riscv_vmv_v_x_i32m2(0, 8);
        v49 = v50;
        vint32m2_t v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v52 = __riscv_vmv_v_x_i32m2(0, 8);
        v51 = v52;
        vint32m2_t v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v54 = __riscv_vmv_v_x_i32m2(0, 8);
        v53 = v54;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v55 = v30 + 32;
        const int8_t* v56 = (const int8_t*) v55;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v57 = __riscv_vle8_v_i8mf2(v56, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v58 = __riscv_vsext_vf2_i16m1(v57, 8);
        const uint8_t* v59 = v30 + 64;
        const int8_t* v60 = (const int8_t*) v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v61 = __riscv_vle8_v_i8mf2(v60, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v62 = __riscv_vsext_vf2_i16m1(v61, 8);
        const uint8_t* v63 = v30 + 96;
        const int8_t* v64 = (const int8_t*) v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v65 = __riscv_vle8_v_i8mf2(v64, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v66 = __riscv_vsext_vf2_i16m1(v65, 8);
        const uint8_t* v67 = v30 + 128;
        const int8_t* v68 = (const int8_t*) v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v69 = __riscv_vle8_v_i8mf2(v68, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v70 = __riscv_vsext_vf2_i16m1(v69, 8);
        vint16m1_t v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v72 = __riscv_vmv_v_x_i16m1(0, 8);
        v71 = v72;
        vint16m1_t v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v74 = __riscv_vmv_v_x_i16m1(0, 8);
        v73 = v74;
        vint16m1_t v75;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v76 = __riscv_vmv_v_x_i16m1(0, 8);
        v75 = v76;
        vint16m1_t v77;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v78 = __riscv_vmv_v_x_i16m1(0, 8);
        v77 = v78;
        vint16m1_t v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v80 = __riscv_vmv_v_x_i16m1(0, 8);
        v79 = v80;
        vint16m1_t v81;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v82 = __riscv_vmv_v_x_i16m1(0, 8);
        v81 = v82;
        vint16m1_t v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v84 = __riscv_vmv_v_x_i16m1(0, 8);
        v83 = v84;
        vint16m1_t v85;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v86 = __riscv_vmv_v_x_i16m1(0, 8);
        v85 = v86;
        vint16m1_t v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v88 = __riscv_vmv_v_x_i16m1(0, 8);
        v87 = v88;
        vint16m1_t v89;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v90 = __riscv_vmv_v_x_i16m1(0, 8);
        v89 = v90;
        vint16m1_t v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v92 = __riscv_vmv_v_x_i16m1(0, 8);
        v91 = v92;
        vint16m1_t v93;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v94 = __riscv_vmv_v_x_i16m1(0, 8);
        v93 = v94;
        vint16m1_t v95;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v96 = __riscv_vmv_v_x_i16m1(0, 8);
        v95 = v96;
        vint16m1_t v97;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v98 = __riscv_vmv_v_x_i16m1(0, 8);
        v97 = v98;
        vint16m1_t v99;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v100 = __riscv_vmv_v_x_i16m1(0, 8);
        v99 = v100;
        vint16m1_t v101;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v102 = __riscv_vmv_v_x_i16m1(0, 8);
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
          vuint8mf2_t v110 = __riscv_vle8_v_u8mf2(v109, 8);
          const uint8_t* v111 = v105 + 288;
          const uint8_t* v112 = (const uint8_t*) v111;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v113 = __riscv_vle8_v_u8mf2(v112, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v114 = __riscv_vand_vx_u8mf2(v110, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v115 = __riscv_vreinterpret_v_u8mf2_i8mf2(v114);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v116 = __riscv_vand_vx_u8mf2(v113, 1, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v117 = __riscv_vmseq_vx_u8mf2_b16(v116, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v118 = __riscv_vadd_vx_i8mf2_mu(v117, v115, v115, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v119 = __riscv_vsrl_vx_u8mf2(v110, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v120 = __riscv_vand_vx_u8mf2(v119, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v121 = __riscv_vreinterpret_v_u8mf2_i8mf2(v120);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v122 = __riscv_vand_vx_u8mf2(v113, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v123 = __riscv_vmseq_vx_u8mf2_b16(v122, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v124 = __riscv_vadd_vx_i8mf2_mu(v123, v121, v121, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v125 = __riscv_vsrl_vx_u8mf2(v110, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v126 = __riscv_vand_vx_u8mf2(v125, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v127 = __riscv_vreinterpret_v_u8mf2_i8mf2(v126);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v128 = __riscv_vand_vx_u8mf2(v113, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v129 = __riscv_vmseq_vx_u8mf2_b16(v128, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v130 = __riscv_vadd_vx_i8mf2_mu(v129, v127, v127, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v131 = __riscv_vsrl_vx_u8mf2(v110, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v132 = __riscv_vand_vx_u8mf2(v131, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v133 = __riscv_vreinterpret_v_u8mf2_i8mf2(v132);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v134 = __riscv_vand_vx_u8mf2(v113, 8, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v135 = __riscv_vmseq_vx_u8mf2_b16(v134, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v136 = __riscv_vadd_vx_i8mf2_mu(v135, v133, v133, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v137 = v107 + 16;
          const int8_t* v138 = (const int8_t*) v137;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v139 = *(const int8_t *)(v138);
          vint16m1_t v140 = v71;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v141 = __riscv_vwmacc_vx_i16m1(v140, v139, v118, 8);
          v71 = v141;
          const uint8_t* v142 = v107 + 144;
          const int8_t* v143 = (const int8_t*) v142;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v144 = *(const int8_t *)(v143);
          vint16m1_t v145 = v73;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v146 = __riscv_vwmacc_vx_i16m1(v145, v144, v124, 8);
          v73 = v146;
          const uint8_t* v147 = v107 + 272;
          const int8_t* v148 = (const int8_t*) v147;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v149 = *(const int8_t *)(v148);
          vint16m1_t v150 = v75;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v151 = __riscv_vwmacc_vx_i16m1(v150, v149, v130, 8);
          v75 = v151;
          const uint8_t* v152 = v107 + 400;
          const int8_t* v153 = (const int8_t*) v152;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v154 = *(const int8_t *)(v153);
          vint16m1_t v155 = v77;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v156 = __riscv_vwmacc_vx_i16m1(v155, v154, v136, 8);
          v77 = v156;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v157 = v107 + 17;
          const int8_t* v158 = (const int8_t*) v157;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v159 = *(const int8_t *)(v158);
          vint16m1_t v160 = v79;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v161 = __riscv_vwmacc_vx_i16m1(v160, v159, v118, 8);
          v79 = v161;
          const uint8_t* v162 = v107 + 145;
          const int8_t* v163 = (const int8_t*) v162;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v164 = *(const int8_t *)(v163);
          vint16m1_t v165 = v81;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v166 = __riscv_vwmacc_vx_i16m1(v165, v164, v124, 8);
          v81 = v166;
          const uint8_t* v167 = v107 + 273;
          const int8_t* v168 = (const int8_t*) v167;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v169 = *(const int8_t *)(v168);
          vint16m1_t v170 = v83;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v171 = __riscv_vwmacc_vx_i16m1(v170, v169, v130, 8);
          v83 = v171;
          const uint8_t* v172 = v107 + 401;
          const int8_t* v173 = (const int8_t*) v172;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v174 = *(const int8_t *)(v173);
          vint16m1_t v175 = v85;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v176 = __riscv_vwmacc_vx_i16m1(v175, v174, v136, 8);
          v85 = v176;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v177 = v107 + 18;
          const int8_t* v178 = (const int8_t*) v177;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v179 = *(const int8_t *)(v178);
          vint16m1_t v180 = v87;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v181 = __riscv_vwmacc_vx_i16m1(v180, v179, v118, 8);
          v87 = v181;
          const uint8_t* v182 = v107 + 146;
          const int8_t* v183 = (const int8_t*) v182;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v184 = *(const int8_t *)(v183);
          vint16m1_t v185 = v89;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v186 = __riscv_vwmacc_vx_i16m1(v185, v184, v124, 8);
          v89 = v186;
          const uint8_t* v187 = v107 + 274;
          const int8_t* v188 = (const int8_t*) v187;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v189 = *(const int8_t *)(v188);
          vint16m1_t v190 = v91;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v191 = __riscv_vwmacc_vx_i16m1(v190, v189, v130, 8);
          v91 = v191;
          const uint8_t* v192 = v107 + 402;
          const int8_t* v193 = (const int8_t*) v192;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v194 = *(const int8_t *)(v193);
          vint16m1_t v195 = v93;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v196 = __riscv_vwmacc_vx_i16m1(v195, v194, v136, 8);
          v93 = v196;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v197 = v107 + 19;
          const int8_t* v198 = (const int8_t*) v197;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v199 = *(const int8_t *)(v198);
          vint16m1_t v200 = v95;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v201 = __riscv_vwmacc_vx_i16m1(v200, v199, v118, 8);
          v95 = v201;
          const uint8_t* v202 = v107 + 147;
          const int8_t* v203 = (const int8_t*) v202;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v204 = *(const int8_t *)(v203);
          vint16m1_t v205 = v97;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v206 = __riscv_vwmacc_vx_i16m1(v205, v204, v124, 8);
          v97 = v206;
          const uint8_t* v207 = v107 + 275;
          const int8_t* v208 = (const int8_t*) v207;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v209 = *(const int8_t *)(v208);
          vint16m1_t v210 = v99;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v211 = __riscv_vwmacc_vx_i16m1(v210, v209, v130, 8);
          v99 = v211;
          const uint8_t* v212 = v107 + 403;
          const int8_t* v213 = (const int8_t*) v212;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v214 = *(const int8_t *)(v213);
          vint16m1_t v215 = v101;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v216 = __riscv_vwmacc_vx_i16m1(v215, v214, v136, 8);
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
        vint32m2_t v234 = __riscv_vwmacc_vv_i32m2(v233, v58, v217, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v235 = __riscv_vwmacc_vv_i32m2(v234, v62, v218, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v236 = __riscv_vwmacc_vv_i32m2(v235, v66, v219, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v237 = __riscv_vwmacc_vv_i32m2(v236, v70, v220, 8);
        v47 = v237;
        vint32m2_t v238 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v239 = __riscv_vwmacc_vv_i32m2(v238, v58, v221, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v240 = __riscv_vwmacc_vv_i32m2(v239, v62, v222, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v241 = __riscv_vwmacc_vv_i32m2(v240, v66, v223, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v242 = __riscv_vwmacc_vv_i32m2(v241, v70, v224, 8);
        v49 = v242;
        vint32m2_t v243 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v244 = __riscv_vwmacc_vv_i32m2(v243, v58, v225, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v245 = __riscv_vwmacc_vv_i32m2(v244, v62, v226, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v246 = __riscv_vwmacc_vv_i32m2(v245, v66, v227, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v247 = __riscv_vwmacc_vv_i32m2(v246, v70, v228, 8);
        v51 = v247;
        vint32m2_t v248 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v249 = __riscv_vwmacc_vv_i32m2(v248, v58, v229, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v250 = __riscv_vwmacc_vv_i32m2(v249, v62, v230, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v251 = __riscv_vwmacc_vv_i32m2(v250, v66, v231, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v252 = __riscv_vwmacc_vv_i32m2(v251, v70, v232, 8);
        v53 = v252;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v253 = v30 + 48;
        const int8_t* v254 = (const int8_t*) v253;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v255 = __riscv_vle8_v_i8mf2(v254, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v256 = __riscv_vsext_vf2_i16m1(v255, 8);
        const uint8_t* v257 = v30 + 80;
        const int8_t* v258 = (const int8_t*) v257;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v259 = __riscv_vle8_v_i8mf2(v258, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v260 = __riscv_vsext_vf2_i16m1(v259, 8);
        const uint8_t* v261 = v30 + 112;
        const int8_t* v262 = (const int8_t*) v261;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v263 = __riscv_vle8_v_i8mf2(v262, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v264 = __riscv_vsext_vf2_i16m1(v263, 8);
        const uint8_t* v265 = v30 + 144;
        const int8_t* v266 = (const int8_t*) v265;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v267 = __riscv_vle8_v_i8mf2(v266, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v268 = __riscv_vsext_vf2_i16m1(v267, 8);
        vint16m1_t v269;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v270 = __riscv_vmv_v_x_i16m1(0, 8);
        v269 = v270;
        vint16m1_t v271;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v272 = __riscv_vmv_v_x_i16m1(0, 8);
        v271 = v272;
        vint16m1_t v273;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v274 = __riscv_vmv_v_x_i16m1(0, 8);
        v273 = v274;
        vint16m1_t v275;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v276 = __riscv_vmv_v_x_i16m1(0, 8);
        v275 = v276;
        vint16m1_t v277;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v278 = __riscv_vmv_v_x_i16m1(0, 8);
        v277 = v278;
        vint16m1_t v279;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v280 = __riscv_vmv_v_x_i16m1(0, 8);
        v279 = v280;
        vint16m1_t v281;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v282 = __riscv_vmv_v_x_i16m1(0, 8);
        v281 = v282;
        vint16m1_t v283;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v284 = __riscv_vmv_v_x_i16m1(0, 8);
        v283 = v284;
        vint16m1_t v285;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v286 = __riscv_vmv_v_x_i16m1(0, 8);
        v285 = v286;
        vint16m1_t v287;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v288 = __riscv_vmv_v_x_i16m1(0, 8);
        v287 = v288;
        vint16m1_t v289;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v290 = __riscv_vmv_v_x_i16m1(0, 8);
        v289 = v290;
        vint16m1_t v291;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v292 = __riscv_vmv_v_x_i16m1(0, 8);
        v291 = v292;
        vint16m1_t v293;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v294 = __riscv_vmv_v_x_i16m1(0, 8);
        v293 = v294;
        vint16m1_t v295;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v296 = __riscv_vmv_v_x_i16m1(0, 8);
        v295 = v296;
        vint16m1_t v297;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v298 = __riscv_vmv_v_x_i16m1(0, 8);
        v297 = v298;
        vint16m1_t v299;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v300 = __riscv_vmv_v_x_i16m1(0, 8);
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
          vuint8mf2_t v308 = __riscv_vle8_v_u8mf2(v307, 8);
          const uint8_t* v309 = v303 + 544;
          const uint8_t* v310 = (const uint8_t*) v309;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v311 = __riscv_vle8_v_u8mf2(v310, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v312 = __riscv_vand_vx_u8mf2(v308, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v313 = __riscv_vreinterpret_v_u8mf2_i8mf2(v312);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v314 = __riscv_vand_vx_u8mf2(v311, 1, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v315 = __riscv_vmseq_vx_u8mf2_b16(v314, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v316 = __riscv_vadd_vx_i8mf2_mu(v315, v313, v313, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v317 = __riscv_vsrl_vx_u8mf2(v308, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v318 = __riscv_vand_vx_u8mf2(v317, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v319 = __riscv_vreinterpret_v_u8mf2_i8mf2(v318);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v320 = __riscv_vand_vx_u8mf2(v311, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v321 = __riscv_vmseq_vx_u8mf2_b16(v320, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v322 = __riscv_vadd_vx_i8mf2_mu(v321, v319, v319, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v323 = __riscv_vsrl_vx_u8mf2(v308, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v324 = __riscv_vand_vx_u8mf2(v323, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v325 = __riscv_vreinterpret_v_u8mf2_i8mf2(v324);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v326 = __riscv_vand_vx_u8mf2(v311, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v327 = __riscv_vmseq_vx_u8mf2_b16(v326, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v328 = __riscv_vadd_vx_i8mf2_mu(v327, v325, v325, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v329 = __riscv_vsrl_vx_u8mf2(v308, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v330 = __riscv_vand_vx_u8mf2(v329, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v331 = __riscv_vreinterpret_v_u8mf2_i8mf2(v330);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v332 = __riscv_vand_vx_u8mf2(v311, 8, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v333 = __riscv_vmseq_vx_u8mf2_b16(v332, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v334 = __riscv_vadd_vx_i8mf2_mu(v333, v331, v331, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v335 = v305 + 80;
          const int8_t* v336 = (const int8_t*) v335;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v337 = *(const int8_t *)(v336);
          vint16m1_t v338 = v269;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v339 = __riscv_vwmacc_vx_i16m1(v338, v337, v316, 8);
          v269 = v339;
          const uint8_t* v340 = v305 + 208;
          const int8_t* v341 = (const int8_t*) v340;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v342 = *(const int8_t *)(v341);
          vint16m1_t v343 = v271;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v344 = __riscv_vwmacc_vx_i16m1(v343, v342, v322, 8);
          v271 = v344;
          const uint8_t* v345 = v305 + 336;
          const int8_t* v346 = (const int8_t*) v345;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v347 = *(const int8_t *)(v346);
          vint16m1_t v348 = v273;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v349 = __riscv_vwmacc_vx_i16m1(v348, v347, v328, 8);
          v273 = v349;
          const uint8_t* v350 = v305 + 464;
          const int8_t* v351 = (const int8_t*) v350;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v352 = *(const int8_t *)(v351);
          vint16m1_t v353 = v275;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v354 = __riscv_vwmacc_vx_i16m1(v353, v352, v334, 8);
          v275 = v354;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v355 = v305 + 81;
          const int8_t* v356 = (const int8_t*) v355;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v357 = *(const int8_t *)(v356);
          vint16m1_t v358 = v277;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v359 = __riscv_vwmacc_vx_i16m1(v358, v357, v316, 8);
          v277 = v359;
          const uint8_t* v360 = v305 + 209;
          const int8_t* v361 = (const int8_t*) v360;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v362 = *(const int8_t *)(v361);
          vint16m1_t v363 = v279;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v364 = __riscv_vwmacc_vx_i16m1(v363, v362, v322, 8);
          v279 = v364;
          const uint8_t* v365 = v305 + 337;
          const int8_t* v366 = (const int8_t*) v365;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v367 = *(const int8_t *)(v366);
          vint16m1_t v368 = v281;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v369 = __riscv_vwmacc_vx_i16m1(v368, v367, v328, 8);
          v281 = v369;
          const uint8_t* v370 = v305 + 465;
          const int8_t* v371 = (const int8_t*) v370;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v372 = *(const int8_t *)(v371);
          vint16m1_t v373 = v283;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v374 = __riscv_vwmacc_vx_i16m1(v373, v372, v334, 8);
          v283 = v374;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v375 = v305 + 82;
          const int8_t* v376 = (const int8_t*) v375;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v377 = *(const int8_t *)(v376);
          vint16m1_t v378 = v285;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v379 = __riscv_vwmacc_vx_i16m1(v378, v377, v316, 8);
          v285 = v379;
          const uint8_t* v380 = v305 + 210;
          const int8_t* v381 = (const int8_t*) v380;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v382 = *(const int8_t *)(v381);
          vint16m1_t v383 = v287;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v384 = __riscv_vwmacc_vx_i16m1(v383, v382, v322, 8);
          v287 = v384;
          const uint8_t* v385 = v305 + 338;
          const int8_t* v386 = (const int8_t*) v385;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v387 = *(const int8_t *)(v386);
          vint16m1_t v388 = v289;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v389 = __riscv_vwmacc_vx_i16m1(v388, v387, v328, 8);
          v289 = v389;
          const uint8_t* v390 = v305 + 466;
          const int8_t* v391 = (const int8_t*) v390;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v392 = *(const int8_t *)(v391);
          vint16m1_t v393 = v291;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v394 = __riscv_vwmacc_vx_i16m1(v393, v392, v334, 8);
          v291 = v394;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v395 = v305 + 83;
          const int8_t* v396 = (const int8_t*) v395;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v397 = *(const int8_t *)(v396);
          vint16m1_t v398 = v293;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v399 = __riscv_vwmacc_vx_i16m1(v398, v397, v316, 8);
          v293 = v399;
          const uint8_t* v400 = v305 + 211;
          const int8_t* v401 = (const int8_t*) v400;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v402 = *(const int8_t *)(v401);
          vint16m1_t v403 = v295;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v404 = __riscv_vwmacc_vx_i16m1(v403, v402, v322, 8);
          v295 = v404;
          const uint8_t* v405 = v305 + 339;
          const int8_t* v406 = (const int8_t*) v405;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v407 = *(const int8_t *)(v406);
          vint16m1_t v408 = v297;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v409 = __riscv_vwmacc_vx_i16m1(v408, v407, v328, 8);
          v297 = v409;
          const uint8_t* v410 = v305 + 467;
          const int8_t* v411 = (const int8_t*) v410;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v412 = *(const int8_t *)(v411);
          vint16m1_t v413 = v299;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v414 = __riscv_vwmacc_vx_i16m1(v413, v412, v334, 8);
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
        vint32m2_t v432 = __riscv_vwmacc_vv_i32m2(v431, v256, v415, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v433 = __riscv_vwmacc_vv_i32m2(v432, v260, v416, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v434 = __riscv_vwmacc_vv_i32m2(v433, v264, v417, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v435 = __riscv_vwmacc_vv_i32m2(v434, v268, v418, 8);
        v47 = v435;
        vint32m2_t v436 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v437 = __riscv_vwmacc_vv_i32m2(v436, v256, v419, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v438 = __riscv_vwmacc_vv_i32m2(v437, v260, v420, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v439 = __riscv_vwmacc_vv_i32m2(v438, v264, v421, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v440 = __riscv_vwmacc_vv_i32m2(v439, v268, v422, 8);
        v49 = v440;
        vint32m2_t v441 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v442 = __riscv_vwmacc_vv_i32m2(v441, v256, v423, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v443 = __riscv_vwmacc_vv_i32m2(v442, v260, v424, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v444 = __riscv_vwmacc_vv_i32m2(v443, v264, v425, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v445 = __riscv_vwmacc_vv_i32m2(v444, v268, v426, 8);
        v51 = v445;
        vint32m2_t v446 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v447 = __riscv_vwmacc_vv_i32m2(v446, v256, v427, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v448 = __riscv_vwmacc_vv_i32m2(v447, v260, v428, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v449 = __riscv_vwmacc_vv_i32m2(v448, v264, v429, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v450 = __riscv_vwmacc_vv_i32m2(v449, v268, v430, 8);
        v53 = v450;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v451 = v30 + 160;
        const int8_t* v452 = (const int8_t*) v451;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v453 = __riscv_vle8_v_i8mf2(v452, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v454 = __riscv_vsext_vf2_i16m1(v453, 8);
        const uint8_t* v455 = v30 + 192;
        const int8_t* v456 = (const int8_t*) v455;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v457 = __riscv_vle8_v_i8mf2(v456, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v458 = __riscv_vsext_vf2_i16m1(v457, 8);
        const uint8_t* v459 = v30 + 224;
        const int8_t* v460 = (const int8_t*) v459;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v461 = __riscv_vle8_v_i8mf2(v460, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v462 = __riscv_vsext_vf2_i16m1(v461, 8);
        const uint8_t* v463 = v30 + 256;
        const int8_t* v464 = (const int8_t*) v463;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v465 = __riscv_vle8_v_i8mf2(v464, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v466 = __riscv_vsext_vf2_i16m1(v465, 8);
        vint16m1_t v467;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v468 = __riscv_vmv_v_x_i16m1(0, 8);
        v467 = v468;
        vint16m1_t v469;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v470 = __riscv_vmv_v_x_i16m1(0, 8);
        v469 = v470;
        vint16m1_t v471;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v472 = __riscv_vmv_v_x_i16m1(0, 8);
        v471 = v472;
        vint16m1_t v473;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v474 = __riscv_vmv_v_x_i16m1(0, 8);
        v473 = v474;
        vint16m1_t v475;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v476 = __riscv_vmv_v_x_i16m1(0, 8);
        v475 = v476;
        vint16m1_t v477;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v478 = __riscv_vmv_v_x_i16m1(0, 8);
        v477 = v478;
        vint16m1_t v479;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v480 = __riscv_vmv_v_x_i16m1(0, 8);
        v479 = v480;
        vint16m1_t v481;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v482 = __riscv_vmv_v_x_i16m1(0, 8);
        v481 = v482;
        vint16m1_t v483;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v484 = __riscv_vmv_v_x_i16m1(0, 8);
        v483 = v484;
        vint16m1_t v485;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v486 = __riscv_vmv_v_x_i16m1(0, 8);
        v485 = v486;
        vint16m1_t v487;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v488 = __riscv_vmv_v_x_i16m1(0, 8);
        v487 = v488;
        vint16m1_t v489;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v490 = __riscv_vmv_v_x_i16m1(0, 8);
        v489 = v490;
        vint16m1_t v491;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v492 = __riscv_vmv_v_x_i16m1(0, 8);
        v491 = v492;
        vint16m1_t v493;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v494 = __riscv_vmv_v_x_i16m1(0, 8);
        v493 = v494;
        vint16m1_t v495;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v496 = __riscv_vmv_v_x_i16m1(0, 8);
        v495 = v496;
        vint16m1_t v497;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v498 = __riscv_vmv_v_x_i16m1(0, 8);
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
          vuint8mf2_t v506 = __riscv_vle8_v_u8mf2(v505, 8);
          const uint8_t* v507 = v501 + 288;
          const uint8_t* v508 = (const uint8_t*) v507;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v509 = __riscv_vle8_v_u8mf2(v508, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v510 = __riscv_vand_vx_u8mf2(v506, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v511 = __riscv_vreinterpret_v_u8mf2_i8mf2(v510);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v512 = __riscv_vand_vx_u8mf2(v509, 16, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v513 = __riscv_vmseq_vx_u8mf2_b16(v512, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v514 = __riscv_vadd_vx_i8mf2_mu(v513, v511, v511, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v515 = __riscv_vsrl_vx_u8mf2(v506, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v516 = __riscv_vand_vx_u8mf2(v515, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v517 = __riscv_vreinterpret_v_u8mf2_i8mf2(v516);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v518 = __riscv_vand_vx_u8mf2(v509, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v519 = __riscv_vmseq_vx_u8mf2_b16(v518, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v520 = __riscv_vadd_vx_i8mf2_mu(v519, v517, v517, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v521 = __riscv_vsrl_vx_u8mf2(v506, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v522 = __riscv_vand_vx_u8mf2(v521, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v523 = __riscv_vreinterpret_v_u8mf2_i8mf2(v522);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v524 = __riscv_vand_vx_u8mf2(v509, 64, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v525 = __riscv_vmseq_vx_u8mf2_b16(v524, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v526 = __riscv_vadd_vx_i8mf2_mu(v525, v523, v523, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v527 = __riscv_vsrl_vx_u8mf2(v506, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v528 = __riscv_vand_vx_u8mf2(v527, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v529 = __riscv_vreinterpret_v_u8mf2_i8mf2(v528);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v530 = __riscv_vand_vx_u8mf2(v509, 128, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v531 = __riscv_vmseq_vx_u8mf2_b16(v530, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v532 = __riscv_vadd_vx_i8mf2_mu(v531, v529, v529, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v533 = v503 + 528;
          const int8_t* v534 = (const int8_t*) v533;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v535 = *(const int8_t *)(v534);
          vint16m1_t v536 = v467;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v537 = __riscv_vwmacc_vx_i16m1(v536, v535, v514, 8);
          v467 = v537;
          const uint8_t* v538 = v503 + 656;
          const int8_t* v539 = (const int8_t*) v538;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v540 = *(const int8_t *)(v539);
          vint16m1_t v541 = v469;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v542 = __riscv_vwmacc_vx_i16m1(v541, v540, v520, 8);
          v469 = v542;
          const uint8_t* v543 = v503 + 784;
          const int8_t* v544 = (const int8_t*) v543;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v545 = *(const int8_t *)(v544);
          vint16m1_t v546 = v471;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v547 = __riscv_vwmacc_vx_i16m1(v546, v545, v526, 8);
          v471 = v547;
          const uint8_t* v548 = v503 + 912;
          const int8_t* v549 = (const int8_t*) v548;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v550 = *(const int8_t *)(v549);
          vint16m1_t v551 = v473;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v552 = __riscv_vwmacc_vx_i16m1(v551, v550, v532, 8);
          v473 = v552;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v553 = v503 + 529;
          const int8_t* v554 = (const int8_t*) v553;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v555 = *(const int8_t *)(v554);
          vint16m1_t v556 = v475;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v557 = __riscv_vwmacc_vx_i16m1(v556, v555, v514, 8);
          v475 = v557;
          const uint8_t* v558 = v503 + 657;
          const int8_t* v559 = (const int8_t*) v558;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v560 = *(const int8_t *)(v559);
          vint16m1_t v561 = v477;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v562 = __riscv_vwmacc_vx_i16m1(v561, v560, v520, 8);
          v477 = v562;
          const uint8_t* v563 = v503 + 785;
          const int8_t* v564 = (const int8_t*) v563;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v565 = *(const int8_t *)(v564);
          vint16m1_t v566 = v479;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v567 = __riscv_vwmacc_vx_i16m1(v566, v565, v526, 8);
          v479 = v567;
          const uint8_t* v568 = v503 + 913;
          const int8_t* v569 = (const int8_t*) v568;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v570 = *(const int8_t *)(v569);
          vint16m1_t v571 = v481;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v572 = __riscv_vwmacc_vx_i16m1(v571, v570, v532, 8);
          v481 = v572;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v573 = v503 + 530;
          const int8_t* v574 = (const int8_t*) v573;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v575 = *(const int8_t *)(v574);
          vint16m1_t v576 = v483;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v577 = __riscv_vwmacc_vx_i16m1(v576, v575, v514, 8);
          v483 = v577;
          const uint8_t* v578 = v503 + 658;
          const int8_t* v579 = (const int8_t*) v578;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v580 = *(const int8_t *)(v579);
          vint16m1_t v581 = v485;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v582 = __riscv_vwmacc_vx_i16m1(v581, v580, v520, 8);
          v485 = v582;
          const uint8_t* v583 = v503 + 786;
          const int8_t* v584 = (const int8_t*) v583;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v585 = *(const int8_t *)(v584);
          vint16m1_t v586 = v487;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v587 = __riscv_vwmacc_vx_i16m1(v586, v585, v526, 8);
          v487 = v587;
          const uint8_t* v588 = v503 + 914;
          const int8_t* v589 = (const int8_t*) v588;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v590 = *(const int8_t *)(v589);
          vint16m1_t v591 = v489;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v592 = __riscv_vwmacc_vx_i16m1(v591, v590, v532, 8);
          v489 = v592;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v593 = v503 + 531;
          const int8_t* v594 = (const int8_t*) v593;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v595 = *(const int8_t *)(v594);
          vint16m1_t v596 = v491;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v597 = __riscv_vwmacc_vx_i16m1(v596, v595, v514, 8);
          v491 = v597;
          const uint8_t* v598 = v503 + 659;
          const int8_t* v599 = (const int8_t*) v598;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v600 = *(const int8_t *)(v599);
          vint16m1_t v601 = v493;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v602 = __riscv_vwmacc_vx_i16m1(v601, v600, v520, 8);
          v493 = v602;
          const uint8_t* v603 = v503 + 787;
          const int8_t* v604 = (const int8_t*) v603;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v605 = *(const int8_t *)(v604);
          vint16m1_t v606 = v495;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v607 = __riscv_vwmacc_vx_i16m1(v606, v605, v526, 8);
          v495 = v607;
          const uint8_t* v608 = v503 + 915;
          const int8_t* v609 = (const int8_t*) v608;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v610 = *(const int8_t *)(v609);
          vint16m1_t v611 = v497;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v612 = __riscv_vwmacc_vx_i16m1(v611, v610, v532, 8);
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
        vint32m2_t v630 = __riscv_vwmacc_vv_i32m2(v629, v454, v613, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v631 = __riscv_vwmacc_vv_i32m2(v630, v458, v614, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v632 = __riscv_vwmacc_vv_i32m2(v631, v462, v615, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v633 = __riscv_vwmacc_vv_i32m2(v632, v466, v616, 8);
        v47 = v633;
        vint32m2_t v634 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v635 = __riscv_vwmacc_vv_i32m2(v634, v454, v617, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v636 = __riscv_vwmacc_vv_i32m2(v635, v458, v618, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v637 = __riscv_vwmacc_vv_i32m2(v636, v462, v619, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v638 = __riscv_vwmacc_vv_i32m2(v637, v466, v620, 8);
        v49 = v638;
        vint32m2_t v639 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v640 = __riscv_vwmacc_vv_i32m2(v639, v454, v621, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v641 = __riscv_vwmacc_vv_i32m2(v640, v458, v622, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v642 = __riscv_vwmacc_vv_i32m2(v641, v462, v623, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v643 = __riscv_vwmacc_vv_i32m2(v642, v466, v624, 8);
        v51 = v643;
        vint32m2_t v644 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v645 = __riscv_vwmacc_vv_i32m2(v644, v454, v625, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v646 = __riscv_vwmacc_vv_i32m2(v645, v458, v626, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v647 = __riscv_vwmacc_vv_i32m2(v646, v462, v627, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v648 = __riscv_vwmacc_vv_i32m2(v647, v466, v628, 8);
        v53 = v648;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v649 = v30 + 176;
        const int8_t* v650 = (const int8_t*) v649;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v651 = __riscv_vle8_v_i8mf2(v650, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v652 = __riscv_vsext_vf2_i16m1(v651, 8);
        const uint8_t* v653 = v30 + 208;
        const int8_t* v654 = (const int8_t*) v653;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v655 = __riscv_vle8_v_i8mf2(v654, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v656 = __riscv_vsext_vf2_i16m1(v655, 8);
        const uint8_t* v657 = v30 + 240;
        const int8_t* v658 = (const int8_t*) v657;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v659 = __riscv_vle8_v_i8mf2(v658, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v660 = __riscv_vsext_vf2_i16m1(v659, 8);
        const uint8_t* v661 = v30 + 272;
        const int8_t* v662 = (const int8_t*) v661;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v663 = __riscv_vle8_v_i8mf2(v662, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v664 = __riscv_vsext_vf2_i16m1(v663, 8);
        vint16m1_t v665;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v666 = __riscv_vmv_v_x_i16m1(0, 8);
        v665 = v666;
        vint16m1_t v667;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v668 = __riscv_vmv_v_x_i16m1(0, 8);
        v667 = v668;
        vint16m1_t v669;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v670 = __riscv_vmv_v_x_i16m1(0, 8);
        v669 = v670;
        vint16m1_t v671;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v672 = __riscv_vmv_v_x_i16m1(0, 8);
        v671 = v672;
        vint16m1_t v673;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v674 = __riscv_vmv_v_x_i16m1(0, 8);
        v673 = v674;
        vint16m1_t v675;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v676 = __riscv_vmv_v_x_i16m1(0, 8);
        v675 = v676;
        vint16m1_t v677;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v678 = __riscv_vmv_v_x_i16m1(0, 8);
        v677 = v678;
        vint16m1_t v679;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v680 = __riscv_vmv_v_x_i16m1(0, 8);
        v679 = v680;
        vint16m1_t v681;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v682 = __riscv_vmv_v_x_i16m1(0, 8);
        v681 = v682;
        vint16m1_t v683;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v684 = __riscv_vmv_v_x_i16m1(0, 8);
        v683 = v684;
        vint16m1_t v685;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v686 = __riscv_vmv_v_x_i16m1(0, 8);
        v685 = v686;
        vint16m1_t v687;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v688 = __riscv_vmv_v_x_i16m1(0, 8);
        v687 = v688;
        vint16m1_t v689;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v690 = __riscv_vmv_v_x_i16m1(0, 8);
        v689 = v690;
        vint16m1_t v691;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v692 = __riscv_vmv_v_x_i16m1(0, 8);
        v691 = v692;
        vint16m1_t v693;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v694 = __riscv_vmv_v_x_i16m1(0, 8);
        v693 = v694;
        vint16m1_t v695;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v696 = __riscv_vmv_v_x_i16m1(0, 8);
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
          vuint8mf2_t v704 = __riscv_vle8_v_u8mf2(v703, 8);
          const uint8_t* v705 = v699 + 544;
          const uint8_t* v706 = (const uint8_t*) v705;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v707 = __riscv_vle8_v_u8mf2(v706, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v708 = __riscv_vand_vx_u8mf2(v704, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v709 = __riscv_vreinterpret_v_u8mf2_i8mf2(v708);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v710 = __riscv_vand_vx_u8mf2(v707, 16, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v711 = __riscv_vmseq_vx_u8mf2_b16(v710, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v712 = __riscv_vadd_vx_i8mf2_mu(v711, v709, v709, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v713 = __riscv_vsrl_vx_u8mf2(v704, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v714 = __riscv_vand_vx_u8mf2(v713, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v715 = __riscv_vreinterpret_v_u8mf2_i8mf2(v714);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v716 = __riscv_vand_vx_u8mf2(v707, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v717 = __riscv_vmseq_vx_u8mf2_b16(v716, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v718 = __riscv_vadd_vx_i8mf2_mu(v717, v715, v715, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v719 = __riscv_vsrl_vx_u8mf2(v704, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v720 = __riscv_vand_vx_u8mf2(v719, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v721 = __riscv_vreinterpret_v_u8mf2_i8mf2(v720);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v722 = __riscv_vand_vx_u8mf2(v707, 64, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v723 = __riscv_vmseq_vx_u8mf2_b16(v722, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v724 = __riscv_vadd_vx_i8mf2_mu(v723, v721, v721, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v725 = __riscv_vsrl_vx_u8mf2(v704, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v726 = __riscv_vand_vx_u8mf2(v725, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v727 = __riscv_vreinterpret_v_u8mf2_i8mf2(v726);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v728 = __riscv_vand_vx_u8mf2(v707, 128, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v729 = __riscv_vmseq_vx_u8mf2_b16(v728, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v730 = __riscv_vadd_vx_i8mf2_mu(v729, v727, v727, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v731 = v701 + 592;
          const int8_t* v732 = (const int8_t*) v731;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v733 = *(const int8_t *)(v732);
          vint16m1_t v734 = v665;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v735 = __riscv_vwmacc_vx_i16m1(v734, v733, v712, 8);
          v665 = v735;
          const uint8_t* v736 = v701 + 720;
          const int8_t* v737 = (const int8_t*) v736;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v738 = *(const int8_t *)(v737);
          vint16m1_t v739 = v667;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v740 = __riscv_vwmacc_vx_i16m1(v739, v738, v718, 8);
          v667 = v740;
          const uint8_t* v741 = v701 + 848;
          const int8_t* v742 = (const int8_t*) v741;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v743 = *(const int8_t *)(v742);
          vint16m1_t v744 = v669;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v745 = __riscv_vwmacc_vx_i16m1(v744, v743, v724, 8);
          v669 = v745;
          const uint8_t* v746 = v701 + 976;
          const int8_t* v747 = (const int8_t*) v746;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v748 = *(const int8_t *)(v747);
          vint16m1_t v749 = v671;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v750 = __riscv_vwmacc_vx_i16m1(v749, v748, v730, 8);
          v671 = v750;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v751 = v701 + 593;
          const int8_t* v752 = (const int8_t*) v751;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v753 = *(const int8_t *)(v752);
          vint16m1_t v754 = v673;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v755 = __riscv_vwmacc_vx_i16m1(v754, v753, v712, 8);
          v673 = v755;
          const uint8_t* v756 = v701 + 721;
          const int8_t* v757 = (const int8_t*) v756;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v758 = *(const int8_t *)(v757);
          vint16m1_t v759 = v675;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v760 = __riscv_vwmacc_vx_i16m1(v759, v758, v718, 8);
          v675 = v760;
          const uint8_t* v761 = v701 + 849;
          const int8_t* v762 = (const int8_t*) v761;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v763 = *(const int8_t *)(v762);
          vint16m1_t v764 = v677;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v765 = __riscv_vwmacc_vx_i16m1(v764, v763, v724, 8);
          v677 = v765;
          const uint8_t* v766 = v701 + 977;
          const int8_t* v767 = (const int8_t*) v766;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v768 = *(const int8_t *)(v767);
          vint16m1_t v769 = v679;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v770 = __riscv_vwmacc_vx_i16m1(v769, v768, v730, 8);
          v679 = v770;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v771 = v701 + 594;
          const int8_t* v772 = (const int8_t*) v771;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v773 = *(const int8_t *)(v772);
          vint16m1_t v774 = v681;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v775 = __riscv_vwmacc_vx_i16m1(v774, v773, v712, 8);
          v681 = v775;
          const uint8_t* v776 = v701 + 722;
          const int8_t* v777 = (const int8_t*) v776;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v778 = *(const int8_t *)(v777);
          vint16m1_t v779 = v683;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v780 = __riscv_vwmacc_vx_i16m1(v779, v778, v718, 8);
          v683 = v780;
          const uint8_t* v781 = v701 + 850;
          const int8_t* v782 = (const int8_t*) v781;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v783 = *(const int8_t *)(v782);
          vint16m1_t v784 = v685;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v785 = __riscv_vwmacc_vx_i16m1(v784, v783, v724, 8);
          v685 = v785;
          const uint8_t* v786 = v701 + 978;
          const int8_t* v787 = (const int8_t*) v786;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v788 = *(const int8_t *)(v787);
          vint16m1_t v789 = v687;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v790 = __riscv_vwmacc_vx_i16m1(v789, v788, v730, 8);
          v687 = v790;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v791 = v701 + 595;
          const int8_t* v792 = (const int8_t*) v791;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v793 = *(const int8_t *)(v792);
          vint16m1_t v794 = v689;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v795 = __riscv_vwmacc_vx_i16m1(v794, v793, v712, 8);
          v689 = v795;
          const uint8_t* v796 = v701 + 723;
          const int8_t* v797 = (const int8_t*) v796;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v798 = *(const int8_t *)(v797);
          vint16m1_t v799 = v691;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v800 = __riscv_vwmacc_vx_i16m1(v799, v798, v718, 8);
          v691 = v800;
          const uint8_t* v801 = v701 + 851;
          const int8_t* v802 = (const int8_t*) v801;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v803 = *(const int8_t *)(v802);
          vint16m1_t v804 = v693;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v805 = __riscv_vwmacc_vx_i16m1(v804, v803, v724, 8);
          v693 = v805;
          const uint8_t* v806 = v701 + 979;
          const int8_t* v807 = (const int8_t*) v806;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v808 = *(const int8_t *)(v807);
          vint16m1_t v809 = v695;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v810 = __riscv_vwmacc_vx_i16m1(v809, v808, v730, 8);
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
        vint32m2_t v828 = __riscv_vwmacc_vv_i32m2(v827, v652, v811, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v829 = __riscv_vwmacc_vv_i32m2(v828, v656, v812, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v830 = __riscv_vwmacc_vv_i32m2(v829, v660, v813, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v831 = __riscv_vwmacc_vv_i32m2(v830, v664, v814, 8);
        v47 = v831;
        vint32m2_t v832 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v833 = __riscv_vwmacc_vv_i32m2(v832, v652, v815, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v834 = __riscv_vwmacc_vv_i32m2(v833, v656, v816, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v835 = __riscv_vwmacc_vv_i32m2(v834, v660, v817, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v836 = __riscv_vwmacc_vv_i32m2(v835, v664, v818, 8);
        v49 = v836;
        vint32m2_t v837 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v838 = __riscv_vwmacc_vv_i32m2(v837, v652, v819, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v839 = __riscv_vwmacc_vv_i32m2(v838, v656, v820, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v840 = __riscv_vwmacc_vv_i32m2(v839, v660, v821, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v841 = __riscv_vwmacc_vv_i32m2(v840, v664, v822, 8);
        v51 = v841;
        vint32m2_t v842 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v843 = __riscv_vwmacc_vv_i32m2(v842, v652, v823, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v844 = __riscv_vwmacc_vv_i32m2(v843, v656, v824, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v845 = __riscv_vwmacc_vv_i32m2(v844, v660, v825, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v846 = __riscv_vwmacc_vv_i32m2(v845, v664, v826, 8);
        v53 = v846;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v847 = __riscv_vfmul_vf_f32m2(v46, v34, 8);
        vint32m2_t v848 = v47;
        vfloat32m2_t v849 = v20;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v850 = __riscv_vfcvt_f_x_v_f32m2(v848, 8);
        vfloat32m2_t v851 = __riscv_vfmacc_vv_f32m2(v849, v850, v847, 8);
        v20 = v851;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v852 = __riscv_vfmul_vf_f32m2(v46, v37, 8);
        vint32m2_t v853 = v49;
        vfloat32m2_t v854 = v22;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v855 = __riscv_vfcvt_f_x_v_f32m2(v853, 8);
        vfloat32m2_t v856 = __riscv_vfmacc_vv_f32m2(v854, v855, v852, 8);
        v22 = v856;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v857 = __riscv_vfmul_vf_f32m2(v46, v40, 8);
        vint32m2_t v858 = v51;
        vfloat32m2_t v859 = v24;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v860 = __riscv_vfcvt_f_x_v_f32m2(v858, 8);
        vfloat32m2_t v861 = __riscv_vfmacc_vv_f32m2(v859, v860, v857, 8);
        v24 = v861;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v862 = __riscv_vfmul_vf_f32m2(v46, v43, 8);
        vint32m2_t v863 = v53;
        vfloat32m2_t v864 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v865 = __riscv_vfcvt_f_x_v_f32m2(v863, 8);
        vfloat32m2_t v866 = __riscv_vfmacc_vv_f32m2(v864, v865, v862, 8);
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
      __riscv_vse32_v_f32m2(v872, v873, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v874 = v12 * 4;
      size_t v875 = v874 + 1;
      size_t v876 = v875 * v7;
      size_t v877 = v16 * 16;
      size_t v878 = v876 + v877;
      float* v879 = v2 + v878;
      vfloat32m2_t v880 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v879, v880, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v881 = v12 * 4;
      size_t v882 = v881 + 2;
      size_t v883 = v882 * v7;
      size_t v884 = v16 * 16;
      size_t v885 = v883 + v884;
      float* v886 = v2 + v885;
      vfloat32m2_t v887 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v886, v887, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v888 = v12 * 4;
      size_t v889 = v888 + 3;
      size_t v890 = v889 * v7;
      size_t v891 = v16 * 16;
      size_t v892 = v890 + v891;
      float* v893 = v2 + v892;
      vfloat32m2_t v894 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v893, v894, 8);
      vfloat32m2_t v895;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v896 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
      v895 = v896;
      vfloat32m2_t v897;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v898 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
      v897 = v898;
      vfloat32m2_t v899;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v900 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
      v899 = v900;
      vfloat32m2_t v901;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
      vfloat32m2_t v902 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
      v901 = v902;
      for (size_t v903 = 0; v903 < v9; v903 += 1) {
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
        size_t v904 = v903 * 1824;
        const uint8_t* v905 = v19 + v904;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
        size_t v906 = v903 * 1168;
        const uint8_t* v907 = v15 + v906;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const float* v908 = (const float*) v907;
        float v909 = *(const float *)(v908);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v910 = v907 + 4;
        const float* v911 = (const float*) v910;
        float v912 = *(const float *)(v911);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v913 = v907 + 8;
        const float* v914 = (const float*) v913;
        float v915 = *(const float *)(v914);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v916 = v907 + 12;
        const float* v917 = (const float*) v916;
        float v918 = *(const float *)(v917);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const uint8_t* v919 = v905 + 16;
        const _Float16* v920 = (const _Float16*) v919;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v921 = __riscv_vle16_v_f16m1(v920, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v922 = __riscv_vfwcvt_f_f_v_f32m2(v921, 8);
        vint32m2_t v923;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v924 = __riscv_vmv_v_x_i32m2(0, 8);
        v923 = v924;
        vint32m2_t v925;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v926 = __riscv_vmv_v_x_i32m2(0, 8);
        v925 = v926;
        vint32m2_t v927;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v928 = __riscv_vmv_v_x_i32m2(0, 8);
        v927 = v928;
        vint32m2_t v929;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v930 = __riscv_vmv_v_x_i32m2(0, 8);
        v929 = v930;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v931 = v905 + 40;
        const int8_t* v932 = (const int8_t*) v931;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v933 = __riscv_vle8_v_i8mf2(v932, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v934 = __riscv_vsext_vf2_i16m1(v933, 8);
        const uint8_t* v935 = v905 + 72;
        const int8_t* v936 = (const int8_t*) v935;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v937 = __riscv_vle8_v_i8mf2(v936, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v938 = __riscv_vsext_vf2_i16m1(v937, 8);
        const uint8_t* v939 = v905 + 104;
        const int8_t* v940 = (const int8_t*) v939;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v941 = __riscv_vle8_v_i8mf2(v940, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v942 = __riscv_vsext_vf2_i16m1(v941, 8);
        const uint8_t* v943 = v905 + 136;
        const int8_t* v944 = (const int8_t*) v943;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v945 = __riscv_vle8_v_i8mf2(v944, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v946 = __riscv_vsext_vf2_i16m1(v945, 8);
        vint16m1_t v947;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v948 = __riscv_vmv_v_x_i16m1(0, 8);
        v947 = v948;
        vint16m1_t v949;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v950 = __riscv_vmv_v_x_i16m1(0, 8);
        v949 = v950;
        vint16m1_t v951;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v952 = __riscv_vmv_v_x_i16m1(0, 8);
        v951 = v952;
        vint16m1_t v953;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v954 = __riscv_vmv_v_x_i16m1(0, 8);
        v953 = v954;
        vint16m1_t v955;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v956 = __riscv_vmv_v_x_i16m1(0, 8);
        v955 = v956;
        vint16m1_t v957;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v958 = __riscv_vmv_v_x_i16m1(0, 8);
        v957 = v958;
        vint16m1_t v959;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v960 = __riscv_vmv_v_x_i16m1(0, 8);
        v959 = v960;
        vint16m1_t v961;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v962 = __riscv_vmv_v_x_i16m1(0, 8);
        v961 = v962;
        vint16m1_t v963;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v964 = __riscv_vmv_v_x_i16m1(0, 8);
        v963 = v964;
        vint16m1_t v965;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v966 = __riscv_vmv_v_x_i16m1(0, 8);
        v965 = v966;
        vint16m1_t v967;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v968 = __riscv_vmv_v_x_i16m1(0, 8);
        v967 = v968;
        vint16m1_t v969;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v970 = __riscv_vmv_v_x_i16m1(0, 8);
        v969 = v970;
        vint16m1_t v971;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v972 = __riscv_vmv_v_x_i16m1(0, 8);
        v971 = v972;
        vint16m1_t v973;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v974 = __riscv_vmv_v_x_i16m1(0, 8);
        v973 = v974;
        vint16m1_t v975;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v976 = __riscv_vmv_v_x_i16m1(0, 8);
        v975 = v976;
        vint16m1_t v977;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v978 = __riscv_vmv_v_x_i16m1(0, 8);
        v977 = v978;
        for (size_t v979 = 0; v979 < 16; v979 += 1) {
          size_t v980 = v979 * 16;
          const uint8_t* v981 = v905 + v980;
          size_t v982 = v979 * 4;
          const uint8_t* v983 = v907 + v982;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
          const uint8_t* v984 = v981 + 808;
          const uint8_t* v985 = (const uint8_t*) v984;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v986 = __riscv_vle8_v_u8mf2(v985, 8);
          const uint8_t* v987 = v981 + 296;
          const uint8_t* v988 = (const uint8_t*) v987;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v989 = __riscv_vle8_v_u8mf2(v988, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v990 = __riscv_vand_vx_u8mf2(v986, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v991 = __riscv_vreinterpret_v_u8mf2_i8mf2(v990);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v992 = __riscv_vand_vx_u8mf2(v989, 1, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v993 = __riscv_vmseq_vx_u8mf2_b16(v992, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v994 = __riscv_vadd_vx_i8mf2_mu(v993, v991, v991, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v995 = __riscv_vsrl_vx_u8mf2(v986, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v996 = __riscv_vand_vx_u8mf2(v995, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v997 = __riscv_vreinterpret_v_u8mf2_i8mf2(v996);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v998 = __riscv_vand_vx_u8mf2(v989, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v999 = __riscv_vmseq_vx_u8mf2_b16(v998, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1000 = __riscv_vadd_vx_i8mf2_mu(v999, v997, v997, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1001 = __riscv_vsrl_vx_u8mf2(v986, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1002 = __riscv_vand_vx_u8mf2(v1001, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1003 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1002);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1004 = __riscv_vand_vx_u8mf2(v989, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1005 = __riscv_vmseq_vx_u8mf2_b16(v1004, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1006 = __riscv_vadd_vx_i8mf2_mu(v1005, v1003, v1003, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1007 = __riscv_vsrl_vx_u8mf2(v986, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1008 = __riscv_vand_vx_u8mf2(v1007, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1009 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1008);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1010 = __riscv_vand_vx_u8mf2(v989, 8, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1011 = __riscv_vmseq_vx_u8mf2_b16(v1010, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1012 = __riscv_vadd_vx_i8mf2_mu(v1011, v1009, v1009, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1013 = v983 + 16;
          const int8_t* v1014 = (const int8_t*) v1013;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1015 = *(const int8_t *)(v1014);
          vint16m1_t v1016 = v947;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1017 = __riscv_vwmacc_vx_i16m1(v1016, v1015, v994, 8);
          v947 = v1017;
          const uint8_t* v1018 = v983 + 144;
          const int8_t* v1019 = (const int8_t*) v1018;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1020 = *(const int8_t *)(v1019);
          vint16m1_t v1021 = v949;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1022 = __riscv_vwmacc_vx_i16m1(v1021, v1020, v1000, 8);
          v949 = v1022;
          const uint8_t* v1023 = v983 + 272;
          const int8_t* v1024 = (const int8_t*) v1023;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1025 = *(const int8_t *)(v1024);
          vint16m1_t v1026 = v951;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1027 = __riscv_vwmacc_vx_i16m1(v1026, v1025, v1006, 8);
          v951 = v1027;
          const uint8_t* v1028 = v983 + 400;
          const int8_t* v1029 = (const int8_t*) v1028;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1030 = *(const int8_t *)(v1029);
          vint16m1_t v1031 = v953;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1032 = __riscv_vwmacc_vx_i16m1(v1031, v1030, v1012, 8);
          v953 = v1032;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1033 = v983 + 17;
          const int8_t* v1034 = (const int8_t*) v1033;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1035 = *(const int8_t *)(v1034);
          vint16m1_t v1036 = v955;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1037 = __riscv_vwmacc_vx_i16m1(v1036, v1035, v994, 8);
          v955 = v1037;
          const uint8_t* v1038 = v983 + 145;
          const int8_t* v1039 = (const int8_t*) v1038;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1040 = *(const int8_t *)(v1039);
          vint16m1_t v1041 = v957;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1042 = __riscv_vwmacc_vx_i16m1(v1041, v1040, v1000, 8);
          v957 = v1042;
          const uint8_t* v1043 = v983 + 273;
          const int8_t* v1044 = (const int8_t*) v1043;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1045 = *(const int8_t *)(v1044);
          vint16m1_t v1046 = v959;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1047 = __riscv_vwmacc_vx_i16m1(v1046, v1045, v1006, 8);
          v959 = v1047;
          const uint8_t* v1048 = v983 + 401;
          const int8_t* v1049 = (const int8_t*) v1048;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1050 = *(const int8_t *)(v1049);
          vint16m1_t v1051 = v961;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1052 = __riscv_vwmacc_vx_i16m1(v1051, v1050, v1012, 8);
          v961 = v1052;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1053 = v983 + 18;
          const int8_t* v1054 = (const int8_t*) v1053;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1055 = *(const int8_t *)(v1054);
          vint16m1_t v1056 = v963;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1057 = __riscv_vwmacc_vx_i16m1(v1056, v1055, v994, 8);
          v963 = v1057;
          const uint8_t* v1058 = v983 + 146;
          const int8_t* v1059 = (const int8_t*) v1058;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1060 = *(const int8_t *)(v1059);
          vint16m1_t v1061 = v965;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1062 = __riscv_vwmacc_vx_i16m1(v1061, v1060, v1000, 8);
          v965 = v1062;
          const uint8_t* v1063 = v983 + 274;
          const int8_t* v1064 = (const int8_t*) v1063;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1065 = *(const int8_t *)(v1064);
          vint16m1_t v1066 = v967;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1067 = __riscv_vwmacc_vx_i16m1(v1066, v1065, v1006, 8);
          v967 = v1067;
          const uint8_t* v1068 = v983 + 402;
          const int8_t* v1069 = (const int8_t*) v1068;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1070 = *(const int8_t *)(v1069);
          vint16m1_t v1071 = v969;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1072 = __riscv_vwmacc_vx_i16m1(v1071, v1070, v1012, 8);
          v969 = v1072;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1073 = v983 + 19;
          const int8_t* v1074 = (const int8_t*) v1073;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1075 = *(const int8_t *)(v1074);
          vint16m1_t v1076 = v971;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1077 = __riscv_vwmacc_vx_i16m1(v1076, v1075, v994, 8);
          v971 = v1077;
          const uint8_t* v1078 = v983 + 147;
          const int8_t* v1079 = (const int8_t*) v1078;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1080 = *(const int8_t *)(v1079);
          vint16m1_t v1081 = v973;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1082 = __riscv_vwmacc_vx_i16m1(v1081, v1080, v1000, 8);
          v973 = v1082;
          const uint8_t* v1083 = v983 + 275;
          const int8_t* v1084 = (const int8_t*) v1083;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1085 = *(const int8_t *)(v1084);
          vint16m1_t v1086 = v975;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1087 = __riscv_vwmacc_vx_i16m1(v1086, v1085, v1006, 8);
          v975 = v1087;
          const uint8_t* v1088 = v983 + 403;
          const int8_t* v1089 = (const int8_t*) v1088;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1090 = *(const int8_t *)(v1089);
          vint16m1_t v1091 = v977;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1092 = __riscv_vwmacc_vx_i16m1(v1091, v1090, v1012, 8);
          v977 = v1092;
        }
        vint16m1_t v1093 = v947;
        vint16m1_t v1094 = v949;
        vint16m1_t v1095 = v951;
        vint16m1_t v1096 = v953;
        vint16m1_t v1097 = v955;
        vint16m1_t v1098 = v957;
        vint16m1_t v1099 = v959;
        vint16m1_t v1100 = v961;
        vint16m1_t v1101 = v963;
        vint16m1_t v1102 = v965;
        vint16m1_t v1103 = v967;
        vint16m1_t v1104 = v969;
        vint16m1_t v1105 = v971;
        vint16m1_t v1106 = v973;
        vint16m1_t v1107 = v975;
        vint16m1_t v1108 = v977;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v1109 = v923;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1110 = __riscv_vwmacc_vv_i32m2(v1109, v934, v1093, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1111 = __riscv_vwmacc_vv_i32m2(v1110, v938, v1094, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1112 = __riscv_vwmacc_vv_i32m2(v1111, v942, v1095, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1113 = __riscv_vwmacc_vv_i32m2(v1112, v946, v1096, 8);
        v923 = v1113;
        vint32m2_t v1114 = v925;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1115 = __riscv_vwmacc_vv_i32m2(v1114, v934, v1097, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1116 = __riscv_vwmacc_vv_i32m2(v1115, v938, v1098, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1117 = __riscv_vwmacc_vv_i32m2(v1116, v942, v1099, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1118 = __riscv_vwmacc_vv_i32m2(v1117, v946, v1100, 8);
        v925 = v1118;
        vint32m2_t v1119 = v927;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1120 = __riscv_vwmacc_vv_i32m2(v1119, v934, v1101, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1121 = __riscv_vwmacc_vv_i32m2(v1120, v938, v1102, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1122 = __riscv_vwmacc_vv_i32m2(v1121, v942, v1103, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1123 = __riscv_vwmacc_vv_i32m2(v1122, v946, v1104, 8);
        v927 = v1123;
        vint32m2_t v1124 = v929;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1125 = __riscv_vwmacc_vv_i32m2(v1124, v934, v1105, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1126 = __riscv_vwmacc_vv_i32m2(v1125, v938, v1106, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1127 = __riscv_vwmacc_vv_i32m2(v1126, v942, v1107, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1128 = __riscv_vwmacc_vv_i32m2(v1127, v946, v1108, 8);
        v929 = v1128;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v1129 = v905 + 56;
        const int8_t* v1130 = (const int8_t*) v1129;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1131 = __riscv_vle8_v_i8mf2(v1130, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1132 = __riscv_vsext_vf2_i16m1(v1131, 8);
        const uint8_t* v1133 = v905 + 88;
        const int8_t* v1134 = (const int8_t*) v1133;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1135 = __riscv_vle8_v_i8mf2(v1134, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1136 = __riscv_vsext_vf2_i16m1(v1135, 8);
        const uint8_t* v1137 = v905 + 120;
        const int8_t* v1138 = (const int8_t*) v1137;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1139 = __riscv_vle8_v_i8mf2(v1138, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1140 = __riscv_vsext_vf2_i16m1(v1139, 8);
        const uint8_t* v1141 = v905 + 152;
        const int8_t* v1142 = (const int8_t*) v1141;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1143 = __riscv_vle8_v_i8mf2(v1142, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1144 = __riscv_vsext_vf2_i16m1(v1143, 8);
        vint16m1_t v1145;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1146 = __riscv_vmv_v_x_i16m1(0, 8);
        v1145 = v1146;
        vint16m1_t v1147;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1148 = __riscv_vmv_v_x_i16m1(0, 8);
        v1147 = v1148;
        vint16m1_t v1149;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1150 = __riscv_vmv_v_x_i16m1(0, 8);
        v1149 = v1150;
        vint16m1_t v1151;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1152 = __riscv_vmv_v_x_i16m1(0, 8);
        v1151 = v1152;
        vint16m1_t v1153;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1154 = __riscv_vmv_v_x_i16m1(0, 8);
        v1153 = v1154;
        vint16m1_t v1155;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1156 = __riscv_vmv_v_x_i16m1(0, 8);
        v1155 = v1156;
        vint16m1_t v1157;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1158 = __riscv_vmv_v_x_i16m1(0, 8);
        v1157 = v1158;
        vint16m1_t v1159;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1160 = __riscv_vmv_v_x_i16m1(0, 8);
        v1159 = v1160;
        vint16m1_t v1161;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1162 = __riscv_vmv_v_x_i16m1(0, 8);
        v1161 = v1162;
        vint16m1_t v1163;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1164 = __riscv_vmv_v_x_i16m1(0, 8);
        v1163 = v1164;
        vint16m1_t v1165;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1166 = __riscv_vmv_v_x_i16m1(0, 8);
        v1165 = v1166;
        vint16m1_t v1167;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1168 = __riscv_vmv_v_x_i16m1(0, 8);
        v1167 = v1168;
        vint16m1_t v1169;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1170 = __riscv_vmv_v_x_i16m1(0, 8);
        v1169 = v1170;
        vint16m1_t v1171;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1172 = __riscv_vmv_v_x_i16m1(0, 8);
        v1171 = v1172;
        vint16m1_t v1173;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1174 = __riscv_vmv_v_x_i16m1(0, 8);
        v1173 = v1174;
        vint16m1_t v1175;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1176 = __riscv_vmv_v_x_i16m1(0, 8);
        v1175 = v1176;
        for (size_t v1177 = 0; v1177 < 16; v1177 += 1) {
          size_t v1178 = v1177 * 16;
          const uint8_t* v1179 = v905 + v1178;
          size_t v1180 = v1177 * 4;
          const uint8_t* v1181 = v907 + v1180;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
          const uint8_t* v1182 = v1179 + 1064;
          const uint8_t* v1183 = (const uint8_t*) v1182;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1184 = __riscv_vle8_v_u8mf2(v1183, 8);
          const uint8_t* v1185 = v1179 + 552;
          const uint8_t* v1186 = (const uint8_t*) v1185;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1187 = __riscv_vle8_v_u8mf2(v1186, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1188 = __riscv_vand_vx_u8mf2(v1184, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1189 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1188);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1190 = __riscv_vand_vx_u8mf2(v1187, 1, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1191 = __riscv_vmseq_vx_u8mf2_b16(v1190, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1192 = __riscv_vadd_vx_i8mf2_mu(v1191, v1189, v1189, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1193 = __riscv_vsrl_vx_u8mf2(v1184, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1194 = __riscv_vand_vx_u8mf2(v1193, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1195 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1194);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1196 = __riscv_vand_vx_u8mf2(v1187, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1197 = __riscv_vmseq_vx_u8mf2_b16(v1196, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1198 = __riscv_vadd_vx_i8mf2_mu(v1197, v1195, v1195, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1199 = __riscv_vsrl_vx_u8mf2(v1184, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1200 = __riscv_vand_vx_u8mf2(v1199, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1201 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1200);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1202 = __riscv_vand_vx_u8mf2(v1187, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1203 = __riscv_vmseq_vx_u8mf2_b16(v1202, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1204 = __riscv_vadd_vx_i8mf2_mu(v1203, v1201, v1201, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1205 = __riscv_vsrl_vx_u8mf2(v1184, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1206 = __riscv_vand_vx_u8mf2(v1205, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1207 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1206);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1208 = __riscv_vand_vx_u8mf2(v1187, 8, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1209 = __riscv_vmseq_vx_u8mf2_b16(v1208, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1210 = __riscv_vadd_vx_i8mf2_mu(v1209, v1207, v1207, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1211 = v1181 + 80;
          const int8_t* v1212 = (const int8_t*) v1211;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1213 = *(const int8_t *)(v1212);
          vint16m1_t v1214 = v1145;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1215 = __riscv_vwmacc_vx_i16m1(v1214, v1213, v1192, 8);
          v1145 = v1215;
          const uint8_t* v1216 = v1181 + 208;
          const int8_t* v1217 = (const int8_t*) v1216;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1218 = *(const int8_t *)(v1217);
          vint16m1_t v1219 = v1147;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1220 = __riscv_vwmacc_vx_i16m1(v1219, v1218, v1198, 8);
          v1147 = v1220;
          const uint8_t* v1221 = v1181 + 336;
          const int8_t* v1222 = (const int8_t*) v1221;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1223 = *(const int8_t *)(v1222);
          vint16m1_t v1224 = v1149;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1225 = __riscv_vwmacc_vx_i16m1(v1224, v1223, v1204, 8);
          v1149 = v1225;
          const uint8_t* v1226 = v1181 + 464;
          const int8_t* v1227 = (const int8_t*) v1226;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1228 = *(const int8_t *)(v1227);
          vint16m1_t v1229 = v1151;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1230 = __riscv_vwmacc_vx_i16m1(v1229, v1228, v1210, 8);
          v1151 = v1230;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1231 = v1181 + 81;
          const int8_t* v1232 = (const int8_t*) v1231;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1233 = *(const int8_t *)(v1232);
          vint16m1_t v1234 = v1153;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1235 = __riscv_vwmacc_vx_i16m1(v1234, v1233, v1192, 8);
          v1153 = v1235;
          const uint8_t* v1236 = v1181 + 209;
          const int8_t* v1237 = (const int8_t*) v1236;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1238 = *(const int8_t *)(v1237);
          vint16m1_t v1239 = v1155;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1240 = __riscv_vwmacc_vx_i16m1(v1239, v1238, v1198, 8);
          v1155 = v1240;
          const uint8_t* v1241 = v1181 + 337;
          const int8_t* v1242 = (const int8_t*) v1241;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1243 = *(const int8_t *)(v1242);
          vint16m1_t v1244 = v1157;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1245 = __riscv_vwmacc_vx_i16m1(v1244, v1243, v1204, 8);
          v1157 = v1245;
          const uint8_t* v1246 = v1181 + 465;
          const int8_t* v1247 = (const int8_t*) v1246;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1248 = *(const int8_t *)(v1247);
          vint16m1_t v1249 = v1159;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1250 = __riscv_vwmacc_vx_i16m1(v1249, v1248, v1210, 8);
          v1159 = v1250;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1251 = v1181 + 82;
          const int8_t* v1252 = (const int8_t*) v1251;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1253 = *(const int8_t *)(v1252);
          vint16m1_t v1254 = v1161;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1255 = __riscv_vwmacc_vx_i16m1(v1254, v1253, v1192, 8);
          v1161 = v1255;
          const uint8_t* v1256 = v1181 + 210;
          const int8_t* v1257 = (const int8_t*) v1256;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1258 = *(const int8_t *)(v1257);
          vint16m1_t v1259 = v1163;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1260 = __riscv_vwmacc_vx_i16m1(v1259, v1258, v1198, 8);
          v1163 = v1260;
          const uint8_t* v1261 = v1181 + 338;
          const int8_t* v1262 = (const int8_t*) v1261;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1263 = *(const int8_t *)(v1262);
          vint16m1_t v1264 = v1165;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1265 = __riscv_vwmacc_vx_i16m1(v1264, v1263, v1204, 8);
          v1165 = v1265;
          const uint8_t* v1266 = v1181 + 466;
          const int8_t* v1267 = (const int8_t*) v1266;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1268 = *(const int8_t *)(v1267);
          vint16m1_t v1269 = v1167;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1270 = __riscv_vwmacc_vx_i16m1(v1269, v1268, v1210, 8);
          v1167 = v1270;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1271 = v1181 + 83;
          const int8_t* v1272 = (const int8_t*) v1271;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1273 = *(const int8_t *)(v1272);
          vint16m1_t v1274 = v1169;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1275 = __riscv_vwmacc_vx_i16m1(v1274, v1273, v1192, 8);
          v1169 = v1275;
          const uint8_t* v1276 = v1181 + 211;
          const int8_t* v1277 = (const int8_t*) v1276;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1278 = *(const int8_t *)(v1277);
          vint16m1_t v1279 = v1171;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1280 = __riscv_vwmacc_vx_i16m1(v1279, v1278, v1198, 8);
          v1171 = v1280;
          const uint8_t* v1281 = v1181 + 339;
          const int8_t* v1282 = (const int8_t*) v1281;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1283 = *(const int8_t *)(v1282);
          vint16m1_t v1284 = v1173;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1285 = __riscv_vwmacc_vx_i16m1(v1284, v1283, v1204, 8);
          v1173 = v1285;
          const uint8_t* v1286 = v1181 + 467;
          const int8_t* v1287 = (const int8_t*) v1286;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1288 = *(const int8_t *)(v1287);
          vint16m1_t v1289 = v1175;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1290 = __riscv_vwmacc_vx_i16m1(v1289, v1288, v1210, 8);
          v1175 = v1290;
        }
        vint16m1_t v1291 = v1145;
        vint16m1_t v1292 = v1147;
        vint16m1_t v1293 = v1149;
        vint16m1_t v1294 = v1151;
        vint16m1_t v1295 = v1153;
        vint16m1_t v1296 = v1155;
        vint16m1_t v1297 = v1157;
        vint16m1_t v1298 = v1159;
        vint16m1_t v1299 = v1161;
        vint16m1_t v1300 = v1163;
        vint16m1_t v1301 = v1165;
        vint16m1_t v1302 = v1167;
        vint16m1_t v1303 = v1169;
        vint16m1_t v1304 = v1171;
        vint16m1_t v1305 = v1173;
        vint16m1_t v1306 = v1175;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v1307 = v923;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1308 = __riscv_vwmacc_vv_i32m2(v1307, v1132, v1291, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1309 = __riscv_vwmacc_vv_i32m2(v1308, v1136, v1292, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1310 = __riscv_vwmacc_vv_i32m2(v1309, v1140, v1293, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1311 = __riscv_vwmacc_vv_i32m2(v1310, v1144, v1294, 8);
        v923 = v1311;
        vint32m2_t v1312 = v925;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1313 = __riscv_vwmacc_vv_i32m2(v1312, v1132, v1295, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1314 = __riscv_vwmacc_vv_i32m2(v1313, v1136, v1296, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1315 = __riscv_vwmacc_vv_i32m2(v1314, v1140, v1297, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1316 = __riscv_vwmacc_vv_i32m2(v1315, v1144, v1298, 8);
        v925 = v1316;
        vint32m2_t v1317 = v927;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1318 = __riscv_vwmacc_vv_i32m2(v1317, v1132, v1299, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1319 = __riscv_vwmacc_vv_i32m2(v1318, v1136, v1300, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1320 = __riscv_vwmacc_vv_i32m2(v1319, v1140, v1301, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1321 = __riscv_vwmacc_vv_i32m2(v1320, v1144, v1302, 8);
        v927 = v1321;
        vint32m2_t v1322 = v929;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1323 = __riscv_vwmacc_vv_i32m2(v1322, v1132, v1303, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1324 = __riscv_vwmacc_vv_i32m2(v1323, v1136, v1304, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1325 = __riscv_vwmacc_vv_i32m2(v1324, v1140, v1305, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1326 = __riscv_vwmacc_vv_i32m2(v1325, v1144, v1306, 8);
        v929 = v1326;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v1327 = v905 + 168;
        const int8_t* v1328 = (const int8_t*) v1327;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1329 = __riscv_vle8_v_i8mf2(v1328, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1330 = __riscv_vsext_vf2_i16m1(v1329, 8);
        const uint8_t* v1331 = v905 + 200;
        const int8_t* v1332 = (const int8_t*) v1331;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1333 = __riscv_vle8_v_i8mf2(v1332, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1334 = __riscv_vsext_vf2_i16m1(v1333, 8);
        const uint8_t* v1335 = v905 + 232;
        const int8_t* v1336 = (const int8_t*) v1335;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1337 = __riscv_vle8_v_i8mf2(v1336, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1338 = __riscv_vsext_vf2_i16m1(v1337, 8);
        const uint8_t* v1339 = v905 + 264;
        const int8_t* v1340 = (const int8_t*) v1339;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1341 = __riscv_vle8_v_i8mf2(v1340, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1342 = __riscv_vsext_vf2_i16m1(v1341, 8);
        vint16m1_t v1343;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1344 = __riscv_vmv_v_x_i16m1(0, 8);
        v1343 = v1344;
        vint16m1_t v1345;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1346 = __riscv_vmv_v_x_i16m1(0, 8);
        v1345 = v1346;
        vint16m1_t v1347;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1348 = __riscv_vmv_v_x_i16m1(0, 8);
        v1347 = v1348;
        vint16m1_t v1349;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1350 = __riscv_vmv_v_x_i16m1(0, 8);
        v1349 = v1350;
        vint16m1_t v1351;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1352 = __riscv_vmv_v_x_i16m1(0, 8);
        v1351 = v1352;
        vint16m1_t v1353;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1354 = __riscv_vmv_v_x_i16m1(0, 8);
        v1353 = v1354;
        vint16m1_t v1355;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1356 = __riscv_vmv_v_x_i16m1(0, 8);
        v1355 = v1356;
        vint16m1_t v1357;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1358 = __riscv_vmv_v_x_i16m1(0, 8);
        v1357 = v1358;
        vint16m1_t v1359;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1360 = __riscv_vmv_v_x_i16m1(0, 8);
        v1359 = v1360;
        vint16m1_t v1361;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1362 = __riscv_vmv_v_x_i16m1(0, 8);
        v1361 = v1362;
        vint16m1_t v1363;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1364 = __riscv_vmv_v_x_i16m1(0, 8);
        v1363 = v1364;
        vint16m1_t v1365;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1366 = __riscv_vmv_v_x_i16m1(0, 8);
        v1365 = v1366;
        vint16m1_t v1367;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1368 = __riscv_vmv_v_x_i16m1(0, 8);
        v1367 = v1368;
        vint16m1_t v1369;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1370 = __riscv_vmv_v_x_i16m1(0, 8);
        v1369 = v1370;
        vint16m1_t v1371;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1372 = __riscv_vmv_v_x_i16m1(0, 8);
        v1371 = v1372;
        vint16m1_t v1373;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1374 = __riscv_vmv_v_x_i16m1(0, 8);
        v1373 = v1374;
        for (size_t v1375 = 0; v1375 < 16; v1375 += 1) {
          size_t v1376 = v1375 * 16;
          const uint8_t* v1377 = v905 + v1376;
          size_t v1378 = v1375 * 4;
          const uint8_t* v1379 = v907 + v1378;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
          const uint8_t* v1380 = v1377 + 1320;
          const uint8_t* v1381 = (const uint8_t*) v1380;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1382 = __riscv_vle8_v_u8mf2(v1381, 8);
          const uint8_t* v1383 = v1377 + 296;
          const uint8_t* v1384 = (const uint8_t*) v1383;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1385 = __riscv_vle8_v_u8mf2(v1384, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1386 = __riscv_vand_vx_u8mf2(v1382, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1387 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1386);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1388 = __riscv_vand_vx_u8mf2(v1385, 16, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1389 = __riscv_vmseq_vx_u8mf2_b16(v1388, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1390 = __riscv_vadd_vx_i8mf2_mu(v1389, v1387, v1387, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1391 = __riscv_vsrl_vx_u8mf2(v1382, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1392 = __riscv_vand_vx_u8mf2(v1391, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1393 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1392);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1394 = __riscv_vand_vx_u8mf2(v1385, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1395 = __riscv_vmseq_vx_u8mf2_b16(v1394, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1396 = __riscv_vadd_vx_i8mf2_mu(v1395, v1393, v1393, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1397 = __riscv_vsrl_vx_u8mf2(v1382, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1398 = __riscv_vand_vx_u8mf2(v1397, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1399 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1398);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1400 = __riscv_vand_vx_u8mf2(v1385, 64, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1401 = __riscv_vmseq_vx_u8mf2_b16(v1400, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1402 = __riscv_vadd_vx_i8mf2_mu(v1401, v1399, v1399, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1403 = __riscv_vsrl_vx_u8mf2(v1382, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1404 = __riscv_vand_vx_u8mf2(v1403, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1405 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1404);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1406 = __riscv_vand_vx_u8mf2(v1385, 128, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1407 = __riscv_vmseq_vx_u8mf2_b16(v1406, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1408 = __riscv_vadd_vx_i8mf2_mu(v1407, v1405, v1405, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1409 = v1379 + 528;
          const int8_t* v1410 = (const int8_t*) v1409;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1411 = *(const int8_t *)(v1410);
          vint16m1_t v1412 = v1343;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1413 = __riscv_vwmacc_vx_i16m1(v1412, v1411, v1390, 8);
          v1343 = v1413;
          const uint8_t* v1414 = v1379 + 656;
          const int8_t* v1415 = (const int8_t*) v1414;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1416 = *(const int8_t *)(v1415);
          vint16m1_t v1417 = v1345;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1418 = __riscv_vwmacc_vx_i16m1(v1417, v1416, v1396, 8);
          v1345 = v1418;
          const uint8_t* v1419 = v1379 + 784;
          const int8_t* v1420 = (const int8_t*) v1419;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1421 = *(const int8_t *)(v1420);
          vint16m1_t v1422 = v1347;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1423 = __riscv_vwmacc_vx_i16m1(v1422, v1421, v1402, 8);
          v1347 = v1423;
          const uint8_t* v1424 = v1379 + 912;
          const int8_t* v1425 = (const int8_t*) v1424;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1426 = *(const int8_t *)(v1425);
          vint16m1_t v1427 = v1349;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1428 = __riscv_vwmacc_vx_i16m1(v1427, v1426, v1408, 8);
          v1349 = v1428;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1429 = v1379 + 529;
          const int8_t* v1430 = (const int8_t*) v1429;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1431 = *(const int8_t *)(v1430);
          vint16m1_t v1432 = v1351;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1433 = __riscv_vwmacc_vx_i16m1(v1432, v1431, v1390, 8);
          v1351 = v1433;
          const uint8_t* v1434 = v1379 + 657;
          const int8_t* v1435 = (const int8_t*) v1434;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1436 = *(const int8_t *)(v1435);
          vint16m1_t v1437 = v1353;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1438 = __riscv_vwmacc_vx_i16m1(v1437, v1436, v1396, 8);
          v1353 = v1438;
          const uint8_t* v1439 = v1379 + 785;
          const int8_t* v1440 = (const int8_t*) v1439;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1441 = *(const int8_t *)(v1440);
          vint16m1_t v1442 = v1355;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1443 = __riscv_vwmacc_vx_i16m1(v1442, v1441, v1402, 8);
          v1355 = v1443;
          const uint8_t* v1444 = v1379 + 913;
          const int8_t* v1445 = (const int8_t*) v1444;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1446 = *(const int8_t *)(v1445);
          vint16m1_t v1447 = v1357;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1448 = __riscv_vwmacc_vx_i16m1(v1447, v1446, v1408, 8);
          v1357 = v1448;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1449 = v1379 + 530;
          const int8_t* v1450 = (const int8_t*) v1449;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1451 = *(const int8_t *)(v1450);
          vint16m1_t v1452 = v1359;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1453 = __riscv_vwmacc_vx_i16m1(v1452, v1451, v1390, 8);
          v1359 = v1453;
          const uint8_t* v1454 = v1379 + 658;
          const int8_t* v1455 = (const int8_t*) v1454;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1456 = *(const int8_t *)(v1455);
          vint16m1_t v1457 = v1361;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1458 = __riscv_vwmacc_vx_i16m1(v1457, v1456, v1396, 8);
          v1361 = v1458;
          const uint8_t* v1459 = v1379 + 786;
          const int8_t* v1460 = (const int8_t*) v1459;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1461 = *(const int8_t *)(v1460);
          vint16m1_t v1462 = v1363;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1463 = __riscv_vwmacc_vx_i16m1(v1462, v1461, v1402, 8);
          v1363 = v1463;
          const uint8_t* v1464 = v1379 + 914;
          const int8_t* v1465 = (const int8_t*) v1464;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1466 = *(const int8_t *)(v1465);
          vint16m1_t v1467 = v1365;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1468 = __riscv_vwmacc_vx_i16m1(v1467, v1466, v1408, 8);
          v1365 = v1468;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1469 = v1379 + 531;
          const int8_t* v1470 = (const int8_t*) v1469;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1471 = *(const int8_t *)(v1470);
          vint16m1_t v1472 = v1367;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1473 = __riscv_vwmacc_vx_i16m1(v1472, v1471, v1390, 8);
          v1367 = v1473;
          const uint8_t* v1474 = v1379 + 659;
          const int8_t* v1475 = (const int8_t*) v1474;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1476 = *(const int8_t *)(v1475);
          vint16m1_t v1477 = v1369;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1478 = __riscv_vwmacc_vx_i16m1(v1477, v1476, v1396, 8);
          v1369 = v1478;
          const uint8_t* v1479 = v1379 + 787;
          const int8_t* v1480 = (const int8_t*) v1479;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1481 = *(const int8_t *)(v1480);
          vint16m1_t v1482 = v1371;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1483 = __riscv_vwmacc_vx_i16m1(v1482, v1481, v1402, 8);
          v1371 = v1483;
          const uint8_t* v1484 = v1379 + 915;
          const int8_t* v1485 = (const int8_t*) v1484;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1486 = *(const int8_t *)(v1485);
          vint16m1_t v1487 = v1373;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1488 = __riscv_vwmacc_vx_i16m1(v1487, v1486, v1408, 8);
          v1373 = v1488;
        }
        vint16m1_t v1489 = v1343;
        vint16m1_t v1490 = v1345;
        vint16m1_t v1491 = v1347;
        vint16m1_t v1492 = v1349;
        vint16m1_t v1493 = v1351;
        vint16m1_t v1494 = v1353;
        vint16m1_t v1495 = v1355;
        vint16m1_t v1496 = v1357;
        vint16m1_t v1497 = v1359;
        vint16m1_t v1498 = v1361;
        vint16m1_t v1499 = v1363;
        vint16m1_t v1500 = v1365;
        vint16m1_t v1501 = v1367;
        vint16m1_t v1502 = v1369;
        vint16m1_t v1503 = v1371;
        vint16m1_t v1504 = v1373;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v1505 = v923;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1506 = __riscv_vwmacc_vv_i32m2(v1505, v1330, v1489, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1507 = __riscv_vwmacc_vv_i32m2(v1506, v1334, v1490, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1508 = __riscv_vwmacc_vv_i32m2(v1507, v1338, v1491, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1509 = __riscv_vwmacc_vv_i32m2(v1508, v1342, v1492, 8);
        v923 = v1509;
        vint32m2_t v1510 = v925;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1511 = __riscv_vwmacc_vv_i32m2(v1510, v1330, v1493, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1512 = __riscv_vwmacc_vv_i32m2(v1511, v1334, v1494, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1513 = __riscv_vwmacc_vv_i32m2(v1512, v1338, v1495, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1514 = __riscv_vwmacc_vv_i32m2(v1513, v1342, v1496, 8);
        v925 = v1514;
        vint32m2_t v1515 = v927;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1516 = __riscv_vwmacc_vv_i32m2(v1515, v1330, v1497, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1517 = __riscv_vwmacc_vv_i32m2(v1516, v1334, v1498, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1518 = __riscv_vwmacc_vv_i32m2(v1517, v1338, v1499, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1519 = __riscv_vwmacc_vv_i32m2(v1518, v1342, v1500, 8);
        v927 = v1519;
        vint32m2_t v1520 = v929;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1521 = __riscv_vwmacc_vv_i32m2(v1520, v1330, v1501, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1522 = __riscv_vwmacc_vv_i32m2(v1521, v1334, v1502, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1523 = __riscv_vwmacc_vv_i32m2(v1522, v1338, v1503, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1524 = __riscv_vwmacc_vv_i32m2(v1523, v1342, v1504, 8);
        v929 = v1524;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v1525 = v905 + 184;
        const int8_t* v1526 = (const int8_t*) v1525;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1527 = __riscv_vle8_v_i8mf2(v1526, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1528 = __riscv_vsext_vf2_i16m1(v1527, 8);
        const uint8_t* v1529 = v905 + 216;
        const int8_t* v1530 = (const int8_t*) v1529;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1531 = __riscv_vle8_v_i8mf2(v1530, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1532 = __riscv_vsext_vf2_i16m1(v1531, 8);
        const uint8_t* v1533 = v905 + 248;
        const int8_t* v1534 = (const int8_t*) v1533;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1535 = __riscv_vle8_v_i8mf2(v1534, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1536 = __riscv_vsext_vf2_i16m1(v1535, 8);
        const uint8_t* v1537 = v905 + 280;
        const int8_t* v1538 = (const int8_t*) v1537;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1539 = __riscv_vle8_v_i8mf2(v1538, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1540 = __riscv_vsext_vf2_i16m1(v1539, 8);
        vint16m1_t v1541;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1542 = __riscv_vmv_v_x_i16m1(0, 8);
        v1541 = v1542;
        vint16m1_t v1543;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1544 = __riscv_vmv_v_x_i16m1(0, 8);
        v1543 = v1544;
        vint16m1_t v1545;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1546 = __riscv_vmv_v_x_i16m1(0, 8);
        v1545 = v1546;
        vint16m1_t v1547;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1548 = __riscv_vmv_v_x_i16m1(0, 8);
        v1547 = v1548;
        vint16m1_t v1549;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1550 = __riscv_vmv_v_x_i16m1(0, 8);
        v1549 = v1550;
        vint16m1_t v1551;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1552 = __riscv_vmv_v_x_i16m1(0, 8);
        v1551 = v1552;
        vint16m1_t v1553;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1554 = __riscv_vmv_v_x_i16m1(0, 8);
        v1553 = v1554;
        vint16m1_t v1555;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1556 = __riscv_vmv_v_x_i16m1(0, 8);
        v1555 = v1556;
        vint16m1_t v1557;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1558 = __riscv_vmv_v_x_i16m1(0, 8);
        v1557 = v1558;
        vint16m1_t v1559;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1560 = __riscv_vmv_v_x_i16m1(0, 8);
        v1559 = v1560;
        vint16m1_t v1561;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1562 = __riscv_vmv_v_x_i16m1(0, 8);
        v1561 = v1562;
        vint16m1_t v1563;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1564 = __riscv_vmv_v_x_i16m1(0, 8);
        v1563 = v1564;
        vint16m1_t v1565;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1566 = __riscv_vmv_v_x_i16m1(0, 8);
        v1565 = v1566;
        vint16m1_t v1567;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1568 = __riscv_vmv_v_x_i16m1(0, 8);
        v1567 = v1568;
        vint16m1_t v1569;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1570 = __riscv_vmv_v_x_i16m1(0, 8);
        v1569 = v1570;
        vint16m1_t v1571;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1572 = __riscv_vmv_v_x_i16m1(0, 8);
        v1571 = v1572;
        for (size_t v1573 = 0; v1573 < 16; v1573 += 1) {
          size_t v1574 = v1573 * 16;
          const uint8_t* v1575 = v905 + v1574;
          size_t v1576 = v1573 * 4;
          const uint8_t* v1577 = v907 + v1576;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_qs_hmask_addr
          const uint8_t* v1578 = v1575 + 1576;
          const uint8_t* v1579 = (const uint8_t*) v1578;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1580 = __riscv_vle8_v_u8mf2(v1579, 8);
          const uint8_t* v1581 = v1575 + 552;
          const uint8_t* v1582 = (const uint8_t*) v1581;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1583 = __riscv_vle8_v_u8mf2(v1582, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1584 = __riscv_vand_vx_u8mf2(v1580, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1585 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1584);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1586 = __riscv_vand_vx_u8mf2(v1583, 16, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1587 = __riscv_vmseq_vx_u8mf2_b16(v1586, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1588 = __riscv_vadd_vx_i8mf2_mu(v1587, v1585, v1585, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1589 = __riscv_vsrl_vx_u8mf2(v1580, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1590 = __riscv_vand_vx_u8mf2(v1589, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1591 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1590);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1592 = __riscv_vand_vx_u8mf2(v1583, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1593 = __riscv_vmseq_vx_u8mf2_b16(v1592, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1594 = __riscv_vadd_vx_i8mf2_mu(v1593, v1591, v1591, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1595 = __riscv_vsrl_vx_u8mf2(v1580, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1596 = __riscv_vand_vx_u8mf2(v1595, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1597 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1596);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1598 = __riscv_vand_vx_u8mf2(v1583, 64, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1599 = __riscv_vmseq_vx_u8mf2_b16(v1598, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1600 = __riscv_vadd_vx_i8mf2_mu(v1599, v1597, v1597, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1601 = __riscv_vsrl_vx_u8mf2(v1580, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1602 = __riscv_vand_vx_u8mf2(v1601, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1603 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1602);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1604 = __riscv_vand_vx_u8mf2(v1583, 128, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vx_u8mf2_b16
          vbool16_t v1605 = __riscv_vmseq_vx_u8mf2_b16(v1604, 0, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8mf2_mu
          vint8mf2_t v1606 = __riscv_vadd_vx_i8mf2_mu(v1605, v1603, v1603, -4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1607 = v1577 + 592;
          const int8_t* v1608 = (const int8_t*) v1607;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1609 = *(const int8_t *)(v1608);
          vint16m1_t v1610 = v1541;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1611 = __riscv_vwmacc_vx_i16m1(v1610, v1609, v1588, 8);
          v1541 = v1611;
          const uint8_t* v1612 = v1577 + 720;
          const int8_t* v1613 = (const int8_t*) v1612;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1614 = *(const int8_t *)(v1613);
          vint16m1_t v1615 = v1543;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1616 = __riscv_vwmacc_vx_i16m1(v1615, v1614, v1594, 8);
          v1543 = v1616;
          const uint8_t* v1617 = v1577 + 848;
          const int8_t* v1618 = (const int8_t*) v1617;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1619 = *(const int8_t *)(v1618);
          vint16m1_t v1620 = v1545;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1621 = __riscv_vwmacc_vx_i16m1(v1620, v1619, v1600, 8);
          v1545 = v1621;
          const uint8_t* v1622 = v1577 + 976;
          const int8_t* v1623 = (const int8_t*) v1622;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1624 = *(const int8_t *)(v1623);
          vint16m1_t v1625 = v1547;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1626 = __riscv_vwmacc_vx_i16m1(v1625, v1624, v1606, 8);
          v1547 = v1626;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1627 = v1577 + 593;
          const int8_t* v1628 = (const int8_t*) v1627;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1629 = *(const int8_t *)(v1628);
          vint16m1_t v1630 = v1549;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1631 = __riscv_vwmacc_vx_i16m1(v1630, v1629, v1588, 8);
          v1549 = v1631;
          const uint8_t* v1632 = v1577 + 721;
          const int8_t* v1633 = (const int8_t*) v1632;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1634 = *(const int8_t *)(v1633);
          vint16m1_t v1635 = v1551;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1636 = __riscv_vwmacc_vx_i16m1(v1635, v1634, v1594, 8);
          v1551 = v1636;
          const uint8_t* v1637 = v1577 + 849;
          const int8_t* v1638 = (const int8_t*) v1637;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1639 = *(const int8_t *)(v1638);
          vint16m1_t v1640 = v1553;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1641 = __riscv_vwmacc_vx_i16m1(v1640, v1639, v1600, 8);
          v1553 = v1641;
          const uint8_t* v1642 = v1577 + 977;
          const int8_t* v1643 = (const int8_t*) v1642;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1644 = *(const int8_t *)(v1643);
          vint16m1_t v1645 = v1555;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1646 = __riscv_vwmacc_vx_i16m1(v1645, v1644, v1606, 8);
          v1555 = v1646;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1647 = v1577 + 594;
          const int8_t* v1648 = (const int8_t*) v1647;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1649 = *(const int8_t *)(v1648);
          vint16m1_t v1650 = v1557;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1651 = __riscv_vwmacc_vx_i16m1(v1650, v1649, v1588, 8);
          v1557 = v1651;
          const uint8_t* v1652 = v1577 + 722;
          const int8_t* v1653 = (const int8_t*) v1652;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1654 = *(const int8_t *)(v1653);
          vint16m1_t v1655 = v1559;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1656 = __riscv_vwmacc_vx_i16m1(v1655, v1654, v1594, 8);
          v1559 = v1656;
          const uint8_t* v1657 = v1577 + 850;
          const int8_t* v1658 = (const int8_t*) v1657;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1659 = *(const int8_t *)(v1658);
          vint16m1_t v1660 = v1561;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1661 = __riscv_vwmacc_vx_i16m1(v1660, v1659, v1600, 8);
          v1561 = v1661;
          const uint8_t* v1662 = v1577 + 978;
          const int8_t* v1663 = (const int8_t*) v1662;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1664 = *(const int8_t *)(v1663);
          vint16m1_t v1665 = v1563;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1666 = __riscv_vwmacc_vx_i16m1(v1665, v1664, v1606, 8);
          v1563 = v1666;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1667 = v1577 + 595;
          const int8_t* v1668 = (const int8_t*) v1667;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1669 = *(const int8_t *)(v1668);
          vint16m1_t v1670 = v1565;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1671 = __riscv_vwmacc_vx_i16m1(v1670, v1669, v1588, 8);
          v1565 = v1671;
          const uint8_t* v1672 = v1577 + 723;
          const int8_t* v1673 = (const int8_t*) v1672;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1674 = *(const int8_t *)(v1673);
          vint16m1_t v1675 = v1567;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1676 = __riscv_vwmacc_vx_i16m1(v1675, v1674, v1594, 8);
          v1567 = v1676;
          const uint8_t* v1677 = v1577 + 851;
          const int8_t* v1678 = (const int8_t*) v1677;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1679 = *(const int8_t *)(v1678);
          vint16m1_t v1680 = v1569;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1681 = __riscv_vwmacc_vx_i16m1(v1680, v1679, v1600, 8);
          v1569 = v1681;
          const uint8_t* v1682 = v1577 + 979;
          const int8_t* v1683 = (const int8_t*) v1682;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1684 = *(const int8_t *)(v1683);
          vint16m1_t v1685 = v1571;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1686 = __riscv_vwmacc_vx_i16m1(v1685, v1684, v1606, 8);
          v1571 = v1686;
        }
        vint16m1_t v1687 = v1541;
        vint16m1_t v1688 = v1543;
        vint16m1_t v1689 = v1545;
        vint16m1_t v1690 = v1547;
        vint16m1_t v1691 = v1549;
        vint16m1_t v1692 = v1551;
        vint16m1_t v1693 = v1553;
        vint16m1_t v1694 = v1555;
        vint16m1_t v1695 = v1557;
        vint16m1_t v1696 = v1559;
        vint16m1_t v1697 = v1561;
        vint16m1_t v1698 = v1563;
        vint16m1_t v1699 = v1565;
        vint16m1_t v1700 = v1567;
        vint16m1_t v1701 = v1569;
        vint16m1_t v1702 = v1571;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v1703 = v923;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1704 = __riscv_vwmacc_vv_i32m2(v1703, v1528, v1687, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1705 = __riscv_vwmacc_vv_i32m2(v1704, v1532, v1688, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1706 = __riscv_vwmacc_vv_i32m2(v1705, v1536, v1689, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1707 = __riscv_vwmacc_vv_i32m2(v1706, v1540, v1690, 8);
        v923 = v1707;
        vint32m2_t v1708 = v925;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1709 = __riscv_vwmacc_vv_i32m2(v1708, v1528, v1691, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1710 = __riscv_vwmacc_vv_i32m2(v1709, v1532, v1692, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1711 = __riscv_vwmacc_vv_i32m2(v1710, v1536, v1693, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1712 = __riscv_vwmacc_vv_i32m2(v1711, v1540, v1694, 8);
        v925 = v1712;
        vint32m2_t v1713 = v927;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1714 = __riscv_vwmacc_vv_i32m2(v1713, v1528, v1695, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1715 = __riscv_vwmacc_vv_i32m2(v1714, v1532, v1696, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1716 = __riscv_vwmacc_vv_i32m2(v1715, v1536, v1697, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1717 = __riscv_vwmacc_vv_i32m2(v1716, v1540, v1698, 8);
        v927 = v1717;
        vint32m2_t v1718 = v929;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1719 = __riscv_vwmacc_vv_i32m2(v1718, v1528, v1699, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1720 = __riscv_vwmacc_vv_i32m2(v1719, v1532, v1700, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1721 = __riscv_vwmacc_vv_i32m2(v1720, v1536, v1701, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1722 = __riscv_vwmacc_vv_i32m2(v1721, v1540, v1702, 8);
        v929 = v1722;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1723 = __riscv_vfmul_vf_f32m2(v922, v909, 8);
        vint32m2_t v1724 = v923;
        vfloat32m2_t v1725 = v895;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1726 = __riscv_vfcvt_f_x_v_f32m2(v1724, 8);
        vfloat32m2_t v1727 = __riscv_vfmacc_vv_f32m2(v1725, v1726, v1723, 8);
        v895 = v1727;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1728 = __riscv_vfmul_vf_f32m2(v922, v912, 8);
        vint32m2_t v1729 = v925;
        vfloat32m2_t v1730 = v897;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1731 = __riscv_vfcvt_f_x_v_f32m2(v1729, 8);
        vfloat32m2_t v1732 = __riscv_vfmacc_vv_f32m2(v1730, v1731, v1728, 8);
        v897 = v1732;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1733 = __riscv_vfmul_vf_f32m2(v922, v915, 8);
        vint32m2_t v1734 = v927;
        vfloat32m2_t v1735 = v899;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1736 = __riscv_vfcvt_f_x_v_f32m2(v1734, 8);
        vfloat32m2_t v1737 = __riscv_vfmacc_vv_f32m2(v1735, v1736, v1733, 8);
        v899 = v1737;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1738 = __riscv_vfmul_vf_f32m2(v922, v918, 8);
        vint32m2_t v1739 = v929;
        vfloat32m2_t v1740 = v901;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1741 = __riscv_vfcvt_f_x_v_f32m2(v1739, 8);
        vfloat32m2_t v1742 = __riscv_vfmacc_vv_f32m2(v1740, v1741, v1738, 8);
        v901 = v1742;
      }
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1743 = v12 * 4;
      size_t v1744 = v1743 + 0;
      size_t v1745 = v1744 * v7;
      size_t v1746 = v16 * 16;
      size_t v1747 = v1745 + v1746;
      size_t v1748 = v1747 + 8;
      float* v1749 = v2 + v1748;
      vfloat32m2_t v1750 = v895;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1749, v1750, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1751 = v12 * 4;
      size_t v1752 = v1751 + 1;
      size_t v1753 = v1752 * v7;
      size_t v1754 = v16 * 16;
      size_t v1755 = v1753 + v1754;
      size_t v1756 = v1755 + 8;
      float* v1757 = v2 + v1756;
      vfloat32m2_t v1758 = v897;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1757, v1758, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1759 = v12 * 4;
      size_t v1760 = v1759 + 2;
      size_t v1761 = v1760 * v7;
      size_t v1762 = v16 * 16;
      size_t v1763 = v1761 + v1762;
      size_t v1764 = v1763 + 8;
      float* v1765 = v2 + v1764;
      vfloat32m2_t v1766 = v899;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1765, v1766, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1767 = v12 * 4;
      size_t v1768 = v1767 + 3;
      size_t v1769 = v1768 * v7;
      size_t v1770 = v16 * 16;
      size_t v1771 = v1769 + v1770;
      size_t v1772 = v1771 + 8;
      float* v1773 = v2 + v1772;
      vfloat32m2_t v1774 = v901;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1773, v1774, 8);
    }
  }
  return;
}


