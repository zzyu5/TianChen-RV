#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K_ROLLED(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5, size_t v6, size_t v7) {
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
        size_t v37 = v36 * 3360;
        const uint8_t* v38 = v19 + v37;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
        size_t v39 = v36 * 1168;
        const uint8_t* v40 = v15 + v39;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const float* v41 = (const float*) v40;
        float v42 = *(const float *)(v41);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v43 = v40 + 4;
        const float* v44 = (const float*) v43;
        float v45 = *(const float *)(v44);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v46 = v40 + 8;
        const float* v47 = (const float*) v46;
        float v48 = *(const float *)(v47);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
        const uint8_t* v49 = v40 + 12;
        const float* v50 = (const float*) v49;
        float v51 = *(const float *)(v50);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const _Float16* v52 = (const _Float16*) v38;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v53 = __riscv_vle16_v_f16m1(v52, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v54 = __riscv_vfwcvt_f_f_v_f32m2(v53, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const uint8_t* v55 = v38 + 16;
        const _Float16* v56 = (const _Float16*) v55;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v57 = __riscv_vle16_v_f16m1(v56, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v58 = __riscv_vfwcvt_f_f_v_f32m2(v57, 8);
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
        vint32m2_t v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v74 = __riscv_vmv_v_x_i32m2(0, 8);
        v73 = v74;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v75 = v38 + 32;
        const int8_t* v76 = (const int8_t*) v75;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v77 = __riscv_vle8_v_i8mf2(v76, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v78 = __riscv_vsext_vf2_i16m1(v77, 8);
        const uint8_t* v79 = v38 + 64;
        const int8_t* v80 = (const int8_t*) v79;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v81 = __riscv_vle8_v_i8mf2(v80, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v82 = __riscv_vsext_vf2_i16m1(v81, 8);
        const uint8_t* v83 = v38 + 96;
        const int8_t* v84 = (const int8_t*) v83;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v85 = __riscv_vle8_v_i8mf2(v84, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v86 = __riscv_vsext_vf2_i16m1(v85, 8);
        const uint8_t* v87 = v38 + 128;
        const int8_t* v88 = (const int8_t*) v87;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v89 = __riscv_vle8_v_i8mf2(v88, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v90 = __riscv_vsext_vf2_i16m1(v89, 8);
        const uint8_t* v91 = v38 + 40;
        const int8_t* v92 = (const int8_t*) v91;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v93 = __riscv_vle8_v_i8mf2(v92, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v94 = __riscv_vsext_vf2_i16m1(v93, 8);
        const uint8_t* v95 = v38 + 72;
        const int8_t* v96 = (const int8_t*) v95;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v97 = __riscv_vle8_v_i8mf2(v96, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v98 = __riscv_vsext_vf2_i16m1(v97, 8);
        const uint8_t* v99 = v38 + 104;
        const int8_t* v100 = (const int8_t*) v99;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v101 = __riscv_vle8_v_i8mf2(v100, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v102 = __riscv_vsext_vf2_i16m1(v101, 8);
        const uint8_t* v103 = v38 + 136;
        const int8_t* v104 = (const int8_t*) v103;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v105 = __riscv_vle8_v_i8mf2(v104, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v106 = __riscv_vsext_vf2_i16m1(v105, 8);
        vint16m1_t v107;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v108 = __riscv_vmv_v_x_i16m1(0, 8);
        v107 = v108;
        vint16m1_t v109;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v110 = __riscv_vmv_v_x_i16m1(0, 8);
        v109 = v110;
        vint16m1_t v111;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v112 = __riscv_vmv_v_x_i16m1(0, 8);
        v111 = v112;
        vint16m1_t v113;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v114 = __riscv_vmv_v_x_i16m1(0, 8);
        v113 = v114;
        vint16m1_t v115;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v116 = __riscv_vmv_v_x_i16m1(0, 8);
        v115 = v116;
        vint16m1_t v117;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v118 = __riscv_vmv_v_x_i16m1(0, 8);
        v117 = v118;
        vint16m1_t v119;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v120 = __riscv_vmv_v_x_i16m1(0, 8);
        v119 = v120;
        vint16m1_t v121;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v122 = __riscv_vmv_v_x_i16m1(0, 8);
        v121 = v122;
        vint16m1_t v123;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v124 = __riscv_vmv_v_x_i16m1(0, 8);
        v123 = v124;
        vint16m1_t v125;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v126 = __riscv_vmv_v_x_i16m1(0, 8);
        v125 = v126;
        vint16m1_t v127;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v128 = __riscv_vmv_v_x_i16m1(0, 8);
        v127 = v128;
        vint16m1_t v129;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v130 = __riscv_vmv_v_x_i16m1(0, 8);
        v129 = v130;
        vint16m1_t v131;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v132 = __riscv_vmv_v_x_i16m1(0, 8);
        v131 = v132;
        vint16m1_t v133;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v134 = __riscv_vmv_v_x_i16m1(0, 8);
        v133 = v134;
        vint16m1_t v135;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v136 = __riscv_vmv_v_x_i16m1(0, 8);
        v135 = v136;
        vint16m1_t v137;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v138 = __riscv_vmv_v_x_i16m1(0, 8);
        v137 = v138;
        vint16m1_t v139;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v140 = __riscv_vmv_v_x_i16m1(0, 8);
        v139 = v140;
        vint16m1_t v141;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v142 = __riscv_vmv_v_x_i16m1(0, 8);
        v141 = v142;
        vint16m1_t v143;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v144 = __riscv_vmv_v_x_i16m1(0, 8);
        v143 = v144;
        vint16m1_t v145;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v146 = __riscv_vmv_v_x_i16m1(0, 8);
        v145 = v146;
        vint16m1_t v147;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v148 = __riscv_vmv_v_x_i16m1(0, 8);
        v147 = v148;
        vint16m1_t v149;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v150 = __riscv_vmv_v_x_i16m1(0, 8);
        v149 = v150;
        vint16m1_t v151;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v152 = __riscv_vmv_v_x_i16m1(0, 8);
        v151 = v152;
        vint16m1_t v153;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v154 = __riscv_vmv_v_x_i16m1(0, 8);
        v153 = v154;
        vint16m1_t v155;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v156 = __riscv_vmv_v_x_i16m1(0, 8);
        v155 = v156;
        vint16m1_t v157;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v158 = __riscv_vmv_v_x_i16m1(0, 8);
        v157 = v158;
        vint16m1_t v159;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v160 = __riscv_vmv_v_x_i16m1(0, 8);
        v159 = v160;
        vint16m1_t v161;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v162 = __riscv_vmv_v_x_i16m1(0, 8);
        v161 = v162;
        vint16m1_t v163;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v164 = __riscv_vmv_v_x_i16m1(0, 8);
        v163 = v164;
        vint16m1_t v165;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v166 = __riscv_vmv_v_x_i16m1(0, 8);
        v165 = v166;
        vint16m1_t v167;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v168 = __riscv_vmv_v_x_i16m1(0, 8);
        v167 = v168;
        vint16m1_t v169;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v170 = __riscv_vmv_v_x_i16m1(0, 8);
        v169 = v170;
        for (size_t v171 = 0; v171 < 8; v171 += 1) {
          size_t v172 = v171 * 16;
          const uint8_t* v173 = v38 + v172;
          size_t v174 = v171 * 4;
          const uint8_t* v175 = v40 + v174;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v176 = v173 + 1312;
          const uint8_t* v177 = (const uint8_t*) v176;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v178 = __riscv_vle8_v_u8mf2(v177, 8);
          const uint8_t* v179 = v173 + 1824;
          const uint8_t* v180 = (const uint8_t*) v179;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v181 = __riscv_vle8_v_u8mf2(v180, 8);
          const uint8_t* v182 = v173 + 288;
          const uint8_t* v183 = (const uint8_t*) v182;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v184 = __riscv_vle8_v_u8mf2(v183, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v185 = __riscv_vand_vx_u8mf2(v178, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v186 = __riscv_vand_vx_u8mf2(v184, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v187 = __riscv_vsll_vx_u8mf2(v186, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v188 = __riscv_vor_vv_u8mf2(v185, v187, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v189 = __riscv_vreinterpret_v_u8mf2_i8mf2(v188);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v190 = __riscv_vsub_vx_i8mf2(v189, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v191 = __riscv_vand_vx_u8mf2(v181, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v192 = __riscv_vsrl_vx_u8mf2(v184, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v193 = __riscv_vand_vx_u8mf2(v192, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v194 = __riscv_vsll_vx_u8mf2(v193, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v195 = __riscv_vor_vv_u8mf2(v191, v194, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v196 = __riscv_vreinterpret_v_u8mf2_i8mf2(v195);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v197 = __riscv_vsub_vx_i8mf2(v196, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v198 = __riscv_vsrl_vx_u8mf2(v178, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v199 = __riscv_vsrl_vx_u8mf2(v184, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v200 = __riscv_vand_vx_u8mf2(v199, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v201 = __riscv_vsll_vx_u8mf2(v200, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v202 = __riscv_vor_vv_u8mf2(v198, v201, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v203 = __riscv_vreinterpret_v_u8mf2_i8mf2(v202);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v204 = __riscv_vsub_vx_i8mf2(v203, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v205 = __riscv_vsrl_vx_u8mf2(v181, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v206 = __riscv_vsrl_vx_u8mf2(v184, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v207 = __riscv_vand_vx_u8mf2(v206, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v208 = __riscv_vsll_vx_u8mf2(v207, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v209 = __riscv_vor_vv_u8mf2(v205, v208, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v210 = __riscv_vreinterpret_v_u8mf2_i8mf2(v209);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v211 = __riscv_vsub_vx_i8mf2(v210, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v212 = v173 + 1320;
          const uint8_t* v213 = (const uint8_t*) v212;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v214 = __riscv_vle8_v_u8mf2(v213, 8);
          const uint8_t* v215 = v173 + 1832;
          const uint8_t* v216 = (const uint8_t*) v215;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v217 = __riscv_vle8_v_u8mf2(v216, 8);
          const uint8_t* v218 = v173 + 296;
          const uint8_t* v219 = (const uint8_t*) v218;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v220 = __riscv_vle8_v_u8mf2(v219, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v221 = __riscv_vand_vx_u8mf2(v214, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v222 = __riscv_vand_vx_u8mf2(v220, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v223 = __riscv_vsll_vx_u8mf2(v222, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v224 = __riscv_vor_vv_u8mf2(v221, v223, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v225 = __riscv_vreinterpret_v_u8mf2_i8mf2(v224);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v226 = __riscv_vsub_vx_i8mf2(v225, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v227 = __riscv_vand_vx_u8mf2(v217, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v228 = __riscv_vsrl_vx_u8mf2(v220, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v229 = __riscv_vand_vx_u8mf2(v228, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v230 = __riscv_vsll_vx_u8mf2(v229, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v231 = __riscv_vor_vv_u8mf2(v227, v230, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v232 = __riscv_vreinterpret_v_u8mf2_i8mf2(v231);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v233 = __riscv_vsub_vx_i8mf2(v232, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v234 = __riscv_vsrl_vx_u8mf2(v214, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v235 = __riscv_vsrl_vx_u8mf2(v220, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v236 = __riscv_vand_vx_u8mf2(v235, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v237 = __riscv_vsll_vx_u8mf2(v236, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v238 = __riscv_vor_vv_u8mf2(v234, v237, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v239 = __riscv_vreinterpret_v_u8mf2_i8mf2(v238);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v240 = __riscv_vsub_vx_i8mf2(v239, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v241 = __riscv_vsrl_vx_u8mf2(v217, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v242 = __riscv_vsrl_vx_u8mf2(v220, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v243 = __riscv_vand_vx_u8mf2(v242, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v244 = __riscv_vsll_vx_u8mf2(v243, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v245 = __riscv_vor_vv_u8mf2(v241, v244, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v246 = __riscv_vreinterpret_v_u8mf2_i8mf2(v245);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v247 = __riscv_vsub_vx_i8mf2(v246, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v248 = v175 + 16;
          const int8_t* v249 = (const int8_t*) v248;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v250 = *(const int8_t *)(v249);
          vint16m1_t v251 = v107;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v252 = __riscv_vwmacc_vx_i16m1(v251, v250, v190, 8);
          v107 = v252;
          vint16m1_t v253 = v115;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v254 = __riscv_vwmacc_vx_i16m1(v253, v250, v226, 8);
          v115 = v254;
          const uint8_t* v255 = v175 + 144;
          const int8_t* v256 = (const int8_t*) v255;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v257 = *(const int8_t *)(v256);
          vint16m1_t v258 = v109;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v259 = __riscv_vwmacc_vx_i16m1(v258, v257, v197, 8);
          v109 = v259;
          vint16m1_t v260 = v117;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v261 = __riscv_vwmacc_vx_i16m1(v260, v257, v233, 8);
          v117 = v261;
          const uint8_t* v262 = v175 + 272;
          const int8_t* v263 = (const int8_t*) v262;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v264 = *(const int8_t *)(v263);
          vint16m1_t v265 = v111;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v266 = __riscv_vwmacc_vx_i16m1(v265, v264, v204, 8);
          v111 = v266;
          vint16m1_t v267 = v119;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v268 = __riscv_vwmacc_vx_i16m1(v267, v264, v240, 8);
          v119 = v268;
          const uint8_t* v269 = v175 + 400;
          const int8_t* v270 = (const int8_t*) v269;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v271 = *(const int8_t *)(v270);
          vint16m1_t v272 = v113;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v273 = __riscv_vwmacc_vx_i16m1(v272, v271, v211, 8);
          v113 = v273;
          vint16m1_t v274 = v121;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v275 = __riscv_vwmacc_vx_i16m1(v274, v271, v247, 8);
          v121 = v275;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v276 = v175 + 17;
          const int8_t* v277 = (const int8_t*) v276;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v278 = *(const int8_t *)(v277);
          vint16m1_t v279 = v123;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v280 = __riscv_vwmacc_vx_i16m1(v279, v278, v190, 8);
          v123 = v280;
          vint16m1_t v281 = v131;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v282 = __riscv_vwmacc_vx_i16m1(v281, v278, v226, 8);
          v131 = v282;
          const uint8_t* v283 = v175 + 145;
          const int8_t* v284 = (const int8_t*) v283;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v285 = *(const int8_t *)(v284);
          vint16m1_t v286 = v125;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v287 = __riscv_vwmacc_vx_i16m1(v286, v285, v197, 8);
          v125 = v287;
          vint16m1_t v288 = v133;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v289 = __riscv_vwmacc_vx_i16m1(v288, v285, v233, 8);
          v133 = v289;
          const uint8_t* v290 = v175 + 273;
          const int8_t* v291 = (const int8_t*) v290;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v292 = *(const int8_t *)(v291);
          vint16m1_t v293 = v127;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v294 = __riscv_vwmacc_vx_i16m1(v293, v292, v204, 8);
          v127 = v294;
          vint16m1_t v295 = v135;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v296 = __riscv_vwmacc_vx_i16m1(v295, v292, v240, 8);
          v135 = v296;
          const uint8_t* v297 = v175 + 401;
          const int8_t* v298 = (const int8_t*) v297;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v299 = *(const int8_t *)(v298);
          vint16m1_t v300 = v129;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v301 = __riscv_vwmacc_vx_i16m1(v300, v299, v211, 8);
          v129 = v301;
          vint16m1_t v302 = v137;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v303 = __riscv_vwmacc_vx_i16m1(v302, v299, v247, 8);
          v137 = v303;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v304 = v175 + 18;
          const int8_t* v305 = (const int8_t*) v304;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v306 = *(const int8_t *)(v305);
          vint16m1_t v307 = v139;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v308 = __riscv_vwmacc_vx_i16m1(v307, v306, v190, 8);
          v139 = v308;
          vint16m1_t v309 = v147;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v310 = __riscv_vwmacc_vx_i16m1(v309, v306, v226, 8);
          v147 = v310;
          const uint8_t* v311 = v175 + 146;
          const int8_t* v312 = (const int8_t*) v311;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v313 = *(const int8_t *)(v312);
          vint16m1_t v314 = v141;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v315 = __riscv_vwmacc_vx_i16m1(v314, v313, v197, 8);
          v141 = v315;
          vint16m1_t v316 = v149;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v317 = __riscv_vwmacc_vx_i16m1(v316, v313, v233, 8);
          v149 = v317;
          const uint8_t* v318 = v175 + 274;
          const int8_t* v319 = (const int8_t*) v318;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v320 = *(const int8_t *)(v319);
          vint16m1_t v321 = v143;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v322 = __riscv_vwmacc_vx_i16m1(v321, v320, v204, 8);
          v143 = v322;
          vint16m1_t v323 = v151;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v324 = __riscv_vwmacc_vx_i16m1(v323, v320, v240, 8);
          v151 = v324;
          const uint8_t* v325 = v175 + 402;
          const int8_t* v326 = (const int8_t*) v325;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v327 = *(const int8_t *)(v326);
          vint16m1_t v328 = v145;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v329 = __riscv_vwmacc_vx_i16m1(v328, v327, v211, 8);
          v145 = v329;
          vint16m1_t v330 = v153;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v331 = __riscv_vwmacc_vx_i16m1(v330, v327, v247, 8);
          v153 = v331;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v332 = v175 + 19;
          const int8_t* v333 = (const int8_t*) v332;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v334 = *(const int8_t *)(v333);
          vint16m1_t v335 = v155;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v336 = __riscv_vwmacc_vx_i16m1(v335, v334, v190, 8);
          v155 = v336;
          vint16m1_t v337 = v163;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v338 = __riscv_vwmacc_vx_i16m1(v337, v334, v226, 8);
          v163 = v338;
          const uint8_t* v339 = v175 + 147;
          const int8_t* v340 = (const int8_t*) v339;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v341 = *(const int8_t *)(v340);
          vint16m1_t v342 = v157;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v343 = __riscv_vwmacc_vx_i16m1(v342, v341, v197, 8);
          v157 = v343;
          vint16m1_t v344 = v165;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v345 = __riscv_vwmacc_vx_i16m1(v344, v341, v233, 8);
          v165 = v345;
          const uint8_t* v346 = v175 + 275;
          const int8_t* v347 = (const int8_t*) v346;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v348 = *(const int8_t *)(v347);
          vint16m1_t v349 = v159;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v350 = __riscv_vwmacc_vx_i16m1(v349, v348, v204, 8);
          v159 = v350;
          vint16m1_t v351 = v167;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v352 = __riscv_vwmacc_vx_i16m1(v351, v348, v240, 8);
          v167 = v352;
          const uint8_t* v353 = v175 + 403;
          const int8_t* v354 = (const int8_t*) v353;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v355 = *(const int8_t *)(v354);
          vint16m1_t v356 = v161;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v357 = __riscv_vwmacc_vx_i16m1(v356, v355, v211, 8);
          v161 = v357;
          vint16m1_t v358 = v169;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v359 = __riscv_vwmacc_vx_i16m1(v358, v355, v247, 8);
          v169 = v359;
        }
        vint16m1_t v360 = v107;
        vint16m1_t v361 = v109;
        vint16m1_t v362 = v111;
        vint16m1_t v363 = v113;
        vint16m1_t v364 = v115;
        vint16m1_t v365 = v117;
        vint16m1_t v366 = v119;
        vint16m1_t v367 = v121;
        vint16m1_t v368 = v123;
        vint16m1_t v369 = v125;
        vint16m1_t v370 = v127;
        vint16m1_t v371 = v129;
        vint16m1_t v372 = v131;
        vint16m1_t v373 = v133;
        vint16m1_t v374 = v135;
        vint16m1_t v375 = v137;
        vint16m1_t v376 = v139;
        vint16m1_t v377 = v141;
        vint16m1_t v378 = v143;
        vint16m1_t v379 = v145;
        vint16m1_t v380 = v147;
        vint16m1_t v381 = v149;
        vint16m1_t v382 = v151;
        vint16m1_t v383 = v153;
        vint16m1_t v384 = v155;
        vint16m1_t v385 = v157;
        vint16m1_t v386 = v159;
        vint16m1_t v387 = v161;
        vint16m1_t v388 = v163;
        vint16m1_t v389 = v165;
        vint16m1_t v390 = v167;
        vint16m1_t v391 = v169;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v392 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v393 = __riscv_vwmacc_vv_i32m2(v392, v78, v360, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v394 = __riscv_vwmacc_vv_i32m2(v393, v82, v361, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v395 = __riscv_vwmacc_vv_i32m2(v394, v86, v362, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v396 = __riscv_vwmacc_vv_i32m2(v395, v90, v363, 8);
        v59 = v396;
        vint32m2_t v397 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v398 = __riscv_vwmacc_vv_i32m2(v397, v94, v364, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v399 = __riscv_vwmacc_vv_i32m2(v398, v98, v365, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v400 = __riscv_vwmacc_vv_i32m2(v399, v102, v366, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v401 = __riscv_vwmacc_vv_i32m2(v400, v106, v367, 8);
        v61 = v401;
        vint32m2_t v402 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v403 = __riscv_vwmacc_vv_i32m2(v402, v78, v368, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v404 = __riscv_vwmacc_vv_i32m2(v403, v82, v369, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v405 = __riscv_vwmacc_vv_i32m2(v404, v86, v370, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v406 = __riscv_vwmacc_vv_i32m2(v405, v90, v371, 8);
        v63 = v406;
        vint32m2_t v407 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v408 = __riscv_vwmacc_vv_i32m2(v407, v94, v372, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v409 = __riscv_vwmacc_vv_i32m2(v408, v98, v373, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v410 = __riscv_vwmacc_vv_i32m2(v409, v102, v374, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v411 = __riscv_vwmacc_vv_i32m2(v410, v106, v375, 8);
        v65 = v411;
        vint32m2_t v412 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v413 = __riscv_vwmacc_vv_i32m2(v412, v78, v376, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v414 = __riscv_vwmacc_vv_i32m2(v413, v82, v377, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v415 = __riscv_vwmacc_vv_i32m2(v414, v86, v378, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v416 = __riscv_vwmacc_vv_i32m2(v415, v90, v379, 8);
        v67 = v416;
        vint32m2_t v417 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v418 = __riscv_vwmacc_vv_i32m2(v417, v94, v380, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v419 = __riscv_vwmacc_vv_i32m2(v418, v98, v381, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v420 = __riscv_vwmacc_vv_i32m2(v419, v102, v382, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v421 = __riscv_vwmacc_vv_i32m2(v420, v106, v383, 8);
        v69 = v421;
        vint32m2_t v422 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v423 = __riscv_vwmacc_vv_i32m2(v422, v78, v384, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v424 = __riscv_vwmacc_vv_i32m2(v423, v82, v385, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v425 = __riscv_vwmacc_vv_i32m2(v424, v86, v386, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v426 = __riscv_vwmacc_vv_i32m2(v425, v90, v387, 8);
        v71 = v426;
        vint32m2_t v427 = v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v428 = __riscv_vwmacc_vv_i32m2(v427, v94, v388, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v429 = __riscv_vwmacc_vv_i32m2(v428, v98, v389, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v430 = __riscv_vwmacc_vv_i32m2(v429, v102, v390, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v431 = __riscv_vwmacc_vv_i32m2(v430, v106, v391, 8);
        v73 = v431;
        vint16m1_t v432;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v433 = __riscv_vmv_v_x_i16m1(0, 8);
        v432 = v433;
        vint16m1_t v434;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v435 = __riscv_vmv_v_x_i16m1(0, 8);
        v434 = v435;
        vint16m1_t v436;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v437 = __riscv_vmv_v_x_i16m1(0, 8);
        v436 = v437;
        vint16m1_t v438;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v439 = __riscv_vmv_v_x_i16m1(0, 8);
        v438 = v439;
        vint16m1_t v440;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v441 = __riscv_vmv_v_x_i16m1(0, 8);
        v440 = v441;
        vint16m1_t v442;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v443 = __riscv_vmv_v_x_i16m1(0, 8);
        v442 = v443;
        vint16m1_t v444;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v445 = __riscv_vmv_v_x_i16m1(0, 8);
        v444 = v445;
        vint16m1_t v446;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v447 = __riscv_vmv_v_x_i16m1(0, 8);
        v446 = v447;
        vint16m1_t v448;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v449 = __riscv_vmv_v_x_i16m1(0, 8);
        v448 = v449;
        vint16m1_t v450;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v451 = __riscv_vmv_v_x_i16m1(0, 8);
        v450 = v451;
        vint16m1_t v452;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v453 = __riscv_vmv_v_x_i16m1(0, 8);
        v452 = v453;
        vint16m1_t v454;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v455 = __riscv_vmv_v_x_i16m1(0, 8);
        v454 = v455;
        vint16m1_t v456;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v457 = __riscv_vmv_v_x_i16m1(0, 8);
        v456 = v457;
        vint16m1_t v458;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v459 = __riscv_vmv_v_x_i16m1(0, 8);
        v458 = v459;
        vint16m1_t v460;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v461 = __riscv_vmv_v_x_i16m1(0, 8);
        v460 = v461;
        vint16m1_t v462;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v463 = __riscv_vmv_v_x_i16m1(0, 8);
        v462 = v463;
        vint16m1_t v464;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v465 = __riscv_vmv_v_x_i16m1(0, 8);
        v464 = v465;
        vint16m1_t v466;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v467 = __riscv_vmv_v_x_i16m1(0, 8);
        v466 = v467;
        vint16m1_t v468;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v469 = __riscv_vmv_v_x_i16m1(0, 8);
        v468 = v469;
        vint16m1_t v470;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v471 = __riscv_vmv_v_x_i16m1(0, 8);
        v470 = v471;
        vint16m1_t v472;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v473 = __riscv_vmv_v_x_i16m1(0, 8);
        v472 = v473;
        vint16m1_t v474;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v475 = __riscv_vmv_v_x_i16m1(0, 8);
        v474 = v475;
        vint16m1_t v476;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v477 = __riscv_vmv_v_x_i16m1(0, 8);
        v476 = v477;
        vint16m1_t v478;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v479 = __riscv_vmv_v_x_i16m1(0, 8);
        v478 = v479;
        vint16m1_t v480;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v481 = __riscv_vmv_v_x_i16m1(0, 8);
        v480 = v481;
        vint16m1_t v482;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v483 = __riscv_vmv_v_x_i16m1(0, 8);
        v482 = v483;
        vint16m1_t v484;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v485 = __riscv_vmv_v_x_i16m1(0, 8);
        v484 = v485;
        vint16m1_t v486;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v487 = __riscv_vmv_v_x_i16m1(0, 8);
        v486 = v487;
        vint16m1_t v488;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v489 = __riscv_vmv_v_x_i16m1(0, 8);
        v488 = v489;
        vint16m1_t v490;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v491 = __riscv_vmv_v_x_i16m1(0, 8);
        v490 = v491;
        vint16m1_t v492;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v493 = __riscv_vmv_v_x_i16m1(0, 8);
        v492 = v493;
        vint16m1_t v494;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v495 = __riscv_vmv_v_x_i16m1(0, 8);
        v494 = v495;
        for (size_t v496 = 0; v496 < 8; v496 += 1) {
          size_t v497 = v496 * 16;
          const uint8_t* v498 = v38 + v497;
          size_t v499 = v496 * 4;
          const uint8_t* v500 = v40 + v499;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v501 = v498 + 1440;
          const uint8_t* v502 = (const uint8_t*) v501;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v503 = __riscv_vle8_v_u8mf2(v502, 8);
          const uint8_t* v504 = v498 + 1952;
          const uint8_t* v505 = (const uint8_t*) v504;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v506 = __riscv_vle8_v_u8mf2(v505, 8);
          const uint8_t* v507 = v498 + 416;
          const uint8_t* v508 = (const uint8_t*) v507;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v509 = __riscv_vle8_v_u8mf2(v508, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v510 = __riscv_vand_vx_u8mf2(v503, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v511 = __riscv_vand_vx_u8mf2(v509, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v512 = __riscv_vsll_vx_u8mf2(v511, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v513 = __riscv_vor_vv_u8mf2(v510, v512, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v514 = __riscv_vreinterpret_v_u8mf2_i8mf2(v513);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v515 = __riscv_vsub_vx_i8mf2(v514, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v516 = __riscv_vand_vx_u8mf2(v506, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v517 = __riscv_vsrl_vx_u8mf2(v509, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v518 = __riscv_vand_vx_u8mf2(v517, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v519 = __riscv_vsll_vx_u8mf2(v518, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v520 = __riscv_vor_vv_u8mf2(v516, v519, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v521 = __riscv_vreinterpret_v_u8mf2_i8mf2(v520);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v522 = __riscv_vsub_vx_i8mf2(v521, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v523 = __riscv_vsrl_vx_u8mf2(v503, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v524 = __riscv_vsrl_vx_u8mf2(v509, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v525 = __riscv_vand_vx_u8mf2(v524, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v526 = __riscv_vsll_vx_u8mf2(v525, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v527 = __riscv_vor_vv_u8mf2(v523, v526, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v528 = __riscv_vreinterpret_v_u8mf2_i8mf2(v527);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v529 = __riscv_vsub_vx_i8mf2(v528, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v530 = __riscv_vsrl_vx_u8mf2(v506, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v531 = __riscv_vsrl_vx_u8mf2(v509, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v532 = __riscv_vand_vx_u8mf2(v531, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v533 = __riscv_vsll_vx_u8mf2(v532, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v534 = __riscv_vor_vv_u8mf2(v530, v533, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v535 = __riscv_vreinterpret_v_u8mf2_i8mf2(v534);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v536 = __riscv_vsub_vx_i8mf2(v535, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v537 = v498 + 1448;
          const uint8_t* v538 = (const uint8_t*) v537;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v539 = __riscv_vle8_v_u8mf2(v538, 8);
          const uint8_t* v540 = v498 + 1960;
          const uint8_t* v541 = (const uint8_t*) v540;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v542 = __riscv_vle8_v_u8mf2(v541, 8);
          const uint8_t* v543 = v498 + 424;
          const uint8_t* v544 = (const uint8_t*) v543;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v545 = __riscv_vle8_v_u8mf2(v544, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v546 = __riscv_vand_vx_u8mf2(v539, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v547 = __riscv_vand_vx_u8mf2(v545, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v548 = __riscv_vsll_vx_u8mf2(v547, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v549 = __riscv_vor_vv_u8mf2(v546, v548, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v550 = __riscv_vreinterpret_v_u8mf2_i8mf2(v549);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v551 = __riscv_vsub_vx_i8mf2(v550, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v552 = __riscv_vand_vx_u8mf2(v542, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v553 = __riscv_vsrl_vx_u8mf2(v545, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v554 = __riscv_vand_vx_u8mf2(v553, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v555 = __riscv_vsll_vx_u8mf2(v554, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v556 = __riscv_vor_vv_u8mf2(v552, v555, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v557 = __riscv_vreinterpret_v_u8mf2_i8mf2(v556);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v558 = __riscv_vsub_vx_i8mf2(v557, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v559 = __riscv_vsrl_vx_u8mf2(v539, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v560 = __riscv_vsrl_vx_u8mf2(v545, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v561 = __riscv_vand_vx_u8mf2(v560, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v562 = __riscv_vsll_vx_u8mf2(v561, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v563 = __riscv_vor_vv_u8mf2(v559, v562, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v564 = __riscv_vreinterpret_v_u8mf2_i8mf2(v563);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v565 = __riscv_vsub_vx_i8mf2(v564, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v566 = __riscv_vsrl_vx_u8mf2(v542, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v567 = __riscv_vsrl_vx_u8mf2(v545, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v568 = __riscv_vand_vx_u8mf2(v567, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v569 = __riscv_vsll_vx_u8mf2(v568, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v570 = __riscv_vor_vv_u8mf2(v566, v569, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v571 = __riscv_vreinterpret_v_u8mf2_i8mf2(v570);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v572 = __riscv_vsub_vx_i8mf2(v571, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v573 = v500 + 48;
          const int8_t* v574 = (const int8_t*) v573;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v575 = *(const int8_t *)(v574);
          vint16m1_t v576 = v432;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v577 = __riscv_vwmacc_vx_i16m1(v576, v575, v515, 8);
          v432 = v577;
          vint16m1_t v578 = v440;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v579 = __riscv_vwmacc_vx_i16m1(v578, v575, v551, 8);
          v440 = v579;
          const uint8_t* v580 = v500 + 176;
          const int8_t* v581 = (const int8_t*) v580;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v582 = *(const int8_t *)(v581);
          vint16m1_t v583 = v434;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v584 = __riscv_vwmacc_vx_i16m1(v583, v582, v522, 8);
          v434 = v584;
          vint16m1_t v585 = v442;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v586 = __riscv_vwmacc_vx_i16m1(v585, v582, v558, 8);
          v442 = v586;
          const uint8_t* v587 = v500 + 304;
          const int8_t* v588 = (const int8_t*) v587;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v589 = *(const int8_t *)(v588);
          vint16m1_t v590 = v436;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v591 = __riscv_vwmacc_vx_i16m1(v590, v589, v529, 8);
          v436 = v591;
          vint16m1_t v592 = v444;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v593 = __riscv_vwmacc_vx_i16m1(v592, v589, v565, 8);
          v444 = v593;
          const uint8_t* v594 = v500 + 432;
          const int8_t* v595 = (const int8_t*) v594;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v596 = *(const int8_t *)(v595);
          vint16m1_t v597 = v438;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v598 = __riscv_vwmacc_vx_i16m1(v597, v596, v536, 8);
          v438 = v598;
          vint16m1_t v599 = v446;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v600 = __riscv_vwmacc_vx_i16m1(v599, v596, v572, 8);
          v446 = v600;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v601 = v500 + 49;
          const int8_t* v602 = (const int8_t*) v601;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v603 = *(const int8_t *)(v602);
          vint16m1_t v604 = v448;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v605 = __riscv_vwmacc_vx_i16m1(v604, v603, v515, 8);
          v448 = v605;
          vint16m1_t v606 = v456;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v607 = __riscv_vwmacc_vx_i16m1(v606, v603, v551, 8);
          v456 = v607;
          const uint8_t* v608 = v500 + 177;
          const int8_t* v609 = (const int8_t*) v608;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v610 = *(const int8_t *)(v609);
          vint16m1_t v611 = v450;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v612 = __riscv_vwmacc_vx_i16m1(v611, v610, v522, 8);
          v450 = v612;
          vint16m1_t v613 = v458;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v614 = __riscv_vwmacc_vx_i16m1(v613, v610, v558, 8);
          v458 = v614;
          const uint8_t* v615 = v500 + 305;
          const int8_t* v616 = (const int8_t*) v615;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v617 = *(const int8_t *)(v616);
          vint16m1_t v618 = v452;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v619 = __riscv_vwmacc_vx_i16m1(v618, v617, v529, 8);
          v452 = v619;
          vint16m1_t v620 = v460;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v621 = __riscv_vwmacc_vx_i16m1(v620, v617, v565, 8);
          v460 = v621;
          const uint8_t* v622 = v500 + 433;
          const int8_t* v623 = (const int8_t*) v622;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v624 = *(const int8_t *)(v623);
          vint16m1_t v625 = v454;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v626 = __riscv_vwmacc_vx_i16m1(v625, v624, v536, 8);
          v454 = v626;
          vint16m1_t v627 = v462;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v628 = __riscv_vwmacc_vx_i16m1(v627, v624, v572, 8);
          v462 = v628;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v629 = v500 + 50;
          const int8_t* v630 = (const int8_t*) v629;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v631 = *(const int8_t *)(v630);
          vint16m1_t v632 = v464;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v633 = __riscv_vwmacc_vx_i16m1(v632, v631, v515, 8);
          v464 = v633;
          vint16m1_t v634 = v472;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v635 = __riscv_vwmacc_vx_i16m1(v634, v631, v551, 8);
          v472 = v635;
          const uint8_t* v636 = v500 + 178;
          const int8_t* v637 = (const int8_t*) v636;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v638 = *(const int8_t *)(v637);
          vint16m1_t v639 = v466;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v640 = __riscv_vwmacc_vx_i16m1(v639, v638, v522, 8);
          v466 = v640;
          vint16m1_t v641 = v474;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v642 = __riscv_vwmacc_vx_i16m1(v641, v638, v558, 8);
          v474 = v642;
          const uint8_t* v643 = v500 + 306;
          const int8_t* v644 = (const int8_t*) v643;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v645 = *(const int8_t *)(v644);
          vint16m1_t v646 = v468;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v647 = __riscv_vwmacc_vx_i16m1(v646, v645, v529, 8);
          v468 = v647;
          vint16m1_t v648 = v476;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v649 = __riscv_vwmacc_vx_i16m1(v648, v645, v565, 8);
          v476 = v649;
          const uint8_t* v650 = v500 + 434;
          const int8_t* v651 = (const int8_t*) v650;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v652 = *(const int8_t *)(v651);
          vint16m1_t v653 = v470;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v654 = __riscv_vwmacc_vx_i16m1(v653, v652, v536, 8);
          v470 = v654;
          vint16m1_t v655 = v478;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v656 = __riscv_vwmacc_vx_i16m1(v655, v652, v572, 8);
          v478 = v656;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v657 = v500 + 51;
          const int8_t* v658 = (const int8_t*) v657;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v659 = *(const int8_t *)(v658);
          vint16m1_t v660 = v480;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v661 = __riscv_vwmacc_vx_i16m1(v660, v659, v515, 8);
          v480 = v661;
          vint16m1_t v662 = v488;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v663 = __riscv_vwmacc_vx_i16m1(v662, v659, v551, 8);
          v488 = v663;
          const uint8_t* v664 = v500 + 179;
          const int8_t* v665 = (const int8_t*) v664;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v666 = *(const int8_t *)(v665);
          vint16m1_t v667 = v482;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v668 = __riscv_vwmacc_vx_i16m1(v667, v666, v522, 8);
          v482 = v668;
          vint16m1_t v669 = v490;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v670 = __riscv_vwmacc_vx_i16m1(v669, v666, v558, 8);
          v490 = v670;
          const uint8_t* v671 = v500 + 307;
          const int8_t* v672 = (const int8_t*) v671;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v673 = *(const int8_t *)(v672);
          vint16m1_t v674 = v484;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v675 = __riscv_vwmacc_vx_i16m1(v674, v673, v529, 8);
          v484 = v675;
          vint16m1_t v676 = v492;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v677 = __riscv_vwmacc_vx_i16m1(v676, v673, v565, 8);
          v492 = v677;
          const uint8_t* v678 = v500 + 435;
          const int8_t* v679 = (const int8_t*) v678;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v680 = *(const int8_t *)(v679);
          vint16m1_t v681 = v486;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v682 = __riscv_vwmacc_vx_i16m1(v681, v680, v536, 8);
          v486 = v682;
          vint16m1_t v683 = v494;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v684 = __riscv_vwmacc_vx_i16m1(v683, v680, v572, 8);
          v494 = v684;
        }
        vint16m1_t v685 = v432;
        vint16m1_t v686 = v434;
        vint16m1_t v687 = v436;
        vint16m1_t v688 = v438;
        vint16m1_t v689 = v440;
        vint16m1_t v690 = v442;
        vint16m1_t v691 = v444;
        vint16m1_t v692 = v446;
        vint16m1_t v693 = v448;
        vint16m1_t v694 = v450;
        vint16m1_t v695 = v452;
        vint16m1_t v696 = v454;
        vint16m1_t v697 = v456;
        vint16m1_t v698 = v458;
        vint16m1_t v699 = v460;
        vint16m1_t v700 = v462;
        vint16m1_t v701 = v464;
        vint16m1_t v702 = v466;
        vint16m1_t v703 = v468;
        vint16m1_t v704 = v470;
        vint16m1_t v705 = v472;
        vint16m1_t v706 = v474;
        vint16m1_t v707 = v476;
        vint16m1_t v708 = v478;
        vint16m1_t v709 = v480;
        vint16m1_t v710 = v482;
        vint16m1_t v711 = v484;
        vint16m1_t v712 = v486;
        vint16m1_t v713 = v488;
        vint16m1_t v714 = v490;
        vint16m1_t v715 = v492;
        vint16m1_t v716 = v494;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v717 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v718 = __riscv_vwmacc_vv_i32m2(v717, v78, v685, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v719 = __riscv_vwmacc_vv_i32m2(v718, v82, v686, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v720 = __riscv_vwmacc_vv_i32m2(v719, v86, v687, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v721 = __riscv_vwmacc_vv_i32m2(v720, v90, v688, 8);
        v59 = v721;
        vint32m2_t v722 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v723 = __riscv_vwmacc_vv_i32m2(v722, v94, v689, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v724 = __riscv_vwmacc_vv_i32m2(v723, v98, v690, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v725 = __riscv_vwmacc_vv_i32m2(v724, v102, v691, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v726 = __riscv_vwmacc_vv_i32m2(v725, v106, v692, 8);
        v61 = v726;
        vint32m2_t v727 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v728 = __riscv_vwmacc_vv_i32m2(v727, v78, v693, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v729 = __riscv_vwmacc_vv_i32m2(v728, v82, v694, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v730 = __riscv_vwmacc_vv_i32m2(v729, v86, v695, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v731 = __riscv_vwmacc_vv_i32m2(v730, v90, v696, 8);
        v63 = v731;
        vint32m2_t v732 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v733 = __riscv_vwmacc_vv_i32m2(v732, v94, v697, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v734 = __riscv_vwmacc_vv_i32m2(v733, v98, v698, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v735 = __riscv_vwmacc_vv_i32m2(v734, v102, v699, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v736 = __riscv_vwmacc_vv_i32m2(v735, v106, v700, 8);
        v65 = v736;
        vint32m2_t v737 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v738 = __riscv_vwmacc_vv_i32m2(v737, v78, v701, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v739 = __riscv_vwmacc_vv_i32m2(v738, v82, v702, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v740 = __riscv_vwmacc_vv_i32m2(v739, v86, v703, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v741 = __riscv_vwmacc_vv_i32m2(v740, v90, v704, 8);
        v67 = v741;
        vint32m2_t v742 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v743 = __riscv_vwmacc_vv_i32m2(v742, v94, v705, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v744 = __riscv_vwmacc_vv_i32m2(v743, v98, v706, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v745 = __riscv_vwmacc_vv_i32m2(v744, v102, v707, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v746 = __riscv_vwmacc_vv_i32m2(v745, v106, v708, 8);
        v69 = v746;
        vint32m2_t v747 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v748 = __riscv_vwmacc_vv_i32m2(v747, v78, v709, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v749 = __riscv_vwmacc_vv_i32m2(v748, v82, v710, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v750 = __riscv_vwmacc_vv_i32m2(v749, v86, v711, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v751 = __riscv_vwmacc_vv_i32m2(v750, v90, v712, 8);
        v71 = v751;
        vint32m2_t v752 = v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v753 = __riscv_vwmacc_vv_i32m2(v752, v94, v713, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v754 = __riscv_vwmacc_vv_i32m2(v753, v98, v714, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v755 = __riscv_vwmacc_vv_i32m2(v754, v102, v715, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v756 = __riscv_vwmacc_vv_i32m2(v755, v106, v716, 8);
        v73 = v756;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v757 = v38 + 48;
        const int8_t* v758 = (const int8_t*) v757;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v759 = __riscv_vle8_v_i8mf2(v758, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v760 = __riscv_vsext_vf2_i16m1(v759, 8);
        const uint8_t* v761 = v38 + 80;
        const int8_t* v762 = (const int8_t*) v761;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v763 = __riscv_vle8_v_i8mf2(v762, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v764 = __riscv_vsext_vf2_i16m1(v763, 8);
        const uint8_t* v765 = v38 + 112;
        const int8_t* v766 = (const int8_t*) v765;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v767 = __riscv_vle8_v_i8mf2(v766, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v768 = __riscv_vsext_vf2_i16m1(v767, 8);
        const uint8_t* v769 = v38 + 144;
        const int8_t* v770 = (const int8_t*) v769;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v771 = __riscv_vle8_v_i8mf2(v770, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v772 = __riscv_vsext_vf2_i16m1(v771, 8);
        const uint8_t* v773 = v38 + 56;
        const int8_t* v774 = (const int8_t*) v773;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v775 = __riscv_vle8_v_i8mf2(v774, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v776 = __riscv_vsext_vf2_i16m1(v775, 8);
        const uint8_t* v777 = v38 + 88;
        const int8_t* v778 = (const int8_t*) v777;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v779 = __riscv_vle8_v_i8mf2(v778, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v780 = __riscv_vsext_vf2_i16m1(v779, 8);
        const uint8_t* v781 = v38 + 120;
        const int8_t* v782 = (const int8_t*) v781;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v783 = __riscv_vle8_v_i8mf2(v782, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v784 = __riscv_vsext_vf2_i16m1(v783, 8);
        const uint8_t* v785 = v38 + 152;
        const int8_t* v786 = (const int8_t*) v785;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v787 = __riscv_vle8_v_i8mf2(v786, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v788 = __riscv_vsext_vf2_i16m1(v787, 8);
        vint16m1_t v789;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v790 = __riscv_vmv_v_x_i16m1(0, 8);
        v789 = v790;
        vint16m1_t v791;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v792 = __riscv_vmv_v_x_i16m1(0, 8);
        v791 = v792;
        vint16m1_t v793;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v794 = __riscv_vmv_v_x_i16m1(0, 8);
        v793 = v794;
        vint16m1_t v795;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v796 = __riscv_vmv_v_x_i16m1(0, 8);
        v795 = v796;
        vint16m1_t v797;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v798 = __riscv_vmv_v_x_i16m1(0, 8);
        v797 = v798;
        vint16m1_t v799;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v800 = __riscv_vmv_v_x_i16m1(0, 8);
        v799 = v800;
        vint16m1_t v801;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v802 = __riscv_vmv_v_x_i16m1(0, 8);
        v801 = v802;
        vint16m1_t v803;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v804 = __riscv_vmv_v_x_i16m1(0, 8);
        v803 = v804;
        vint16m1_t v805;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v806 = __riscv_vmv_v_x_i16m1(0, 8);
        v805 = v806;
        vint16m1_t v807;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v808 = __riscv_vmv_v_x_i16m1(0, 8);
        v807 = v808;
        vint16m1_t v809;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v810 = __riscv_vmv_v_x_i16m1(0, 8);
        v809 = v810;
        vint16m1_t v811;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v812 = __riscv_vmv_v_x_i16m1(0, 8);
        v811 = v812;
        vint16m1_t v813;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v814 = __riscv_vmv_v_x_i16m1(0, 8);
        v813 = v814;
        vint16m1_t v815;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v816 = __riscv_vmv_v_x_i16m1(0, 8);
        v815 = v816;
        vint16m1_t v817;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v818 = __riscv_vmv_v_x_i16m1(0, 8);
        v817 = v818;
        vint16m1_t v819;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v820 = __riscv_vmv_v_x_i16m1(0, 8);
        v819 = v820;
        vint16m1_t v821;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v822 = __riscv_vmv_v_x_i16m1(0, 8);
        v821 = v822;
        vint16m1_t v823;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v824 = __riscv_vmv_v_x_i16m1(0, 8);
        v823 = v824;
        vint16m1_t v825;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v826 = __riscv_vmv_v_x_i16m1(0, 8);
        v825 = v826;
        vint16m1_t v827;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v828 = __riscv_vmv_v_x_i16m1(0, 8);
        v827 = v828;
        vint16m1_t v829;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v830 = __riscv_vmv_v_x_i16m1(0, 8);
        v829 = v830;
        vint16m1_t v831;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v832 = __riscv_vmv_v_x_i16m1(0, 8);
        v831 = v832;
        vint16m1_t v833;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v834 = __riscv_vmv_v_x_i16m1(0, 8);
        v833 = v834;
        vint16m1_t v835;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v836 = __riscv_vmv_v_x_i16m1(0, 8);
        v835 = v836;
        vint16m1_t v837;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v838 = __riscv_vmv_v_x_i16m1(0, 8);
        v837 = v838;
        vint16m1_t v839;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v840 = __riscv_vmv_v_x_i16m1(0, 8);
        v839 = v840;
        vint16m1_t v841;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v842 = __riscv_vmv_v_x_i16m1(0, 8);
        v841 = v842;
        vint16m1_t v843;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v844 = __riscv_vmv_v_x_i16m1(0, 8);
        v843 = v844;
        vint16m1_t v845;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v846 = __riscv_vmv_v_x_i16m1(0, 8);
        v845 = v846;
        vint16m1_t v847;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v848 = __riscv_vmv_v_x_i16m1(0, 8);
        v847 = v848;
        vint16m1_t v849;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v850 = __riscv_vmv_v_x_i16m1(0, 8);
        v849 = v850;
        vint16m1_t v851;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v852 = __riscv_vmv_v_x_i16m1(0, 8);
        v851 = v852;
        for (size_t v853 = 0; v853 < 8; v853 += 1) {
          size_t v854 = v853 * 16;
          const uint8_t* v855 = v38 + v854;
          size_t v856 = v853 * 4;
          const uint8_t* v857 = v40 + v856;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v858 = v855 + 1568;
          const uint8_t* v859 = (const uint8_t*) v858;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v860 = __riscv_vle8_v_u8mf2(v859, 8);
          const uint8_t* v861 = v855 + 2080;
          const uint8_t* v862 = (const uint8_t*) v861;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v863 = __riscv_vle8_v_u8mf2(v862, 8);
          const uint8_t* v864 = v855 + 544;
          const uint8_t* v865 = (const uint8_t*) v864;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v866 = __riscv_vle8_v_u8mf2(v865, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v867 = __riscv_vand_vx_u8mf2(v860, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v868 = __riscv_vand_vx_u8mf2(v866, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v869 = __riscv_vsll_vx_u8mf2(v868, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v870 = __riscv_vor_vv_u8mf2(v867, v869, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v871 = __riscv_vreinterpret_v_u8mf2_i8mf2(v870);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v872 = __riscv_vsub_vx_i8mf2(v871, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v873 = __riscv_vand_vx_u8mf2(v863, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v874 = __riscv_vsrl_vx_u8mf2(v866, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v875 = __riscv_vand_vx_u8mf2(v874, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v876 = __riscv_vsll_vx_u8mf2(v875, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v877 = __riscv_vor_vv_u8mf2(v873, v876, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v878 = __riscv_vreinterpret_v_u8mf2_i8mf2(v877);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v879 = __riscv_vsub_vx_i8mf2(v878, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v880 = __riscv_vsrl_vx_u8mf2(v860, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v881 = __riscv_vsrl_vx_u8mf2(v866, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v882 = __riscv_vand_vx_u8mf2(v881, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v883 = __riscv_vsll_vx_u8mf2(v882, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v884 = __riscv_vor_vv_u8mf2(v880, v883, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v885 = __riscv_vreinterpret_v_u8mf2_i8mf2(v884);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v886 = __riscv_vsub_vx_i8mf2(v885, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v887 = __riscv_vsrl_vx_u8mf2(v863, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v888 = __riscv_vsrl_vx_u8mf2(v866, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v889 = __riscv_vand_vx_u8mf2(v888, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v890 = __riscv_vsll_vx_u8mf2(v889, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v891 = __riscv_vor_vv_u8mf2(v887, v890, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v892 = __riscv_vreinterpret_v_u8mf2_i8mf2(v891);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v893 = __riscv_vsub_vx_i8mf2(v892, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v894 = v855 + 1576;
          const uint8_t* v895 = (const uint8_t*) v894;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v896 = __riscv_vle8_v_u8mf2(v895, 8);
          const uint8_t* v897 = v855 + 2088;
          const uint8_t* v898 = (const uint8_t*) v897;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v899 = __riscv_vle8_v_u8mf2(v898, 8);
          const uint8_t* v900 = v855 + 552;
          const uint8_t* v901 = (const uint8_t*) v900;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v902 = __riscv_vle8_v_u8mf2(v901, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v903 = __riscv_vand_vx_u8mf2(v896, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v904 = __riscv_vand_vx_u8mf2(v902, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v905 = __riscv_vsll_vx_u8mf2(v904, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v906 = __riscv_vor_vv_u8mf2(v903, v905, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v907 = __riscv_vreinterpret_v_u8mf2_i8mf2(v906);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v908 = __riscv_vsub_vx_i8mf2(v907, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v909 = __riscv_vand_vx_u8mf2(v899, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v910 = __riscv_vsrl_vx_u8mf2(v902, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v911 = __riscv_vand_vx_u8mf2(v910, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v912 = __riscv_vsll_vx_u8mf2(v911, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v913 = __riscv_vor_vv_u8mf2(v909, v912, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v914 = __riscv_vreinterpret_v_u8mf2_i8mf2(v913);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v915 = __riscv_vsub_vx_i8mf2(v914, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v916 = __riscv_vsrl_vx_u8mf2(v896, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v917 = __riscv_vsrl_vx_u8mf2(v902, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v918 = __riscv_vand_vx_u8mf2(v917, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v919 = __riscv_vsll_vx_u8mf2(v918, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v920 = __riscv_vor_vv_u8mf2(v916, v919, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v921 = __riscv_vreinterpret_v_u8mf2_i8mf2(v920);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v922 = __riscv_vsub_vx_i8mf2(v921, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v923 = __riscv_vsrl_vx_u8mf2(v899, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v924 = __riscv_vsrl_vx_u8mf2(v902, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v925 = __riscv_vand_vx_u8mf2(v924, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v926 = __riscv_vsll_vx_u8mf2(v925, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v927 = __riscv_vor_vv_u8mf2(v923, v926, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v928 = __riscv_vreinterpret_v_u8mf2_i8mf2(v927);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v929 = __riscv_vsub_vx_i8mf2(v928, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v930 = v857 + 80;
          const int8_t* v931 = (const int8_t*) v930;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v932 = *(const int8_t *)(v931);
          vint16m1_t v933 = v789;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v934 = __riscv_vwmacc_vx_i16m1(v933, v932, v872, 8);
          v789 = v934;
          vint16m1_t v935 = v797;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v936 = __riscv_vwmacc_vx_i16m1(v935, v932, v908, 8);
          v797 = v936;
          const uint8_t* v937 = v857 + 208;
          const int8_t* v938 = (const int8_t*) v937;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v939 = *(const int8_t *)(v938);
          vint16m1_t v940 = v791;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v941 = __riscv_vwmacc_vx_i16m1(v940, v939, v879, 8);
          v791 = v941;
          vint16m1_t v942 = v799;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v943 = __riscv_vwmacc_vx_i16m1(v942, v939, v915, 8);
          v799 = v943;
          const uint8_t* v944 = v857 + 336;
          const int8_t* v945 = (const int8_t*) v944;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v946 = *(const int8_t *)(v945);
          vint16m1_t v947 = v793;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v948 = __riscv_vwmacc_vx_i16m1(v947, v946, v886, 8);
          v793 = v948;
          vint16m1_t v949 = v801;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v950 = __riscv_vwmacc_vx_i16m1(v949, v946, v922, 8);
          v801 = v950;
          const uint8_t* v951 = v857 + 464;
          const int8_t* v952 = (const int8_t*) v951;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v953 = *(const int8_t *)(v952);
          vint16m1_t v954 = v795;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v955 = __riscv_vwmacc_vx_i16m1(v954, v953, v893, 8);
          v795 = v955;
          vint16m1_t v956 = v803;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v957 = __riscv_vwmacc_vx_i16m1(v956, v953, v929, 8);
          v803 = v957;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v958 = v857 + 81;
          const int8_t* v959 = (const int8_t*) v958;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v960 = *(const int8_t *)(v959);
          vint16m1_t v961 = v805;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v962 = __riscv_vwmacc_vx_i16m1(v961, v960, v872, 8);
          v805 = v962;
          vint16m1_t v963 = v813;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v964 = __riscv_vwmacc_vx_i16m1(v963, v960, v908, 8);
          v813 = v964;
          const uint8_t* v965 = v857 + 209;
          const int8_t* v966 = (const int8_t*) v965;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v967 = *(const int8_t *)(v966);
          vint16m1_t v968 = v807;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v969 = __riscv_vwmacc_vx_i16m1(v968, v967, v879, 8);
          v807 = v969;
          vint16m1_t v970 = v815;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v971 = __riscv_vwmacc_vx_i16m1(v970, v967, v915, 8);
          v815 = v971;
          const uint8_t* v972 = v857 + 337;
          const int8_t* v973 = (const int8_t*) v972;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v974 = *(const int8_t *)(v973);
          vint16m1_t v975 = v809;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v976 = __riscv_vwmacc_vx_i16m1(v975, v974, v886, 8);
          v809 = v976;
          vint16m1_t v977 = v817;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v978 = __riscv_vwmacc_vx_i16m1(v977, v974, v922, 8);
          v817 = v978;
          const uint8_t* v979 = v857 + 465;
          const int8_t* v980 = (const int8_t*) v979;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v981 = *(const int8_t *)(v980);
          vint16m1_t v982 = v811;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v983 = __riscv_vwmacc_vx_i16m1(v982, v981, v893, 8);
          v811 = v983;
          vint16m1_t v984 = v819;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v985 = __riscv_vwmacc_vx_i16m1(v984, v981, v929, 8);
          v819 = v985;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v986 = v857 + 82;
          const int8_t* v987 = (const int8_t*) v986;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v988 = *(const int8_t *)(v987);
          vint16m1_t v989 = v821;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v990 = __riscv_vwmacc_vx_i16m1(v989, v988, v872, 8);
          v821 = v990;
          vint16m1_t v991 = v829;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v992 = __riscv_vwmacc_vx_i16m1(v991, v988, v908, 8);
          v829 = v992;
          const uint8_t* v993 = v857 + 210;
          const int8_t* v994 = (const int8_t*) v993;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v995 = *(const int8_t *)(v994);
          vint16m1_t v996 = v823;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v997 = __riscv_vwmacc_vx_i16m1(v996, v995, v879, 8);
          v823 = v997;
          vint16m1_t v998 = v831;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v999 = __riscv_vwmacc_vx_i16m1(v998, v995, v915, 8);
          v831 = v999;
          const uint8_t* v1000 = v857 + 338;
          const int8_t* v1001 = (const int8_t*) v1000;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1002 = *(const int8_t *)(v1001);
          vint16m1_t v1003 = v825;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1004 = __riscv_vwmacc_vx_i16m1(v1003, v1002, v886, 8);
          v825 = v1004;
          vint16m1_t v1005 = v833;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1006 = __riscv_vwmacc_vx_i16m1(v1005, v1002, v922, 8);
          v833 = v1006;
          const uint8_t* v1007 = v857 + 466;
          const int8_t* v1008 = (const int8_t*) v1007;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1009 = *(const int8_t *)(v1008);
          vint16m1_t v1010 = v827;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1011 = __riscv_vwmacc_vx_i16m1(v1010, v1009, v893, 8);
          v827 = v1011;
          vint16m1_t v1012 = v835;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1013 = __riscv_vwmacc_vx_i16m1(v1012, v1009, v929, 8);
          v835 = v1013;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1014 = v857 + 83;
          const int8_t* v1015 = (const int8_t*) v1014;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1016 = *(const int8_t *)(v1015);
          vint16m1_t v1017 = v837;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1018 = __riscv_vwmacc_vx_i16m1(v1017, v1016, v872, 8);
          v837 = v1018;
          vint16m1_t v1019 = v845;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1020 = __riscv_vwmacc_vx_i16m1(v1019, v1016, v908, 8);
          v845 = v1020;
          const uint8_t* v1021 = v857 + 211;
          const int8_t* v1022 = (const int8_t*) v1021;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1023 = *(const int8_t *)(v1022);
          vint16m1_t v1024 = v839;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1025 = __riscv_vwmacc_vx_i16m1(v1024, v1023, v879, 8);
          v839 = v1025;
          vint16m1_t v1026 = v847;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1027 = __riscv_vwmacc_vx_i16m1(v1026, v1023, v915, 8);
          v847 = v1027;
          const uint8_t* v1028 = v857 + 339;
          const int8_t* v1029 = (const int8_t*) v1028;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1030 = *(const int8_t *)(v1029);
          vint16m1_t v1031 = v841;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1032 = __riscv_vwmacc_vx_i16m1(v1031, v1030, v886, 8);
          v841 = v1032;
          vint16m1_t v1033 = v849;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1034 = __riscv_vwmacc_vx_i16m1(v1033, v1030, v922, 8);
          v849 = v1034;
          const uint8_t* v1035 = v857 + 467;
          const int8_t* v1036 = (const int8_t*) v1035;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1037 = *(const int8_t *)(v1036);
          vint16m1_t v1038 = v843;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1039 = __riscv_vwmacc_vx_i16m1(v1038, v1037, v893, 8);
          v843 = v1039;
          vint16m1_t v1040 = v851;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1041 = __riscv_vwmacc_vx_i16m1(v1040, v1037, v929, 8);
          v851 = v1041;
        }
        vint16m1_t v1042 = v789;
        vint16m1_t v1043 = v791;
        vint16m1_t v1044 = v793;
        vint16m1_t v1045 = v795;
        vint16m1_t v1046 = v797;
        vint16m1_t v1047 = v799;
        vint16m1_t v1048 = v801;
        vint16m1_t v1049 = v803;
        vint16m1_t v1050 = v805;
        vint16m1_t v1051 = v807;
        vint16m1_t v1052 = v809;
        vint16m1_t v1053 = v811;
        vint16m1_t v1054 = v813;
        vint16m1_t v1055 = v815;
        vint16m1_t v1056 = v817;
        vint16m1_t v1057 = v819;
        vint16m1_t v1058 = v821;
        vint16m1_t v1059 = v823;
        vint16m1_t v1060 = v825;
        vint16m1_t v1061 = v827;
        vint16m1_t v1062 = v829;
        vint16m1_t v1063 = v831;
        vint16m1_t v1064 = v833;
        vint16m1_t v1065 = v835;
        vint16m1_t v1066 = v837;
        vint16m1_t v1067 = v839;
        vint16m1_t v1068 = v841;
        vint16m1_t v1069 = v843;
        vint16m1_t v1070 = v845;
        vint16m1_t v1071 = v847;
        vint16m1_t v1072 = v849;
        vint16m1_t v1073 = v851;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v1074 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1075 = __riscv_vwmacc_vv_i32m2(v1074, v760, v1042, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1076 = __riscv_vwmacc_vv_i32m2(v1075, v764, v1043, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1077 = __riscv_vwmacc_vv_i32m2(v1076, v768, v1044, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1078 = __riscv_vwmacc_vv_i32m2(v1077, v772, v1045, 8);
        v59 = v1078;
        vint32m2_t v1079 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1080 = __riscv_vwmacc_vv_i32m2(v1079, v776, v1046, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1081 = __riscv_vwmacc_vv_i32m2(v1080, v780, v1047, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1082 = __riscv_vwmacc_vv_i32m2(v1081, v784, v1048, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1083 = __riscv_vwmacc_vv_i32m2(v1082, v788, v1049, 8);
        v61 = v1083;
        vint32m2_t v1084 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1085 = __riscv_vwmacc_vv_i32m2(v1084, v760, v1050, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1086 = __riscv_vwmacc_vv_i32m2(v1085, v764, v1051, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1087 = __riscv_vwmacc_vv_i32m2(v1086, v768, v1052, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1088 = __riscv_vwmacc_vv_i32m2(v1087, v772, v1053, 8);
        v63 = v1088;
        vint32m2_t v1089 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1090 = __riscv_vwmacc_vv_i32m2(v1089, v776, v1054, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1091 = __riscv_vwmacc_vv_i32m2(v1090, v780, v1055, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1092 = __riscv_vwmacc_vv_i32m2(v1091, v784, v1056, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1093 = __riscv_vwmacc_vv_i32m2(v1092, v788, v1057, 8);
        v65 = v1093;
        vint32m2_t v1094 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1095 = __riscv_vwmacc_vv_i32m2(v1094, v760, v1058, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1096 = __riscv_vwmacc_vv_i32m2(v1095, v764, v1059, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1097 = __riscv_vwmacc_vv_i32m2(v1096, v768, v1060, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1098 = __riscv_vwmacc_vv_i32m2(v1097, v772, v1061, 8);
        v67 = v1098;
        vint32m2_t v1099 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1100 = __riscv_vwmacc_vv_i32m2(v1099, v776, v1062, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1101 = __riscv_vwmacc_vv_i32m2(v1100, v780, v1063, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1102 = __riscv_vwmacc_vv_i32m2(v1101, v784, v1064, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1103 = __riscv_vwmacc_vv_i32m2(v1102, v788, v1065, 8);
        v69 = v1103;
        vint32m2_t v1104 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1105 = __riscv_vwmacc_vv_i32m2(v1104, v760, v1066, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1106 = __riscv_vwmacc_vv_i32m2(v1105, v764, v1067, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1107 = __riscv_vwmacc_vv_i32m2(v1106, v768, v1068, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1108 = __riscv_vwmacc_vv_i32m2(v1107, v772, v1069, 8);
        v71 = v1108;
        vint32m2_t v1109 = v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1110 = __riscv_vwmacc_vv_i32m2(v1109, v776, v1070, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1111 = __riscv_vwmacc_vv_i32m2(v1110, v780, v1071, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1112 = __riscv_vwmacc_vv_i32m2(v1111, v784, v1072, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1113 = __riscv_vwmacc_vv_i32m2(v1112, v788, v1073, 8);
        v73 = v1113;
        vint16m1_t v1114;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1115 = __riscv_vmv_v_x_i16m1(0, 8);
        v1114 = v1115;
        vint16m1_t v1116;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1117 = __riscv_vmv_v_x_i16m1(0, 8);
        v1116 = v1117;
        vint16m1_t v1118;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1119 = __riscv_vmv_v_x_i16m1(0, 8);
        v1118 = v1119;
        vint16m1_t v1120;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1121 = __riscv_vmv_v_x_i16m1(0, 8);
        v1120 = v1121;
        vint16m1_t v1122;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1123 = __riscv_vmv_v_x_i16m1(0, 8);
        v1122 = v1123;
        vint16m1_t v1124;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1125 = __riscv_vmv_v_x_i16m1(0, 8);
        v1124 = v1125;
        vint16m1_t v1126;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1127 = __riscv_vmv_v_x_i16m1(0, 8);
        v1126 = v1127;
        vint16m1_t v1128;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1129 = __riscv_vmv_v_x_i16m1(0, 8);
        v1128 = v1129;
        vint16m1_t v1130;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1131 = __riscv_vmv_v_x_i16m1(0, 8);
        v1130 = v1131;
        vint16m1_t v1132;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1133 = __riscv_vmv_v_x_i16m1(0, 8);
        v1132 = v1133;
        vint16m1_t v1134;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1135 = __riscv_vmv_v_x_i16m1(0, 8);
        v1134 = v1135;
        vint16m1_t v1136;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1137 = __riscv_vmv_v_x_i16m1(0, 8);
        v1136 = v1137;
        vint16m1_t v1138;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1139 = __riscv_vmv_v_x_i16m1(0, 8);
        v1138 = v1139;
        vint16m1_t v1140;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1141 = __riscv_vmv_v_x_i16m1(0, 8);
        v1140 = v1141;
        vint16m1_t v1142;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1143 = __riscv_vmv_v_x_i16m1(0, 8);
        v1142 = v1143;
        vint16m1_t v1144;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1145 = __riscv_vmv_v_x_i16m1(0, 8);
        v1144 = v1145;
        vint16m1_t v1146;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1147 = __riscv_vmv_v_x_i16m1(0, 8);
        v1146 = v1147;
        vint16m1_t v1148;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1149 = __riscv_vmv_v_x_i16m1(0, 8);
        v1148 = v1149;
        vint16m1_t v1150;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1151 = __riscv_vmv_v_x_i16m1(0, 8);
        v1150 = v1151;
        vint16m1_t v1152;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1153 = __riscv_vmv_v_x_i16m1(0, 8);
        v1152 = v1153;
        vint16m1_t v1154;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1155 = __riscv_vmv_v_x_i16m1(0, 8);
        v1154 = v1155;
        vint16m1_t v1156;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1157 = __riscv_vmv_v_x_i16m1(0, 8);
        v1156 = v1157;
        vint16m1_t v1158;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1159 = __riscv_vmv_v_x_i16m1(0, 8);
        v1158 = v1159;
        vint16m1_t v1160;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1161 = __riscv_vmv_v_x_i16m1(0, 8);
        v1160 = v1161;
        vint16m1_t v1162;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1163 = __riscv_vmv_v_x_i16m1(0, 8);
        v1162 = v1163;
        vint16m1_t v1164;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1165 = __riscv_vmv_v_x_i16m1(0, 8);
        v1164 = v1165;
        vint16m1_t v1166;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1167 = __riscv_vmv_v_x_i16m1(0, 8);
        v1166 = v1167;
        vint16m1_t v1168;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1169 = __riscv_vmv_v_x_i16m1(0, 8);
        v1168 = v1169;
        vint16m1_t v1170;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1171 = __riscv_vmv_v_x_i16m1(0, 8);
        v1170 = v1171;
        vint16m1_t v1172;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1173 = __riscv_vmv_v_x_i16m1(0, 8);
        v1172 = v1173;
        vint16m1_t v1174;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1175 = __riscv_vmv_v_x_i16m1(0, 8);
        v1174 = v1175;
        vint16m1_t v1176;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1177 = __riscv_vmv_v_x_i16m1(0, 8);
        v1176 = v1177;
        for (size_t v1178 = 0; v1178 < 8; v1178 += 1) {
          size_t v1179 = v1178 * 16;
          const uint8_t* v1180 = v38 + v1179;
          size_t v1181 = v1178 * 4;
          const uint8_t* v1182 = v40 + v1181;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v1183 = v1180 + 1696;
          const uint8_t* v1184 = (const uint8_t*) v1183;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1185 = __riscv_vle8_v_u8mf2(v1184, 8);
          const uint8_t* v1186 = v1180 + 2208;
          const uint8_t* v1187 = (const uint8_t*) v1186;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1188 = __riscv_vle8_v_u8mf2(v1187, 8);
          const uint8_t* v1189 = v1180 + 672;
          const uint8_t* v1190 = (const uint8_t*) v1189;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1191 = __riscv_vle8_v_u8mf2(v1190, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1192 = __riscv_vand_vx_u8mf2(v1185, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1193 = __riscv_vand_vx_u8mf2(v1191, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1194 = __riscv_vsll_vx_u8mf2(v1193, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1195 = __riscv_vor_vv_u8mf2(v1192, v1194, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1196 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1195);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1197 = __riscv_vsub_vx_i8mf2(v1196, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1198 = __riscv_vand_vx_u8mf2(v1188, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1199 = __riscv_vsrl_vx_u8mf2(v1191, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1200 = __riscv_vand_vx_u8mf2(v1199, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1201 = __riscv_vsll_vx_u8mf2(v1200, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1202 = __riscv_vor_vv_u8mf2(v1198, v1201, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1203 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1202);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1204 = __riscv_vsub_vx_i8mf2(v1203, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1205 = __riscv_vsrl_vx_u8mf2(v1185, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1206 = __riscv_vsrl_vx_u8mf2(v1191, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1207 = __riscv_vand_vx_u8mf2(v1206, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1208 = __riscv_vsll_vx_u8mf2(v1207, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1209 = __riscv_vor_vv_u8mf2(v1205, v1208, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1210 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1209);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1211 = __riscv_vsub_vx_i8mf2(v1210, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1212 = __riscv_vsrl_vx_u8mf2(v1188, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1213 = __riscv_vsrl_vx_u8mf2(v1191, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1214 = __riscv_vand_vx_u8mf2(v1213, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1215 = __riscv_vsll_vx_u8mf2(v1214, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1216 = __riscv_vor_vv_u8mf2(v1212, v1215, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1217 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1216);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1218 = __riscv_vsub_vx_i8mf2(v1217, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v1219 = v1180 + 1704;
          const uint8_t* v1220 = (const uint8_t*) v1219;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1221 = __riscv_vle8_v_u8mf2(v1220, 8);
          const uint8_t* v1222 = v1180 + 2216;
          const uint8_t* v1223 = (const uint8_t*) v1222;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1224 = __riscv_vle8_v_u8mf2(v1223, 8);
          const uint8_t* v1225 = v1180 + 680;
          const uint8_t* v1226 = (const uint8_t*) v1225;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1227 = __riscv_vle8_v_u8mf2(v1226, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1228 = __riscv_vand_vx_u8mf2(v1221, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1229 = __riscv_vand_vx_u8mf2(v1227, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1230 = __riscv_vsll_vx_u8mf2(v1229, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1231 = __riscv_vor_vv_u8mf2(v1228, v1230, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1232 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1231);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1233 = __riscv_vsub_vx_i8mf2(v1232, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1234 = __riscv_vand_vx_u8mf2(v1224, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1235 = __riscv_vsrl_vx_u8mf2(v1227, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1236 = __riscv_vand_vx_u8mf2(v1235, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1237 = __riscv_vsll_vx_u8mf2(v1236, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1238 = __riscv_vor_vv_u8mf2(v1234, v1237, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1239 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1238);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1240 = __riscv_vsub_vx_i8mf2(v1239, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1241 = __riscv_vsrl_vx_u8mf2(v1221, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1242 = __riscv_vsrl_vx_u8mf2(v1227, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1243 = __riscv_vand_vx_u8mf2(v1242, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1244 = __riscv_vsll_vx_u8mf2(v1243, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1245 = __riscv_vor_vv_u8mf2(v1241, v1244, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1246 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1245);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1247 = __riscv_vsub_vx_i8mf2(v1246, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1248 = __riscv_vsrl_vx_u8mf2(v1224, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1249 = __riscv_vsrl_vx_u8mf2(v1227, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1250 = __riscv_vand_vx_u8mf2(v1249, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1251 = __riscv_vsll_vx_u8mf2(v1250, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1252 = __riscv_vor_vv_u8mf2(v1248, v1251, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1253 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1252);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1254 = __riscv_vsub_vx_i8mf2(v1253, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1255 = v1182 + 112;
          const int8_t* v1256 = (const int8_t*) v1255;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1257 = *(const int8_t *)(v1256);
          vint16m1_t v1258 = v1114;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1259 = __riscv_vwmacc_vx_i16m1(v1258, v1257, v1197, 8);
          v1114 = v1259;
          vint16m1_t v1260 = v1122;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1261 = __riscv_vwmacc_vx_i16m1(v1260, v1257, v1233, 8);
          v1122 = v1261;
          const uint8_t* v1262 = v1182 + 240;
          const int8_t* v1263 = (const int8_t*) v1262;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1264 = *(const int8_t *)(v1263);
          vint16m1_t v1265 = v1116;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1266 = __riscv_vwmacc_vx_i16m1(v1265, v1264, v1204, 8);
          v1116 = v1266;
          vint16m1_t v1267 = v1124;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1268 = __riscv_vwmacc_vx_i16m1(v1267, v1264, v1240, 8);
          v1124 = v1268;
          const uint8_t* v1269 = v1182 + 368;
          const int8_t* v1270 = (const int8_t*) v1269;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1271 = *(const int8_t *)(v1270);
          vint16m1_t v1272 = v1118;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1273 = __riscv_vwmacc_vx_i16m1(v1272, v1271, v1211, 8);
          v1118 = v1273;
          vint16m1_t v1274 = v1126;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1275 = __riscv_vwmacc_vx_i16m1(v1274, v1271, v1247, 8);
          v1126 = v1275;
          const uint8_t* v1276 = v1182 + 496;
          const int8_t* v1277 = (const int8_t*) v1276;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1278 = *(const int8_t *)(v1277);
          vint16m1_t v1279 = v1120;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1280 = __riscv_vwmacc_vx_i16m1(v1279, v1278, v1218, 8);
          v1120 = v1280;
          vint16m1_t v1281 = v1128;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1282 = __riscv_vwmacc_vx_i16m1(v1281, v1278, v1254, 8);
          v1128 = v1282;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1283 = v1182 + 113;
          const int8_t* v1284 = (const int8_t*) v1283;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1285 = *(const int8_t *)(v1284);
          vint16m1_t v1286 = v1130;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1287 = __riscv_vwmacc_vx_i16m1(v1286, v1285, v1197, 8);
          v1130 = v1287;
          vint16m1_t v1288 = v1138;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1289 = __riscv_vwmacc_vx_i16m1(v1288, v1285, v1233, 8);
          v1138 = v1289;
          const uint8_t* v1290 = v1182 + 241;
          const int8_t* v1291 = (const int8_t*) v1290;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1292 = *(const int8_t *)(v1291);
          vint16m1_t v1293 = v1132;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1294 = __riscv_vwmacc_vx_i16m1(v1293, v1292, v1204, 8);
          v1132 = v1294;
          vint16m1_t v1295 = v1140;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1296 = __riscv_vwmacc_vx_i16m1(v1295, v1292, v1240, 8);
          v1140 = v1296;
          const uint8_t* v1297 = v1182 + 369;
          const int8_t* v1298 = (const int8_t*) v1297;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1299 = *(const int8_t *)(v1298);
          vint16m1_t v1300 = v1134;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1301 = __riscv_vwmacc_vx_i16m1(v1300, v1299, v1211, 8);
          v1134 = v1301;
          vint16m1_t v1302 = v1142;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1303 = __riscv_vwmacc_vx_i16m1(v1302, v1299, v1247, 8);
          v1142 = v1303;
          const uint8_t* v1304 = v1182 + 497;
          const int8_t* v1305 = (const int8_t*) v1304;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1306 = *(const int8_t *)(v1305);
          vint16m1_t v1307 = v1136;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1308 = __riscv_vwmacc_vx_i16m1(v1307, v1306, v1218, 8);
          v1136 = v1308;
          vint16m1_t v1309 = v1144;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1310 = __riscv_vwmacc_vx_i16m1(v1309, v1306, v1254, 8);
          v1144 = v1310;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1311 = v1182 + 114;
          const int8_t* v1312 = (const int8_t*) v1311;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1313 = *(const int8_t *)(v1312);
          vint16m1_t v1314 = v1146;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1315 = __riscv_vwmacc_vx_i16m1(v1314, v1313, v1197, 8);
          v1146 = v1315;
          vint16m1_t v1316 = v1154;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1317 = __riscv_vwmacc_vx_i16m1(v1316, v1313, v1233, 8);
          v1154 = v1317;
          const uint8_t* v1318 = v1182 + 242;
          const int8_t* v1319 = (const int8_t*) v1318;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1320 = *(const int8_t *)(v1319);
          vint16m1_t v1321 = v1148;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1322 = __riscv_vwmacc_vx_i16m1(v1321, v1320, v1204, 8);
          v1148 = v1322;
          vint16m1_t v1323 = v1156;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1324 = __riscv_vwmacc_vx_i16m1(v1323, v1320, v1240, 8);
          v1156 = v1324;
          const uint8_t* v1325 = v1182 + 370;
          const int8_t* v1326 = (const int8_t*) v1325;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1327 = *(const int8_t *)(v1326);
          vint16m1_t v1328 = v1150;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1329 = __riscv_vwmacc_vx_i16m1(v1328, v1327, v1211, 8);
          v1150 = v1329;
          vint16m1_t v1330 = v1158;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1331 = __riscv_vwmacc_vx_i16m1(v1330, v1327, v1247, 8);
          v1158 = v1331;
          const uint8_t* v1332 = v1182 + 498;
          const int8_t* v1333 = (const int8_t*) v1332;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1334 = *(const int8_t *)(v1333);
          vint16m1_t v1335 = v1152;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1336 = __riscv_vwmacc_vx_i16m1(v1335, v1334, v1218, 8);
          v1152 = v1336;
          vint16m1_t v1337 = v1160;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1338 = __riscv_vwmacc_vx_i16m1(v1337, v1334, v1254, 8);
          v1160 = v1338;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1339 = v1182 + 115;
          const int8_t* v1340 = (const int8_t*) v1339;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1341 = *(const int8_t *)(v1340);
          vint16m1_t v1342 = v1162;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1343 = __riscv_vwmacc_vx_i16m1(v1342, v1341, v1197, 8);
          v1162 = v1343;
          vint16m1_t v1344 = v1170;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1345 = __riscv_vwmacc_vx_i16m1(v1344, v1341, v1233, 8);
          v1170 = v1345;
          const uint8_t* v1346 = v1182 + 243;
          const int8_t* v1347 = (const int8_t*) v1346;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1348 = *(const int8_t *)(v1347);
          vint16m1_t v1349 = v1164;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1350 = __riscv_vwmacc_vx_i16m1(v1349, v1348, v1204, 8);
          v1164 = v1350;
          vint16m1_t v1351 = v1172;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1352 = __riscv_vwmacc_vx_i16m1(v1351, v1348, v1240, 8);
          v1172 = v1352;
          const uint8_t* v1353 = v1182 + 371;
          const int8_t* v1354 = (const int8_t*) v1353;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1355 = *(const int8_t *)(v1354);
          vint16m1_t v1356 = v1166;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1357 = __riscv_vwmacc_vx_i16m1(v1356, v1355, v1211, 8);
          v1166 = v1357;
          vint16m1_t v1358 = v1174;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1359 = __riscv_vwmacc_vx_i16m1(v1358, v1355, v1247, 8);
          v1174 = v1359;
          const uint8_t* v1360 = v1182 + 499;
          const int8_t* v1361 = (const int8_t*) v1360;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1362 = *(const int8_t *)(v1361);
          vint16m1_t v1363 = v1168;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1364 = __riscv_vwmacc_vx_i16m1(v1363, v1362, v1218, 8);
          v1168 = v1364;
          vint16m1_t v1365 = v1176;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1366 = __riscv_vwmacc_vx_i16m1(v1365, v1362, v1254, 8);
          v1176 = v1366;
        }
        vint16m1_t v1367 = v1114;
        vint16m1_t v1368 = v1116;
        vint16m1_t v1369 = v1118;
        vint16m1_t v1370 = v1120;
        vint16m1_t v1371 = v1122;
        vint16m1_t v1372 = v1124;
        vint16m1_t v1373 = v1126;
        vint16m1_t v1374 = v1128;
        vint16m1_t v1375 = v1130;
        vint16m1_t v1376 = v1132;
        vint16m1_t v1377 = v1134;
        vint16m1_t v1378 = v1136;
        vint16m1_t v1379 = v1138;
        vint16m1_t v1380 = v1140;
        vint16m1_t v1381 = v1142;
        vint16m1_t v1382 = v1144;
        vint16m1_t v1383 = v1146;
        vint16m1_t v1384 = v1148;
        vint16m1_t v1385 = v1150;
        vint16m1_t v1386 = v1152;
        vint16m1_t v1387 = v1154;
        vint16m1_t v1388 = v1156;
        vint16m1_t v1389 = v1158;
        vint16m1_t v1390 = v1160;
        vint16m1_t v1391 = v1162;
        vint16m1_t v1392 = v1164;
        vint16m1_t v1393 = v1166;
        vint16m1_t v1394 = v1168;
        vint16m1_t v1395 = v1170;
        vint16m1_t v1396 = v1172;
        vint16m1_t v1397 = v1174;
        vint16m1_t v1398 = v1176;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v1399 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1400 = __riscv_vwmacc_vv_i32m2(v1399, v760, v1367, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1401 = __riscv_vwmacc_vv_i32m2(v1400, v764, v1368, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1402 = __riscv_vwmacc_vv_i32m2(v1401, v768, v1369, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1403 = __riscv_vwmacc_vv_i32m2(v1402, v772, v1370, 8);
        v59 = v1403;
        vint32m2_t v1404 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1405 = __riscv_vwmacc_vv_i32m2(v1404, v776, v1371, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1406 = __riscv_vwmacc_vv_i32m2(v1405, v780, v1372, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1407 = __riscv_vwmacc_vv_i32m2(v1406, v784, v1373, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1408 = __riscv_vwmacc_vv_i32m2(v1407, v788, v1374, 8);
        v61 = v1408;
        vint32m2_t v1409 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1410 = __riscv_vwmacc_vv_i32m2(v1409, v760, v1375, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1411 = __riscv_vwmacc_vv_i32m2(v1410, v764, v1376, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1412 = __riscv_vwmacc_vv_i32m2(v1411, v768, v1377, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1413 = __riscv_vwmacc_vv_i32m2(v1412, v772, v1378, 8);
        v63 = v1413;
        vint32m2_t v1414 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1415 = __riscv_vwmacc_vv_i32m2(v1414, v776, v1379, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1416 = __riscv_vwmacc_vv_i32m2(v1415, v780, v1380, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1417 = __riscv_vwmacc_vv_i32m2(v1416, v784, v1381, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1418 = __riscv_vwmacc_vv_i32m2(v1417, v788, v1382, 8);
        v65 = v1418;
        vint32m2_t v1419 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1420 = __riscv_vwmacc_vv_i32m2(v1419, v760, v1383, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1421 = __riscv_vwmacc_vv_i32m2(v1420, v764, v1384, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1422 = __riscv_vwmacc_vv_i32m2(v1421, v768, v1385, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1423 = __riscv_vwmacc_vv_i32m2(v1422, v772, v1386, 8);
        v67 = v1423;
        vint32m2_t v1424 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1425 = __riscv_vwmacc_vv_i32m2(v1424, v776, v1387, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1426 = __riscv_vwmacc_vv_i32m2(v1425, v780, v1388, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1427 = __riscv_vwmacc_vv_i32m2(v1426, v784, v1389, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1428 = __riscv_vwmacc_vv_i32m2(v1427, v788, v1390, 8);
        v69 = v1428;
        vint32m2_t v1429 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1430 = __riscv_vwmacc_vv_i32m2(v1429, v760, v1391, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1431 = __riscv_vwmacc_vv_i32m2(v1430, v764, v1392, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1432 = __riscv_vwmacc_vv_i32m2(v1431, v768, v1393, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1433 = __riscv_vwmacc_vv_i32m2(v1432, v772, v1394, 8);
        v71 = v1433;
        vint32m2_t v1434 = v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1435 = __riscv_vwmacc_vv_i32m2(v1434, v776, v1395, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1436 = __riscv_vwmacc_vv_i32m2(v1435, v780, v1396, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1437 = __riscv_vwmacc_vv_i32m2(v1436, v784, v1397, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1438 = __riscv_vwmacc_vv_i32m2(v1437, v788, v1398, 8);
        v73 = v1438;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v1439 = v38 + 160;
        const int8_t* v1440 = (const int8_t*) v1439;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1441 = __riscv_vle8_v_i8mf2(v1440, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1442 = __riscv_vsext_vf2_i16m1(v1441, 8);
        const uint8_t* v1443 = v38 + 192;
        const int8_t* v1444 = (const int8_t*) v1443;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1445 = __riscv_vle8_v_i8mf2(v1444, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1446 = __riscv_vsext_vf2_i16m1(v1445, 8);
        const uint8_t* v1447 = v38 + 224;
        const int8_t* v1448 = (const int8_t*) v1447;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1449 = __riscv_vle8_v_i8mf2(v1448, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1450 = __riscv_vsext_vf2_i16m1(v1449, 8);
        const uint8_t* v1451 = v38 + 256;
        const int8_t* v1452 = (const int8_t*) v1451;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1453 = __riscv_vle8_v_i8mf2(v1452, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1454 = __riscv_vsext_vf2_i16m1(v1453, 8);
        const uint8_t* v1455 = v38 + 168;
        const int8_t* v1456 = (const int8_t*) v1455;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1457 = __riscv_vle8_v_i8mf2(v1456, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1458 = __riscv_vsext_vf2_i16m1(v1457, 8);
        const uint8_t* v1459 = v38 + 200;
        const int8_t* v1460 = (const int8_t*) v1459;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1461 = __riscv_vle8_v_i8mf2(v1460, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1462 = __riscv_vsext_vf2_i16m1(v1461, 8);
        const uint8_t* v1463 = v38 + 232;
        const int8_t* v1464 = (const int8_t*) v1463;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1465 = __riscv_vle8_v_i8mf2(v1464, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1466 = __riscv_vsext_vf2_i16m1(v1465, 8);
        const uint8_t* v1467 = v38 + 264;
        const int8_t* v1468 = (const int8_t*) v1467;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v1469 = __riscv_vle8_v_i8mf2(v1468, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v1470 = __riscv_vsext_vf2_i16m1(v1469, 8);
        vint16m1_t v1471;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1472 = __riscv_vmv_v_x_i16m1(0, 8);
        v1471 = v1472;
        vint16m1_t v1473;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1474 = __riscv_vmv_v_x_i16m1(0, 8);
        v1473 = v1474;
        vint16m1_t v1475;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1476 = __riscv_vmv_v_x_i16m1(0, 8);
        v1475 = v1476;
        vint16m1_t v1477;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1478 = __riscv_vmv_v_x_i16m1(0, 8);
        v1477 = v1478;
        vint16m1_t v1479;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1480 = __riscv_vmv_v_x_i16m1(0, 8);
        v1479 = v1480;
        vint16m1_t v1481;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1482 = __riscv_vmv_v_x_i16m1(0, 8);
        v1481 = v1482;
        vint16m1_t v1483;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1484 = __riscv_vmv_v_x_i16m1(0, 8);
        v1483 = v1484;
        vint16m1_t v1485;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1486 = __riscv_vmv_v_x_i16m1(0, 8);
        v1485 = v1486;
        vint16m1_t v1487;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1488 = __riscv_vmv_v_x_i16m1(0, 8);
        v1487 = v1488;
        vint16m1_t v1489;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1490 = __riscv_vmv_v_x_i16m1(0, 8);
        v1489 = v1490;
        vint16m1_t v1491;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1492 = __riscv_vmv_v_x_i16m1(0, 8);
        v1491 = v1492;
        vint16m1_t v1493;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1494 = __riscv_vmv_v_x_i16m1(0, 8);
        v1493 = v1494;
        vint16m1_t v1495;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1496 = __riscv_vmv_v_x_i16m1(0, 8);
        v1495 = v1496;
        vint16m1_t v1497;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1498 = __riscv_vmv_v_x_i16m1(0, 8);
        v1497 = v1498;
        vint16m1_t v1499;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1500 = __riscv_vmv_v_x_i16m1(0, 8);
        v1499 = v1500;
        vint16m1_t v1501;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1502 = __riscv_vmv_v_x_i16m1(0, 8);
        v1501 = v1502;
        vint16m1_t v1503;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1504 = __riscv_vmv_v_x_i16m1(0, 8);
        v1503 = v1504;
        vint16m1_t v1505;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1506 = __riscv_vmv_v_x_i16m1(0, 8);
        v1505 = v1506;
        vint16m1_t v1507;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1508 = __riscv_vmv_v_x_i16m1(0, 8);
        v1507 = v1508;
        vint16m1_t v1509;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1510 = __riscv_vmv_v_x_i16m1(0, 8);
        v1509 = v1510;
        vint16m1_t v1511;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1512 = __riscv_vmv_v_x_i16m1(0, 8);
        v1511 = v1512;
        vint16m1_t v1513;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1514 = __riscv_vmv_v_x_i16m1(0, 8);
        v1513 = v1514;
        vint16m1_t v1515;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1516 = __riscv_vmv_v_x_i16m1(0, 8);
        v1515 = v1516;
        vint16m1_t v1517;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1518 = __riscv_vmv_v_x_i16m1(0, 8);
        v1517 = v1518;
        vint16m1_t v1519;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1520 = __riscv_vmv_v_x_i16m1(0, 8);
        v1519 = v1520;
        vint16m1_t v1521;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1522 = __riscv_vmv_v_x_i16m1(0, 8);
        v1521 = v1522;
        vint16m1_t v1523;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1524 = __riscv_vmv_v_x_i16m1(0, 8);
        v1523 = v1524;
        vint16m1_t v1525;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1526 = __riscv_vmv_v_x_i16m1(0, 8);
        v1525 = v1526;
        vint16m1_t v1527;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1528 = __riscv_vmv_v_x_i16m1(0, 8);
        v1527 = v1528;
        vint16m1_t v1529;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1530 = __riscv_vmv_v_x_i16m1(0, 8);
        v1529 = v1530;
        vint16m1_t v1531;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1532 = __riscv_vmv_v_x_i16m1(0, 8);
        v1531 = v1532;
        vint16m1_t v1533;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1534 = __riscv_vmv_v_x_i16m1(0, 8);
        v1533 = v1534;
        for (size_t v1535 = 0; v1535 < 8; v1535 += 1) {
          size_t v1536 = v1535 * 16;
          const uint8_t* v1537 = v38 + v1536;
          size_t v1538 = v1535 * 4;
          const uint8_t* v1539 = v40 + v1538;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v1540 = v1537 + 2336;
          const uint8_t* v1541 = (const uint8_t*) v1540;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1542 = __riscv_vle8_v_u8mf2(v1541, 8);
          const uint8_t* v1543 = v1537 + 2848;
          const uint8_t* v1544 = (const uint8_t*) v1543;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1545 = __riscv_vle8_v_u8mf2(v1544, 8);
          const uint8_t* v1546 = v1537 + 800;
          const uint8_t* v1547 = (const uint8_t*) v1546;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1548 = __riscv_vle8_v_u8mf2(v1547, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1549 = __riscv_vand_vx_u8mf2(v1542, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1550 = __riscv_vand_vx_u8mf2(v1548, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1551 = __riscv_vsll_vx_u8mf2(v1550, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1552 = __riscv_vor_vv_u8mf2(v1549, v1551, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1553 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1552);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1554 = __riscv_vsub_vx_i8mf2(v1553, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1555 = __riscv_vand_vx_u8mf2(v1545, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1556 = __riscv_vsrl_vx_u8mf2(v1548, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1557 = __riscv_vand_vx_u8mf2(v1556, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1558 = __riscv_vsll_vx_u8mf2(v1557, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1559 = __riscv_vor_vv_u8mf2(v1555, v1558, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1560 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1559);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1561 = __riscv_vsub_vx_i8mf2(v1560, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1562 = __riscv_vsrl_vx_u8mf2(v1542, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1563 = __riscv_vsrl_vx_u8mf2(v1548, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1564 = __riscv_vand_vx_u8mf2(v1563, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1565 = __riscv_vsll_vx_u8mf2(v1564, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1566 = __riscv_vor_vv_u8mf2(v1562, v1565, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1567 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1566);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1568 = __riscv_vsub_vx_i8mf2(v1567, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1569 = __riscv_vsrl_vx_u8mf2(v1545, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1570 = __riscv_vsrl_vx_u8mf2(v1548, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1571 = __riscv_vand_vx_u8mf2(v1570, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1572 = __riscv_vsll_vx_u8mf2(v1571, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1573 = __riscv_vor_vv_u8mf2(v1569, v1572, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1574 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1573);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1575 = __riscv_vsub_vx_i8mf2(v1574, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v1576 = v1537 + 2344;
          const uint8_t* v1577 = (const uint8_t*) v1576;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1578 = __riscv_vle8_v_u8mf2(v1577, 8);
          const uint8_t* v1579 = v1537 + 2856;
          const uint8_t* v1580 = (const uint8_t*) v1579;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1581 = __riscv_vle8_v_u8mf2(v1580, 8);
          const uint8_t* v1582 = v1537 + 808;
          const uint8_t* v1583 = (const uint8_t*) v1582;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1584 = __riscv_vle8_v_u8mf2(v1583, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1585 = __riscv_vand_vx_u8mf2(v1578, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1586 = __riscv_vand_vx_u8mf2(v1584, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1587 = __riscv_vsll_vx_u8mf2(v1586, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1588 = __riscv_vor_vv_u8mf2(v1585, v1587, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1589 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1588);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1590 = __riscv_vsub_vx_i8mf2(v1589, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1591 = __riscv_vand_vx_u8mf2(v1581, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1592 = __riscv_vsrl_vx_u8mf2(v1584, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1593 = __riscv_vand_vx_u8mf2(v1592, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1594 = __riscv_vsll_vx_u8mf2(v1593, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1595 = __riscv_vor_vv_u8mf2(v1591, v1594, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1596 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1595);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1597 = __riscv_vsub_vx_i8mf2(v1596, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1598 = __riscv_vsrl_vx_u8mf2(v1578, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1599 = __riscv_vsrl_vx_u8mf2(v1584, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1600 = __riscv_vand_vx_u8mf2(v1599, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1601 = __riscv_vsll_vx_u8mf2(v1600, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1602 = __riscv_vor_vv_u8mf2(v1598, v1601, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1603 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1602);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1604 = __riscv_vsub_vx_i8mf2(v1603, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1605 = __riscv_vsrl_vx_u8mf2(v1581, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1606 = __riscv_vsrl_vx_u8mf2(v1584, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1607 = __riscv_vand_vx_u8mf2(v1606, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1608 = __riscv_vsll_vx_u8mf2(v1607, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1609 = __riscv_vor_vv_u8mf2(v1605, v1608, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1610 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1609);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1611 = __riscv_vsub_vx_i8mf2(v1610, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1612 = v1539 + 528;
          const int8_t* v1613 = (const int8_t*) v1612;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1614 = *(const int8_t *)(v1613);
          vint16m1_t v1615 = v1471;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1616 = __riscv_vwmacc_vx_i16m1(v1615, v1614, v1554, 8);
          v1471 = v1616;
          vint16m1_t v1617 = v1479;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1618 = __riscv_vwmacc_vx_i16m1(v1617, v1614, v1590, 8);
          v1479 = v1618;
          const uint8_t* v1619 = v1539 + 656;
          const int8_t* v1620 = (const int8_t*) v1619;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1621 = *(const int8_t *)(v1620);
          vint16m1_t v1622 = v1473;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1623 = __riscv_vwmacc_vx_i16m1(v1622, v1621, v1561, 8);
          v1473 = v1623;
          vint16m1_t v1624 = v1481;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1625 = __riscv_vwmacc_vx_i16m1(v1624, v1621, v1597, 8);
          v1481 = v1625;
          const uint8_t* v1626 = v1539 + 784;
          const int8_t* v1627 = (const int8_t*) v1626;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1628 = *(const int8_t *)(v1627);
          vint16m1_t v1629 = v1475;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1630 = __riscv_vwmacc_vx_i16m1(v1629, v1628, v1568, 8);
          v1475 = v1630;
          vint16m1_t v1631 = v1483;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1632 = __riscv_vwmacc_vx_i16m1(v1631, v1628, v1604, 8);
          v1483 = v1632;
          const uint8_t* v1633 = v1539 + 912;
          const int8_t* v1634 = (const int8_t*) v1633;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1635 = *(const int8_t *)(v1634);
          vint16m1_t v1636 = v1477;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1637 = __riscv_vwmacc_vx_i16m1(v1636, v1635, v1575, 8);
          v1477 = v1637;
          vint16m1_t v1638 = v1485;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1639 = __riscv_vwmacc_vx_i16m1(v1638, v1635, v1611, 8);
          v1485 = v1639;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1640 = v1539 + 529;
          const int8_t* v1641 = (const int8_t*) v1640;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1642 = *(const int8_t *)(v1641);
          vint16m1_t v1643 = v1487;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1644 = __riscv_vwmacc_vx_i16m1(v1643, v1642, v1554, 8);
          v1487 = v1644;
          vint16m1_t v1645 = v1495;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1646 = __riscv_vwmacc_vx_i16m1(v1645, v1642, v1590, 8);
          v1495 = v1646;
          const uint8_t* v1647 = v1539 + 657;
          const int8_t* v1648 = (const int8_t*) v1647;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1649 = *(const int8_t *)(v1648);
          vint16m1_t v1650 = v1489;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1651 = __riscv_vwmacc_vx_i16m1(v1650, v1649, v1561, 8);
          v1489 = v1651;
          vint16m1_t v1652 = v1497;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1653 = __riscv_vwmacc_vx_i16m1(v1652, v1649, v1597, 8);
          v1497 = v1653;
          const uint8_t* v1654 = v1539 + 785;
          const int8_t* v1655 = (const int8_t*) v1654;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1656 = *(const int8_t *)(v1655);
          vint16m1_t v1657 = v1491;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1658 = __riscv_vwmacc_vx_i16m1(v1657, v1656, v1568, 8);
          v1491 = v1658;
          vint16m1_t v1659 = v1499;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1660 = __riscv_vwmacc_vx_i16m1(v1659, v1656, v1604, 8);
          v1499 = v1660;
          const uint8_t* v1661 = v1539 + 913;
          const int8_t* v1662 = (const int8_t*) v1661;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1663 = *(const int8_t *)(v1662);
          vint16m1_t v1664 = v1493;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1665 = __riscv_vwmacc_vx_i16m1(v1664, v1663, v1575, 8);
          v1493 = v1665;
          vint16m1_t v1666 = v1501;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1667 = __riscv_vwmacc_vx_i16m1(v1666, v1663, v1611, 8);
          v1501 = v1667;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1668 = v1539 + 530;
          const int8_t* v1669 = (const int8_t*) v1668;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1670 = *(const int8_t *)(v1669);
          vint16m1_t v1671 = v1503;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1672 = __riscv_vwmacc_vx_i16m1(v1671, v1670, v1554, 8);
          v1503 = v1672;
          vint16m1_t v1673 = v1511;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1674 = __riscv_vwmacc_vx_i16m1(v1673, v1670, v1590, 8);
          v1511 = v1674;
          const uint8_t* v1675 = v1539 + 658;
          const int8_t* v1676 = (const int8_t*) v1675;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1677 = *(const int8_t *)(v1676);
          vint16m1_t v1678 = v1505;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1679 = __riscv_vwmacc_vx_i16m1(v1678, v1677, v1561, 8);
          v1505 = v1679;
          vint16m1_t v1680 = v1513;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1681 = __riscv_vwmacc_vx_i16m1(v1680, v1677, v1597, 8);
          v1513 = v1681;
          const uint8_t* v1682 = v1539 + 786;
          const int8_t* v1683 = (const int8_t*) v1682;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1684 = *(const int8_t *)(v1683);
          vint16m1_t v1685 = v1507;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1686 = __riscv_vwmacc_vx_i16m1(v1685, v1684, v1568, 8);
          v1507 = v1686;
          vint16m1_t v1687 = v1515;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1688 = __riscv_vwmacc_vx_i16m1(v1687, v1684, v1604, 8);
          v1515 = v1688;
          const uint8_t* v1689 = v1539 + 914;
          const int8_t* v1690 = (const int8_t*) v1689;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1691 = *(const int8_t *)(v1690);
          vint16m1_t v1692 = v1509;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1693 = __riscv_vwmacc_vx_i16m1(v1692, v1691, v1575, 8);
          v1509 = v1693;
          vint16m1_t v1694 = v1517;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1695 = __riscv_vwmacc_vx_i16m1(v1694, v1691, v1611, 8);
          v1517 = v1695;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1696 = v1539 + 531;
          const int8_t* v1697 = (const int8_t*) v1696;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1698 = *(const int8_t *)(v1697);
          vint16m1_t v1699 = v1519;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1700 = __riscv_vwmacc_vx_i16m1(v1699, v1698, v1554, 8);
          v1519 = v1700;
          vint16m1_t v1701 = v1527;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1702 = __riscv_vwmacc_vx_i16m1(v1701, v1698, v1590, 8);
          v1527 = v1702;
          const uint8_t* v1703 = v1539 + 659;
          const int8_t* v1704 = (const int8_t*) v1703;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1705 = *(const int8_t *)(v1704);
          vint16m1_t v1706 = v1521;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1707 = __riscv_vwmacc_vx_i16m1(v1706, v1705, v1561, 8);
          v1521 = v1707;
          vint16m1_t v1708 = v1529;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1709 = __riscv_vwmacc_vx_i16m1(v1708, v1705, v1597, 8);
          v1529 = v1709;
          const uint8_t* v1710 = v1539 + 787;
          const int8_t* v1711 = (const int8_t*) v1710;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1712 = *(const int8_t *)(v1711);
          vint16m1_t v1713 = v1523;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1714 = __riscv_vwmacc_vx_i16m1(v1713, v1712, v1568, 8);
          v1523 = v1714;
          vint16m1_t v1715 = v1531;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1716 = __riscv_vwmacc_vx_i16m1(v1715, v1712, v1604, 8);
          v1531 = v1716;
          const uint8_t* v1717 = v1539 + 915;
          const int8_t* v1718 = (const int8_t*) v1717;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1719 = *(const int8_t *)(v1718);
          vint16m1_t v1720 = v1525;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1721 = __riscv_vwmacc_vx_i16m1(v1720, v1719, v1575, 8);
          v1525 = v1721;
          vint16m1_t v1722 = v1533;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1723 = __riscv_vwmacc_vx_i16m1(v1722, v1719, v1611, 8);
          v1533 = v1723;
        }
        vint16m1_t v1724 = v1471;
        vint16m1_t v1725 = v1473;
        vint16m1_t v1726 = v1475;
        vint16m1_t v1727 = v1477;
        vint16m1_t v1728 = v1479;
        vint16m1_t v1729 = v1481;
        vint16m1_t v1730 = v1483;
        vint16m1_t v1731 = v1485;
        vint16m1_t v1732 = v1487;
        vint16m1_t v1733 = v1489;
        vint16m1_t v1734 = v1491;
        vint16m1_t v1735 = v1493;
        vint16m1_t v1736 = v1495;
        vint16m1_t v1737 = v1497;
        vint16m1_t v1738 = v1499;
        vint16m1_t v1739 = v1501;
        vint16m1_t v1740 = v1503;
        vint16m1_t v1741 = v1505;
        vint16m1_t v1742 = v1507;
        vint16m1_t v1743 = v1509;
        vint16m1_t v1744 = v1511;
        vint16m1_t v1745 = v1513;
        vint16m1_t v1746 = v1515;
        vint16m1_t v1747 = v1517;
        vint16m1_t v1748 = v1519;
        vint16m1_t v1749 = v1521;
        vint16m1_t v1750 = v1523;
        vint16m1_t v1751 = v1525;
        vint16m1_t v1752 = v1527;
        vint16m1_t v1753 = v1529;
        vint16m1_t v1754 = v1531;
        vint16m1_t v1755 = v1533;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v1756 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1757 = __riscv_vwmacc_vv_i32m2(v1756, v1442, v1724, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1758 = __riscv_vwmacc_vv_i32m2(v1757, v1446, v1725, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1759 = __riscv_vwmacc_vv_i32m2(v1758, v1450, v1726, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1760 = __riscv_vwmacc_vv_i32m2(v1759, v1454, v1727, 8);
        v59 = v1760;
        vint32m2_t v1761 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1762 = __riscv_vwmacc_vv_i32m2(v1761, v1458, v1728, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1763 = __riscv_vwmacc_vv_i32m2(v1762, v1462, v1729, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1764 = __riscv_vwmacc_vv_i32m2(v1763, v1466, v1730, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1765 = __riscv_vwmacc_vv_i32m2(v1764, v1470, v1731, 8);
        v61 = v1765;
        vint32m2_t v1766 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1767 = __riscv_vwmacc_vv_i32m2(v1766, v1442, v1732, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1768 = __riscv_vwmacc_vv_i32m2(v1767, v1446, v1733, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1769 = __riscv_vwmacc_vv_i32m2(v1768, v1450, v1734, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1770 = __riscv_vwmacc_vv_i32m2(v1769, v1454, v1735, 8);
        v63 = v1770;
        vint32m2_t v1771 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1772 = __riscv_vwmacc_vv_i32m2(v1771, v1458, v1736, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1773 = __riscv_vwmacc_vv_i32m2(v1772, v1462, v1737, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1774 = __riscv_vwmacc_vv_i32m2(v1773, v1466, v1738, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1775 = __riscv_vwmacc_vv_i32m2(v1774, v1470, v1739, 8);
        v65 = v1775;
        vint32m2_t v1776 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1777 = __riscv_vwmacc_vv_i32m2(v1776, v1442, v1740, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1778 = __riscv_vwmacc_vv_i32m2(v1777, v1446, v1741, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1779 = __riscv_vwmacc_vv_i32m2(v1778, v1450, v1742, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1780 = __riscv_vwmacc_vv_i32m2(v1779, v1454, v1743, 8);
        v67 = v1780;
        vint32m2_t v1781 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1782 = __riscv_vwmacc_vv_i32m2(v1781, v1458, v1744, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1783 = __riscv_vwmacc_vv_i32m2(v1782, v1462, v1745, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1784 = __riscv_vwmacc_vv_i32m2(v1783, v1466, v1746, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1785 = __riscv_vwmacc_vv_i32m2(v1784, v1470, v1747, 8);
        v69 = v1785;
        vint32m2_t v1786 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1787 = __riscv_vwmacc_vv_i32m2(v1786, v1442, v1748, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1788 = __riscv_vwmacc_vv_i32m2(v1787, v1446, v1749, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1789 = __riscv_vwmacc_vv_i32m2(v1788, v1450, v1750, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1790 = __riscv_vwmacc_vv_i32m2(v1789, v1454, v1751, 8);
        v71 = v1790;
        vint32m2_t v1791 = v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1792 = __riscv_vwmacc_vv_i32m2(v1791, v1458, v1752, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1793 = __riscv_vwmacc_vv_i32m2(v1792, v1462, v1753, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1794 = __riscv_vwmacc_vv_i32m2(v1793, v1466, v1754, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1795 = __riscv_vwmacc_vv_i32m2(v1794, v1470, v1755, 8);
        v73 = v1795;
        vint16m1_t v1796;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1797 = __riscv_vmv_v_x_i16m1(0, 8);
        v1796 = v1797;
        vint16m1_t v1798;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1799 = __riscv_vmv_v_x_i16m1(0, 8);
        v1798 = v1799;
        vint16m1_t v1800;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1801 = __riscv_vmv_v_x_i16m1(0, 8);
        v1800 = v1801;
        vint16m1_t v1802;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1803 = __riscv_vmv_v_x_i16m1(0, 8);
        v1802 = v1803;
        vint16m1_t v1804;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1805 = __riscv_vmv_v_x_i16m1(0, 8);
        v1804 = v1805;
        vint16m1_t v1806;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1807 = __riscv_vmv_v_x_i16m1(0, 8);
        v1806 = v1807;
        vint16m1_t v1808;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1809 = __riscv_vmv_v_x_i16m1(0, 8);
        v1808 = v1809;
        vint16m1_t v1810;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1811 = __riscv_vmv_v_x_i16m1(0, 8);
        v1810 = v1811;
        vint16m1_t v1812;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1813 = __riscv_vmv_v_x_i16m1(0, 8);
        v1812 = v1813;
        vint16m1_t v1814;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1815 = __riscv_vmv_v_x_i16m1(0, 8);
        v1814 = v1815;
        vint16m1_t v1816;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1817 = __riscv_vmv_v_x_i16m1(0, 8);
        v1816 = v1817;
        vint16m1_t v1818;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1819 = __riscv_vmv_v_x_i16m1(0, 8);
        v1818 = v1819;
        vint16m1_t v1820;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1821 = __riscv_vmv_v_x_i16m1(0, 8);
        v1820 = v1821;
        vint16m1_t v1822;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1823 = __riscv_vmv_v_x_i16m1(0, 8);
        v1822 = v1823;
        vint16m1_t v1824;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1825 = __riscv_vmv_v_x_i16m1(0, 8);
        v1824 = v1825;
        vint16m1_t v1826;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1827 = __riscv_vmv_v_x_i16m1(0, 8);
        v1826 = v1827;
        vint16m1_t v1828;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1829 = __riscv_vmv_v_x_i16m1(0, 8);
        v1828 = v1829;
        vint16m1_t v1830;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1831 = __riscv_vmv_v_x_i16m1(0, 8);
        v1830 = v1831;
        vint16m1_t v1832;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1833 = __riscv_vmv_v_x_i16m1(0, 8);
        v1832 = v1833;
        vint16m1_t v1834;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1835 = __riscv_vmv_v_x_i16m1(0, 8);
        v1834 = v1835;
        vint16m1_t v1836;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1837 = __riscv_vmv_v_x_i16m1(0, 8);
        v1836 = v1837;
        vint16m1_t v1838;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1839 = __riscv_vmv_v_x_i16m1(0, 8);
        v1838 = v1839;
        vint16m1_t v1840;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1841 = __riscv_vmv_v_x_i16m1(0, 8);
        v1840 = v1841;
        vint16m1_t v1842;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1843 = __riscv_vmv_v_x_i16m1(0, 8);
        v1842 = v1843;
        vint16m1_t v1844;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1845 = __riscv_vmv_v_x_i16m1(0, 8);
        v1844 = v1845;
        vint16m1_t v1846;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1847 = __riscv_vmv_v_x_i16m1(0, 8);
        v1846 = v1847;
        vint16m1_t v1848;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1849 = __riscv_vmv_v_x_i16m1(0, 8);
        v1848 = v1849;
        vint16m1_t v1850;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1851 = __riscv_vmv_v_x_i16m1(0, 8);
        v1850 = v1851;
        vint16m1_t v1852;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1853 = __riscv_vmv_v_x_i16m1(0, 8);
        v1852 = v1853;
        vint16m1_t v1854;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1855 = __riscv_vmv_v_x_i16m1(0, 8);
        v1854 = v1855;
        vint16m1_t v1856;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1857 = __riscv_vmv_v_x_i16m1(0, 8);
        v1856 = v1857;
        vint16m1_t v1858;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1859 = __riscv_vmv_v_x_i16m1(0, 8);
        v1858 = v1859;
        for (size_t v1860 = 0; v1860 < 8; v1860 += 1) {
          size_t v1861 = v1860 * 16;
          const uint8_t* v1862 = v38 + v1861;
          size_t v1863 = v1860 * 4;
          const uint8_t* v1864 = v40 + v1863;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v1865 = v1862 + 2464;
          const uint8_t* v1866 = (const uint8_t*) v1865;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1867 = __riscv_vle8_v_u8mf2(v1866, 8);
          const uint8_t* v1868 = v1862 + 2976;
          const uint8_t* v1869 = (const uint8_t*) v1868;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1870 = __riscv_vle8_v_u8mf2(v1869, 8);
          const uint8_t* v1871 = v1862 + 928;
          const uint8_t* v1872 = (const uint8_t*) v1871;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1873 = __riscv_vle8_v_u8mf2(v1872, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1874 = __riscv_vand_vx_u8mf2(v1867, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1875 = __riscv_vand_vx_u8mf2(v1873, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1876 = __riscv_vsll_vx_u8mf2(v1875, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1877 = __riscv_vor_vv_u8mf2(v1874, v1876, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1878 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1877);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1879 = __riscv_vsub_vx_i8mf2(v1878, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1880 = __riscv_vand_vx_u8mf2(v1870, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1881 = __riscv_vsrl_vx_u8mf2(v1873, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1882 = __riscv_vand_vx_u8mf2(v1881, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1883 = __riscv_vsll_vx_u8mf2(v1882, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1884 = __riscv_vor_vv_u8mf2(v1880, v1883, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1885 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1884);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1886 = __riscv_vsub_vx_i8mf2(v1885, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1887 = __riscv_vsrl_vx_u8mf2(v1867, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1888 = __riscv_vsrl_vx_u8mf2(v1873, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1889 = __riscv_vand_vx_u8mf2(v1888, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1890 = __riscv_vsll_vx_u8mf2(v1889, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1891 = __riscv_vor_vv_u8mf2(v1887, v1890, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1892 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1891);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1893 = __riscv_vsub_vx_i8mf2(v1892, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1894 = __riscv_vsrl_vx_u8mf2(v1870, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1895 = __riscv_vsrl_vx_u8mf2(v1873, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1896 = __riscv_vand_vx_u8mf2(v1895, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1897 = __riscv_vsll_vx_u8mf2(v1896, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1898 = __riscv_vor_vv_u8mf2(v1894, v1897, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1899 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1898);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1900 = __riscv_vsub_vx_i8mf2(v1899, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v1901 = v1862 + 2472;
          const uint8_t* v1902 = (const uint8_t*) v1901;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1903 = __riscv_vle8_v_u8mf2(v1902, 8);
          const uint8_t* v1904 = v1862 + 2984;
          const uint8_t* v1905 = (const uint8_t*) v1904;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1906 = __riscv_vle8_v_u8mf2(v1905, 8);
          const uint8_t* v1907 = v1862 + 936;
          const uint8_t* v1908 = (const uint8_t*) v1907;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1909 = __riscv_vle8_v_u8mf2(v1908, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1910 = __riscv_vand_vx_u8mf2(v1903, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1911 = __riscv_vand_vx_u8mf2(v1909, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1912 = __riscv_vsll_vx_u8mf2(v1911, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1913 = __riscv_vor_vv_u8mf2(v1910, v1912, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1914 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1913);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1915 = __riscv_vsub_vx_i8mf2(v1914, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1916 = __riscv_vand_vx_u8mf2(v1906, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1917 = __riscv_vsrl_vx_u8mf2(v1909, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1918 = __riscv_vand_vx_u8mf2(v1917, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1919 = __riscv_vsll_vx_u8mf2(v1918, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1920 = __riscv_vor_vv_u8mf2(v1916, v1919, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1921 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1920);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1922 = __riscv_vsub_vx_i8mf2(v1921, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1923 = __riscv_vsrl_vx_u8mf2(v1903, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1924 = __riscv_vsrl_vx_u8mf2(v1909, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1925 = __riscv_vand_vx_u8mf2(v1924, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1926 = __riscv_vsll_vx_u8mf2(v1925, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1927 = __riscv_vor_vv_u8mf2(v1923, v1926, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1928 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1927);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1929 = __riscv_vsub_vx_i8mf2(v1928, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1930 = __riscv_vsrl_vx_u8mf2(v1906, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1931 = __riscv_vsrl_vx_u8mf2(v1909, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1932 = __riscv_vand_vx_u8mf2(v1931, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1933 = __riscv_vsll_vx_u8mf2(v1932, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1934 = __riscv_vor_vv_u8mf2(v1930, v1933, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1935 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1934);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v1936 = __riscv_vsub_vx_i8mf2(v1935, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1937 = v1864 + 560;
          const int8_t* v1938 = (const int8_t*) v1937;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1939 = *(const int8_t *)(v1938);
          vint16m1_t v1940 = v1796;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1941 = __riscv_vwmacc_vx_i16m1(v1940, v1939, v1879, 8);
          v1796 = v1941;
          vint16m1_t v1942 = v1804;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1943 = __riscv_vwmacc_vx_i16m1(v1942, v1939, v1915, 8);
          v1804 = v1943;
          const uint8_t* v1944 = v1864 + 688;
          const int8_t* v1945 = (const int8_t*) v1944;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1946 = *(const int8_t *)(v1945);
          vint16m1_t v1947 = v1798;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1948 = __riscv_vwmacc_vx_i16m1(v1947, v1946, v1886, 8);
          v1798 = v1948;
          vint16m1_t v1949 = v1806;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1950 = __riscv_vwmacc_vx_i16m1(v1949, v1946, v1922, 8);
          v1806 = v1950;
          const uint8_t* v1951 = v1864 + 816;
          const int8_t* v1952 = (const int8_t*) v1951;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1953 = *(const int8_t *)(v1952);
          vint16m1_t v1954 = v1800;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1955 = __riscv_vwmacc_vx_i16m1(v1954, v1953, v1893, 8);
          v1800 = v1955;
          vint16m1_t v1956 = v1808;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1957 = __riscv_vwmacc_vx_i16m1(v1956, v1953, v1929, 8);
          v1808 = v1957;
          const uint8_t* v1958 = v1864 + 944;
          const int8_t* v1959 = (const int8_t*) v1958;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1960 = *(const int8_t *)(v1959);
          vint16m1_t v1961 = v1802;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1962 = __riscv_vwmacc_vx_i16m1(v1961, v1960, v1900, 8);
          v1802 = v1962;
          vint16m1_t v1963 = v1810;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1964 = __riscv_vwmacc_vx_i16m1(v1963, v1960, v1936, 8);
          v1810 = v1964;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1965 = v1864 + 561;
          const int8_t* v1966 = (const int8_t*) v1965;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1967 = *(const int8_t *)(v1966);
          vint16m1_t v1968 = v1812;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1969 = __riscv_vwmacc_vx_i16m1(v1968, v1967, v1879, 8);
          v1812 = v1969;
          vint16m1_t v1970 = v1820;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1971 = __riscv_vwmacc_vx_i16m1(v1970, v1967, v1915, 8);
          v1820 = v1971;
          const uint8_t* v1972 = v1864 + 689;
          const int8_t* v1973 = (const int8_t*) v1972;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1974 = *(const int8_t *)(v1973);
          vint16m1_t v1975 = v1814;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1976 = __riscv_vwmacc_vx_i16m1(v1975, v1974, v1886, 8);
          v1814 = v1976;
          vint16m1_t v1977 = v1822;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1978 = __riscv_vwmacc_vx_i16m1(v1977, v1974, v1922, 8);
          v1822 = v1978;
          const uint8_t* v1979 = v1864 + 817;
          const int8_t* v1980 = (const int8_t*) v1979;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1981 = *(const int8_t *)(v1980);
          vint16m1_t v1982 = v1816;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1983 = __riscv_vwmacc_vx_i16m1(v1982, v1981, v1893, 8);
          v1816 = v1983;
          vint16m1_t v1984 = v1824;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1985 = __riscv_vwmacc_vx_i16m1(v1984, v1981, v1929, 8);
          v1824 = v1985;
          const uint8_t* v1986 = v1864 + 945;
          const int8_t* v1987 = (const int8_t*) v1986;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1988 = *(const int8_t *)(v1987);
          vint16m1_t v1989 = v1818;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1990 = __riscv_vwmacc_vx_i16m1(v1989, v1988, v1900, 8);
          v1818 = v1990;
          vint16m1_t v1991 = v1826;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1992 = __riscv_vwmacc_vx_i16m1(v1991, v1988, v1936, 8);
          v1826 = v1992;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v1993 = v1864 + 562;
          const int8_t* v1994 = (const int8_t*) v1993;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1995 = *(const int8_t *)(v1994);
          vint16m1_t v1996 = v1828;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1997 = __riscv_vwmacc_vx_i16m1(v1996, v1995, v1879, 8);
          v1828 = v1997;
          vint16m1_t v1998 = v1836;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1999 = __riscv_vwmacc_vx_i16m1(v1998, v1995, v1915, 8);
          v1836 = v1999;
          const uint8_t* v2000 = v1864 + 690;
          const int8_t* v2001 = (const int8_t*) v2000;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2002 = *(const int8_t *)(v2001);
          vint16m1_t v2003 = v1830;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2004 = __riscv_vwmacc_vx_i16m1(v2003, v2002, v1886, 8);
          v1830 = v2004;
          vint16m1_t v2005 = v1838;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2006 = __riscv_vwmacc_vx_i16m1(v2005, v2002, v1922, 8);
          v1838 = v2006;
          const uint8_t* v2007 = v1864 + 818;
          const int8_t* v2008 = (const int8_t*) v2007;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2009 = *(const int8_t *)(v2008);
          vint16m1_t v2010 = v1832;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2011 = __riscv_vwmacc_vx_i16m1(v2010, v2009, v1893, 8);
          v1832 = v2011;
          vint16m1_t v2012 = v1840;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2013 = __riscv_vwmacc_vx_i16m1(v2012, v2009, v1929, 8);
          v1840 = v2013;
          const uint8_t* v2014 = v1864 + 946;
          const int8_t* v2015 = (const int8_t*) v2014;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2016 = *(const int8_t *)(v2015);
          vint16m1_t v2017 = v1834;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2018 = __riscv_vwmacc_vx_i16m1(v2017, v2016, v1900, 8);
          v1834 = v2018;
          vint16m1_t v2019 = v1842;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2020 = __riscv_vwmacc_vx_i16m1(v2019, v2016, v1936, 8);
          v1842 = v2020;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v2021 = v1864 + 563;
          const int8_t* v2022 = (const int8_t*) v2021;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2023 = *(const int8_t *)(v2022);
          vint16m1_t v2024 = v1844;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2025 = __riscv_vwmacc_vx_i16m1(v2024, v2023, v1879, 8);
          v1844 = v2025;
          vint16m1_t v2026 = v1852;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2027 = __riscv_vwmacc_vx_i16m1(v2026, v2023, v1915, 8);
          v1852 = v2027;
          const uint8_t* v2028 = v1864 + 691;
          const int8_t* v2029 = (const int8_t*) v2028;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2030 = *(const int8_t *)(v2029);
          vint16m1_t v2031 = v1846;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2032 = __riscv_vwmacc_vx_i16m1(v2031, v2030, v1886, 8);
          v1846 = v2032;
          vint16m1_t v2033 = v1854;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2034 = __riscv_vwmacc_vx_i16m1(v2033, v2030, v1922, 8);
          v1854 = v2034;
          const uint8_t* v2035 = v1864 + 819;
          const int8_t* v2036 = (const int8_t*) v2035;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2037 = *(const int8_t *)(v2036);
          vint16m1_t v2038 = v1848;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2039 = __riscv_vwmacc_vx_i16m1(v2038, v2037, v1893, 8);
          v1848 = v2039;
          vint16m1_t v2040 = v1856;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2041 = __riscv_vwmacc_vx_i16m1(v2040, v2037, v1929, 8);
          v1856 = v2041;
          const uint8_t* v2042 = v1864 + 947;
          const int8_t* v2043 = (const int8_t*) v2042;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2044 = *(const int8_t *)(v2043);
          vint16m1_t v2045 = v1850;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2046 = __riscv_vwmacc_vx_i16m1(v2045, v2044, v1900, 8);
          v1850 = v2046;
          vint16m1_t v2047 = v1858;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2048 = __riscv_vwmacc_vx_i16m1(v2047, v2044, v1936, 8);
          v1858 = v2048;
        }
        vint16m1_t v2049 = v1796;
        vint16m1_t v2050 = v1798;
        vint16m1_t v2051 = v1800;
        vint16m1_t v2052 = v1802;
        vint16m1_t v2053 = v1804;
        vint16m1_t v2054 = v1806;
        vint16m1_t v2055 = v1808;
        vint16m1_t v2056 = v1810;
        vint16m1_t v2057 = v1812;
        vint16m1_t v2058 = v1814;
        vint16m1_t v2059 = v1816;
        vint16m1_t v2060 = v1818;
        vint16m1_t v2061 = v1820;
        vint16m1_t v2062 = v1822;
        vint16m1_t v2063 = v1824;
        vint16m1_t v2064 = v1826;
        vint16m1_t v2065 = v1828;
        vint16m1_t v2066 = v1830;
        vint16m1_t v2067 = v1832;
        vint16m1_t v2068 = v1834;
        vint16m1_t v2069 = v1836;
        vint16m1_t v2070 = v1838;
        vint16m1_t v2071 = v1840;
        vint16m1_t v2072 = v1842;
        vint16m1_t v2073 = v1844;
        vint16m1_t v2074 = v1846;
        vint16m1_t v2075 = v1848;
        vint16m1_t v2076 = v1850;
        vint16m1_t v2077 = v1852;
        vint16m1_t v2078 = v1854;
        vint16m1_t v2079 = v1856;
        vint16m1_t v2080 = v1858;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v2081 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2082 = __riscv_vwmacc_vv_i32m2(v2081, v1442, v2049, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2083 = __riscv_vwmacc_vv_i32m2(v2082, v1446, v2050, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2084 = __riscv_vwmacc_vv_i32m2(v2083, v1450, v2051, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2085 = __riscv_vwmacc_vv_i32m2(v2084, v1454, v2052, 8);
        v59 = v2085;
        vint32m2_t v2086 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2087 = __riscv_vwmacc_vv_i32m2(v2086, v1458, v2053, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2088 = __riscv_vwmacc_vv_i32m2(v2087, v1462, v2054, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2089 = __riscv_vwmacc_vv_i32m2(v2088, v1466, v2055, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2090 = __riscv_vwmacc_vv_i32m2(v2089, v1470, v2056, 8);
        v61 = v2090;
        vint32m2_t v2091 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2092 = __riscv_vwmacc_vv_i32m2(v2091, v1442, v2057, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2093 = __riscv_vwmacc_vv_i32m2(v2092, v1446, v2058, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2094 = __riscv_vwmacc_vv_i32m2(v2093, v1450, v2059, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2095 = __riscv_vwmacc_vv_i32m2(v2094, v1454, v2060, 8);
        v63 = v2095;
        vint32m2_t v2096 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2097 = __riscv_vwmacc_vv_i32m2(v2096, v1458, v2061, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2098 = __riscv_vwmacc_vv_i32m2(v2097, v1462, v2062, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2099 = __riscv_vwmacc_vv_i32m2(v2098, v1466, v2063, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2100 = __riscv_vwmacc_vv_i32m2(v2099, v1470, v2064, 8);
        v65 = v2100;
        vint32m2_t v2101 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2102 = __riscv_vwmacc_vv_i32m2(v2101, v1442, v2065, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2103 = __riscv_vwmacc_vv_i32m2(v2102, v1446, v2066, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2104 = __riscv_vwmacc_vv_i32m2(v2103, v1450, v2067, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2105 = __riscv_vwmacc_vv_i32m2(v2104, v1454, v2068, 8);
        v67 = v2105;
        vint32m2_t v2106 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2107 = __riscv_vwmacc_vv_i32m2(v2106, v1458, v2069, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2108 = __riscv_vwmacc_vv_i32m2(v2107, v1462, v2070, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2109 = __riscv_vwmacc_vv_i32m2(v2108, v1466, v2071, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2110 = __riscv_vwmacc_vv_i32m2(v2109, v1470, v2072, 8);
        v69 = v2110;
        vint32m2_t v2111 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2112 = __riscv_vwmacc_vv_i32m2(v2111, v1442, v2073, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2113 = __riscv_vwmacc_vv_i32m2(v2112, v1446, v2074, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2114 = __riscv_vwmacc_vv_i32m2(v2113, v1450, v2075, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2115 = __riscv_vwmacc_vv_i32m2(v2114, v1454, v2076, 8);
        v71 = v2115;
        vint32m2_t v2116 = v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2117 = __riscv_vwmacc_vv_i32m2(v2116, v1458, v2077, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2118 = __riscv_vwmacc_vv_i32m2(v2117, v1462, v2078, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2119 = __riscv_vwmacc_vv_i32m2(v2118, v1466, v2079, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2120 = __riscv_vwmacc_vv_i32m2(v2119, v1470, v2080, 8);
        v73 = v2120;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signed_scale_unpack
        const uint8_t* v2121 = v38 + 176;
        const int8_t* v2122 = (const int8_t*) v2121;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v2123 = __riscv_vle8_v_i8mf2(v2122, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v2124 = __riscv_vsext_vf2_i16m1(v2123, 8);
        const uint8_t* v2125 = v38 + 208;
        const int8_t* v2126 = (const int8_t*) v2125;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v2127 = __riscv_vle8_v_i8mf2(v2126, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v2128 = __riscv_vsext_vf2_i16m1(v2127, 8);
        const uint8_t* v2129 = v38 + 240;
        const int8_t* v2130 = (const int8_t*) v2129;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v2131 = __riscv_vle8_v_i8mf2(v2130, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v2132 = __riscv_vsext_vf2_i16m1(v2131, 8);
        const uint8_t* v2133 = v38 + 272;
        const int8_t* v2134 = (const int8_t*) v2133;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v2135 = __riscv_vle8_v_i8mf2(v2134, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v2136 = __riscv_vsext_vf2_i16m1(v2135, 8);
        const uint8_t* v2137 = v38 + 184;
        const int8_t* v2138 = (const int8_t*) v2137;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v2139 = __riscv_vle8_v_i8mf2(v2138, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v2140 = __riscv_vsext_vf2_i16m1(v2139, 8);
        const uint8_t* v2141 = v38 + 216;
        const int8_t* v2142 = (const int8_t*) v2141;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v2143 = __riscv_vle8_v_i8mf2(v2142, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v2144 = __riscv_vsext_vf2_i16m1(v2143, 8);
        const uint8_t* v2145 = v38 + 248;
        const int8_t* v2146 = (const int8_t*) v2145;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v2147 = __riscv_vle8_v_i8mf2(v2146, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v2148 = __riscv_vsext_vf2_i16m1(v2147, 8);
        const uint8_t* v2149 = v38 + 280;
        const int8_t* v2150 = (const int8_t*) v2149;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v2151 = __riscv_vle8_v_i8mf2(v2150, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i16m1
        vint16m1_t v2152 = __riscv_vsext_vf2_i16m1(v2151, 8);
        vint16m1_t v2153;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2154 = __riscv_vmv_v_x_i16m1(0, 8);
        v2153 = v2154;
        vint16m1_t v2155;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2156 = __riscv_vmv_v_x_i16m1(0, 8);
        v2155 = v2156;
        vint16m1_t v2157;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2158 = __riscv_vmv_v_x_i16m1(0, 8);
        v2157 = v2158;
        vint16m1_t v2159;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2160 = __riscv_vmv_v_x_i16m1(0, 8);
        v2159 = v2160;
        vint16m1_t v2161;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2162 = __riscv_vmv_v_x_i16m1(0, 8);
        v2161 = v2162;
        vint16m1_t v2163;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2164 = __riscv_vmv_v_x_i16m1(0, 8);
        v2163 = v2164;
        vint16m1_t v2165;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2166 = __riscv_vmv_v_x_i16m1(0, 8);
        v2165 = v2166;
        vint16m1_t v2167;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2168 = __riscv_vmv_v_x_i16m1(0, 8);
        v2167 = v2168;
        vint16m1_t v2169;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2170 = __riscv_vmv_v_x_i16m1(0, 8);
        v2169 = v2170;
        vint16m1_t v2171;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2172 = __riscv_vmv_v_x_i16m1(0, 8);
        v2171 = v2172;
        vint16m1_t v2173;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2174 = __riscv_vmv_v_x_i16m1(0, 8);
        v2173 = v2174;
        vint16m1_t v2175;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2176 = __riscv_vmv_v_x_i16m1(0, 8);
        v2175 = v2176;
        vint16m1_t v2177;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2178 = __riscv_vmv_v_x_i16m1(0, 8);
        v2177 = v2178;
        vint16m1_t v2179;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2180 = __riscv_vmv_v_x_i16m1(0, 8);
        v2179 = v2180;
        vint16m1_t v2181;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2182 = __riscv_vmv_v_x_i16m1(0, 8);
        v2181 = v2182;
        vint16m1_t v2183;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2184 = __riscv_vmv_v_x_i16m1(0, 8);
        v2183 = v2184;
        vint16m1_t v2185;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2186 = __riscv_vmv_v_x_i16m1(0, 8);
        v2185 = v2186;
        vint16m1_t v2187;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2188 = __riscv_vmv_v_x_i16m1(0, 8);
        v2187 = v2188;
        vint16m1_t v2189;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2190 = __riscv_vmv_v_x_i16m1(0, 8);
        v2189 = v2190;
        vint16m1_t v2191;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2192 = __riscv_vmv_v_x_i16m1(0, 8);
        v2191 = v2192;
        vint16m1_t v2193;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2194 = __riscv_vmv_v_x_i16m1(0, 8);
        v2193 = v2194;
        vint16m1_t v2195;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2196 = __riscv_vmv_v_x_i16m1(0, 8);
        v2195 = v2196;
        vint16m1_t v2197;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2198 = __riscv_vmv_v_x_i16m1(0, 8);
        v2197 = v2198;
        vint16m1_t v2199;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2200 = __riscv_vmv_v_x_i16m1(0, 8);
        v2199 = v2200;
        vint16m1_t v2201;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2202 = __riscv_vmv_v_x_i16m1(0, 8);
        v2201 = v2202;
        vint16m1_t v2203;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2204 = __riscv_vmv_v_x_i16m1(0, 8);
        v2203 = v2204;
        vint16m1_t v2205;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2206 = __riscv_vmv_v_x_i16m1(0, 8);
        v2205 = v2206;
        vint16m1_t v2207;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2208 = __riscv_vmv_v_x_i16m1(0, 8);
        v2207 = v2208;
        vint16m1_t v2209;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2210 = __riscv_vmv_v_x_i16m1(0, 8);
        v2209 = v2210;
        vint16m1_t v2211;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2212 = __riscv_vmv_v_x_i16m1(0, 8);
        v2211 = v2212;
        vint16m1_t v2213;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2214 = __riscv_vmv_v_x_i16m1(0, 8);
        v2213 = v2214;
        vint16m1_t v2215;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2216 = __riscv_vmv_v_x_i16m1(0, 8);
        v2215 = v2216;
        for (size_t v2217 = 0; v2217 < 8; v2217 += 1) {
          size_t v2218 = v2217 * 16;
          const uint8_t* v2219 = v38 + v2218;
          size_t v2220 = v2217 * 4;
          const uint8_t* v2221 = v40 + v2220;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v2222 = v2219 + 2592;
          const uint8_t* v2223 = (const uint8_t*) v2222;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v2224 = __riscv_vle8_v_u8mf2(v2223, 8);
          const uint8_t* v2225 = v2219 + 3104;
          const uint8_t* v2226 = (const uint8_t*) v2225;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v2227 = __riscv_vle8_v_u8mf2(v2226, 8);
          const uint8_t* v2228 = v2219 + 1056;
          const uint8_t* v2229 = (const uint8_t*) v2228;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v2230 = __riscv_vle8_v_u8mf2(v2229, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2231 = __riscv_vand_vx_u8mf2(v2224, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2232 = __riscv_vand_vx_u8mf2(v2230, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2233 = __riscv_vsll_vx_u8mf2(v2232, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2234 = __riscv_vor_vv_u8mf2(v2231, v2233, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2235 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2234);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2236 = __riscv_vsub_vx_i8mf2(v2235, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2237 = __riscv_vand_vx_u8mf2(v2227, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2238 = __riscv_vsrl_vx_u8mf2(v2230, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2239 = __riscv_vand_vx_u8mf2(v2238, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2240 = __riscv_vsll_vx_u8mf2(v2239, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2241 = __riscv_vor_vv_u8mf2(v2237, v2240, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2242 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2241);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2243 = __riscv_vsub_vx_i8mf2(v2242, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2244 = __riscv_vsrl_vx_u8mf2(v2224, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2245 = __riscv_vsrl_vx_u8mf2(v2230, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2246 = __riscv_vand_vx_u8mf2(v2245, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2247 = __riscv_vsll_vx_u8mf2(v2246, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2248 = __riscv_vor_vv_u8mf2(v2244, v2247, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2249 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2248);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2250 = __riscv_vsub_vx_i8mf2(v2249, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2251 = __riscv_vsrl_vx_u8mf2(v2227, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2252 = __riscv_vsrl_vx_u8mf2(v2230, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2253 = __riscv_vand_vx_u8mf2(v2252, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2254 = __riscv_vsll_vx_u8mf2(v2253, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2255 = __riscv_vor_vv_u8mf2(v2251, v2254, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2256 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2255);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2257 = __riscv_vsub_vx_i8mf2(v2256, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v2258 = v2219 + 2600;
          const uint8_t* v2259 = (const uint8_t*) v2258;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v2260 = __riscv_vle8_v_u8mf2(v2259, 8);
          const uint8_t* v2261 = v2219 + 3112;
          const uint8_t* v2262 = (const uint8_t*) v2261;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v2263 = __riscv_vle8_v_u8mf2(v2262, 8);
          const uint8_t* v2264 = v2219 + 1064;
          const uint8_t* v2265 = (const uint8_t*) v2264;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v2266 = __riscv_vle8_v_u8mf2(v2265, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2267 = __riscv_vand_vx_u8mf2(v2260, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2268 = __riscv_vand_vx_u8mf2(v2266, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2269 = __riscv_vsll_vx_u8mf2(v2268, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2270 = __riscv_vor_vv_u8mf2(v2267, v2269, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2271 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2270);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2272 = __riscv_vsub_vx_i8mf2(v2271, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2273 = __riscv_vand_vx_u8mf2(v2263, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2274 = __riscv_vsrl_vx_u8mf2(v2266, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2275 = __riscv_vand_vx_u8mf2(v2274, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2276 = __riscv_vsll_vx_u8mf2(v2275, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2277 = __riscv_vor_vv_u8mf2(v2273, v2276, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2278 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2277);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2279 = __riscv_vsub_vx_i8mf2(v2278, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2280 = __riscv_vsrl_vx_u8mf2(v2260, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2281 = __riscv_vsrl_vx_u8mf2(v2266, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2282 = __riscv_vand_vx_u8mf2(v2281, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2283 = __riscv_vsll_vx_u8mf2(v2282, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2284 = __riscv_vor_vv_u8mf2(v2280, v2283, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2285 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2284);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2286 = __riscv_vsub_vx_i8mf2(v2285, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2287 = __riscv_vsrl_vx_u8mf2(v2263, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2288 = __riscv_vsrl_vx_u8mf2(v2266, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2289 = __riscv_vand_vx_u8mf2(v2288, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2290 = __riscv_vsll_vx_u8mf2(v2289, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2291 = __riscv_vor_vv_u8mf2(v2287, v2290, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2292 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2291);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2293 = __riscv_vsub_vx_i8mf2(v2292, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v2294 = v2221 + 592;
          const int8_t* v2295 = (const int8_t*) v2294;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2296 = *(const int8_t *)(v2295);
          vint16m1_t v2297 = v2153;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2298 = __riscv_vwmacc_vx_i16m1(v2297, v2296, v2236, 8);
          v2153 = v2298;
          vint16m1_t v2299 = v2161;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2300 = __riscv_vwmacc_vx_i16m1(v2299, v2296, v2272, 8);
          v2161 = v2300;
          const uint8_t* v2301 = v2221 + 720;
          const int8_t* v2302 = (const int8_t*) v2301;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2303 = *(const int8_t *)(v2302);
          vint16m1_t v2304 = v2155;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2305 = __riscv_vwmacc_vx_i16m1(v2304, v2303, v2243, 8);
          v2155 = v2305;
          vint16m1_t v2306 = v2163;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2307 = __riscv_vwmacc_vx_i16m1(v2306, v2303, v2279, 8);
          v2163 = v2307;
          const uint8_t* v2308 = v2221 + 848;
          const int8_t* v2309 = (const int8_t*) v2308;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2310 = *(const int8_t *)(v2309);
          vint16m1_t v2311 = v2157;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2312 = __riscv_vwmacc_vx_i16m1(v2311, v2310, v2250, 8);
          v2157 = v2312;
          vint16m1_t v2313 = v2165;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2314 = __riscv_vwmacc_vx_i16m1(v2313, v2310, v2286, 8);
          v2165 = v2314;
          const uint8_t* v2315 = v2221 + 976;
          const int8_t* v2316 = (const int8_t*) v2315;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2317 = *(const int8_t *)(v2316);
          vint16m1_t v2318 = v2159;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2319 = __riscv_vwmacc_vx_i16m1(v2318, v2317, v2257, 8);
          v2159 = v2319;
          vint16m1_t v2320 = v2167;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2321 = __riscv_vwmacc_vx_i16m1(v2320, v2317, v2293, 8);
          v2167 = v2321;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v2322 = v2221 + 593;
          const int8_t* v2323 = (const int8_t*) v2322;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2324 = *(const int8_t *)(v2323);
          vint16m1_t v2325 = v2169;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2326 = __riscv_vwmacc_vx_i16m1(v2325, v2324, v2236, 8);
          v2169 = v2326;
          vint16m1_t v2327 = v2177;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2328 = __riscv_vwmacc_vx_i16m1(v2327, v2324, v2272, 8);
          v2177 = v2328;
          const uint8_t* v2329 = v2221 + 721;
          const int8_t* v2330 = (const int8_t*) v2329;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2331 = *(const int8_t *)(v2330);
          vint16m1_t v2332 = v2171;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2333 = __riscv_vwmacc_vx_i16m1(v2332, v2331, v2243, 8);
          v2171 = v2333;
          vint16m1_t v2334 = v2179;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2335 = __riscv_vwmacc_vx_i16m1(v2334, v2331, v2279, 8);
          v2179 = v2335;
          const uint8_t* v2336 = v2221 + 849;
          const int8_t* v2337 = (const int8_t*) v2336;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2338 = *(const int8_t *)(v2337);
          vint16m1_t v2339 = v2173;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2340 = __riscv_vwmacc_vx_i16m1(v2339, v2338, v2250, 8);
          v2173 = v2340;
          vint16m1_t v2341 = v2181;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2342 = __riscv_vwmacc_vx_i16m1(v2341, v2338, v2286, 8);
          v2181 = v2342;
          const uint8_t* v2343 = v2221 + 977;
          const int8_t* v2344 = (const int8_t*) v2343;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2345 = *(const int8_t *)(v2344);
          vint16m1_t v2346 = v2175;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2347 = __riscv_vwmacc_vx_i16m1(v2346, v2345, v2257, 8);
          v2175 = v2347;
          vint16m1_t v2348 = v2183;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2349 = __riscv_vwmacc_vx_i16m1(v2348, v2345, v2293, 8);
          v2183 = v2349;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v2350 = v2221 + 594;
          const int8_t* v2351 = (const int8_t*) v2350;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2352 = *(const int8_t *)(v2351);
          vint16m1_t v2353 = v2185;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2354 = __riscv_vwmacc_vx_i16m1(v2353, v2352, v2236, 8);
          v2185 = v2354;
          vint16m1_t v2355 = v2193;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2356 = __riscv_vwmacc_vx_i16m1(v2355, v2352, v2272, 8);
          v2193 = v2356;
          const uint8_t* v2357 = v2221 + 722;
          const int8_t* v2358 = (const int8_t*) v2357;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2359 = *(const int8_t *)(v2358);
          vint16m1_t v2360 = v2187;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2361 = __riscv_vwmacc_vx_i16m1(v2360, v2359, v2243, 8);
          v2187 = v2361;
          vint16m1_t v2362 = v2195;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2363 = __riscv_vwmacc_vx_i16m1(v2362, v2359, v2279, 8);
          v2195 = v2363;
          const uint8_t* v2364 = v2221 + 850;
          const int8_t* v2365 = (const int8_t*) v2364;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2366 = *(const int8_t *)(v2365);
          vint16m1_t v2367 = v2189;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2368 = __riscv_vwmacc_vx_i16m1(v2367, v2366, v2250, 8);
          v2189 = v2368;
          vint16m1_t v2369 = v2197;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2370 = __riscv_vwmacc_vx_i16m1(v2369, v2366, v2286, 8);
          v2197 = v2370;
          const uint8_t* v2371 = v2221 + 978;
          const int8_t* v2372 = (const int8_t*) v2371;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2373 = *(const int8_t *)(v2372);
          vint16m1_t v2374 = v2191;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2375 = __riscv_vwmacc_vx_i16m1(v2374, v2373, v2257, 8);
          v2191 = v2375;
          vint16m1_t v2376 = v2199;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2377 = __riscv_vwmacc_vx_i16m1(v2376, v2373, v2293, 8);
          v2199 = v2377;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v2378 = v2221 + 595;
          const int8_t* v2379 = (const int8_t*) v2378;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2380 = *(const int8_t *)(v2379);
          vint16m1_t v2381 = v2201;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2382 = __riscv_vwmacc_vx_i16m1(v2381, v2380, v2236, 8);
          v2201 = v2382;
          vint16m1_t v2383 = v2209;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2384 = __riscv_vwmacc_vx_i16m1(v2383, v2380, v2272, 8);
          v2209 = v2384;
          const uint8_t* v2385 = v2221 + 723;
          const int8_t* v2386 = (const int8_t*) v2385;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2387 = *(const int8_t *)(v2386);
          vint16m1_t v2388 = v2203;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2389 = __riscv_vwmacc_vx_i16m1(v2388, v2387, v2243, 8);
          v2203 = v2389;
          vint16m1_t v2390 = v2211;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2391 = __riscv_vwmacc_vx_i16m1(v2390, v2387, v2279, 8);
          v2211 = v2391;
          const uint8_t* v2392 = v2221 + 851;
          const int8_t* v2393 = (const int8_t*) v2392;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2394 = *(const int8_t *)(v2393);
          vint16m1_t v2395 = v2205;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2396 = __riscv_vwmacc_vx_i16m1(v2395, v2394, v2250, 8);
          v2205 = v2396;
          vint16m1_t v2397 = v2213;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2398 = __riscv_vwmacc_vx_i16m1(v2397, v2394, v2286, 8);
          v2213 = v2398;
          const uint8_t* v2399 = v2221 + 979;
          const int8_t* v2400 = (const int8_t*) v2399;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2401 = *(const int8_t *)(v2400);
          vint16m1_t v2402 = v2207;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2403 = __riscv_vwmacc_vx_i16m1(v2402, v2401, v2257, 8);
          v2207 = v2403;
          vint16m1_t v2404 = v2215;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2405 = __riscv_vwmacc_vx_i16m1(v2404, v2401, v2293, 8);
          v2215 = v2405;
        }
        vint16m1_t v2406 = v2153;
        vint16m1_t v2407 = v2155;
        vint16m1_t v2408 = v2157;
        vint16m1_t v2409 = v2159;
        vint16m1_t v2410 = v2161;
        vint16m1_t v2411 = v2163;
        vint16m1_t v2412 = v2165;
        vint16m1_t v2413 = v2167;
        vint16m1_t v2414 = v2169;
        vint16m1_t v2415 = v2171;
        vint16m1_t v2416 = v2173;
        vint16m1_t v2417 = v2175;
        vint16m1_t v2418 = v2177;
        vint16m1_t v2419 = v2179;
        vint16m1_t v2420 = v2181;
        vint16m1_t v2421 = v2183;
        vint16m1_t v2422 = v2185;
        vint16m1_t v2423 = v2187;
        vint16m1_t v2424 = v2189;
        vint16m1_t v2425 = v2191;
        vint16m1_t v2426 = v2193;
        vint16m1_t v2427 = v2195;
        vint16m1_t v2428 = v2197;
        vint16m1_t v2429 = v2199;
        vint16m1_t v2430 = v2201;
        vint16m1_t v2431 = v2203;
        vint16m1_t v2432 = v2205;
        vint16m1_t v2433 = v2207;
        vint16m1_t v2434 = v2209;
        vint16m1_t v2435 = v2211;
        vint16m1_t v2436 = v2213;
        vint16m1_t v2437 = v2215;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v2438 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2439 = __riscv_vwmacc_vv_i32m2(v2438, v2124, v2406, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2440 = __riscv_vwmacc_vv_i32m2(v2439, v2128, v2407, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2441 = __riscv_vwmacc_vv_i32m2(v2440, v2132, v2408, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2442 = __riscv_vwmacc_vv_i32m2(v2441, v2136, v2409, 8);
        v59 = v2442;
        vint32m2_t v2443 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2444 = __riscv_vwmacc_vv_i32m2(v2443, v2140, v2410, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2445 = __riscv_vwmacc_vv_i32m2(v2444, v2144, v2411, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2446 = __riscv_vwmacc_vv_i32m2(v2445, v2148, v2412, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2447 = __riscv_vwmacc_vv_i32m2(v2446, v2152, v2413, 8);
        v61 = v2447;
        vint32m2_t v2448 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2449 = __riscv_vwmacc_vv_i32m2(v2448, v2124, v2414, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2450 = __riscv_vwmacc_vv_i32m2(v2449, v2128, v2415, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2451 = __riscv_vwmacc_vv_i32m2(v2450, v2132, v2416, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2452 = __riscv_vwmacc_vv_i32m2(v2451, v2136, v2417, 8);
        v63 = v2452;
        vint32m2_t v2453 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2454 = __riscv_vwmacc_vv_i32m2(v2453, v2140, v2418, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2455 = __riscv_vwmacc_vv_i32m2(v2454, v2144, v2419, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2456 = __riscv_vwmacc_vv_i32m2(v2455, v2148, v2420, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2457 = __riscv_vwmacc_vv_i32m2(v2456, v2152, v2421, 8);
        v65 = v2457;
        vint32m2_t v2458 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2459 = __riscv_vwmacc_vv_i32m2(v2458, v2124, v2422, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2460 = __riscv_vwmacc_vv_i32m2(v2459, v2128, v2423, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2461 = __riscv_vwmacc_vv_i32m2(v2460, v2132, v2424, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2462 = __riscv_vwmacc_vv_i32m2(v2461, v2136, v2425, 8);
        v67 = v2462;
        vint32m2_t v2463 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2464 = __riscv_vwmacc_vv_i32m2(v2463, v2140, v2426, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2465 = __riscv_vwmacc_vv_i32m2(v2464, v2144, v2427, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2466 = __riscv_vwmacc_vv_i32m2(v2465, v2148, v2428, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2467 = __riscv_vwmacc_vv_i32m2(v2466, v2152, v2429, 8);
        v69 = v2467;
        vint32m2_t v2468 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2469 = __riscv_vwmacc_vv_i32m2(v2468, v2124, v2430, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2470 = __riscv_vwmacc_vv_i32m2(v2469, v2128, v2431, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2471 = __riscv_vwmacc_vv_i32m2(v2470, v2132, v2432, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2472 = __riscv_vwmacc_vv_i32m2(v2471, v2136, v2433, 8);
        v71 = v2472;
        vint32m2_t v2473 = v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2474 = __riscv_vwmacc_vv_i32m2(v2473, v2140, v2434, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2475 = __riscv_vwmacc_vv_i32m2(v2474, v2144, v2435, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2476 = __riscv_vwmacc_vv_i32m2(v2475, v2148, v2436, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2477 = __riscv_vwmacc_vv_i32m2(v2476, v2152, v2437, 8);
        v73 = v2477;
        vint16m1_t v2478;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2479 = __riscv_vmv_v_x_i16m1(0, 8);
        v2478 = v2479;
        vint16m1_t v2480;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2481 = __riscv_vmv_v_x_i16m1(0, 8);
        v2480 = v2481;
        vint16m1_t v2482;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2483 = __riscv_vmv_v_x_i16m1(0, 8);
        v2482 = v2483;
        vint16m1_t v2484;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2485 = __riscv_vmv_v_x_i16m1(0, 8);
        v2484 = v2485;
        vint16m1_t v2486;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2487 = __riscv_vmv_v_x_i16m1(0, 8);
        v2486 = v2487;
        vint16m1_t v2488;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2489 = __riscv_vmv_v_x_i16m1(0, 8);
        v2488 = v2489;
        vint16m1_t v2490;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2491 = __riscv_vmv_v_x_i16m1(0, 8);
        v2490 = v2491;
        vint16m1_t v2492;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2493 = __riscv_vmv_v_x_i16m1(0, 8);
        v2492 = v2493;
        vint16m1_t v2494;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2495 = __riscv_vmv_v_x_i16m1(0, 8);
        v2494 = v2495;
        vint16m1_t v2496;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2497 = __riscv_vmv_v_x_i16m1(0, 8);
        v2496 = v2497;
        vint16m1_t v2498;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2499 = __riscv_vmv_v_x_i16m1(0, 8);
        v2498 = v2499;
        vint16m1_t v2500;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2501 = __riscv_vmv_v_x_i16m1(0, 8);
        v2500 = v2501;
        vint16m1_t v2502;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2503 = __riscv_vmv_v_x_i16m1(0, 8);
        v2502 = v2503;
        vint16m1_t v2504;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2505 = __riscv_vmv_v_x_i16m1(0, 8);
        v2504 = v2505;
        vint16m1_t v2506;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2507 = __riscv_vmv_v_x_i16m1(0, 8);
        v2506 = v2507;
        vint16m1_t v2508;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2509 = __riscv_vmv_v_x_i16m1(0, 8);
        v2508 = v2509;
        vint16m1_t v2510;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2511 = __riscv_vmv_v_x_i16m1(0, 8);
        v2510 = v2511;
        vint16m1_t v2512;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2513 = __riscv_vmv_v_x_i16m1(0, 8);
        v2512 = v2513;
        vint16m1_t v2514;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2515 = __riscv_vmv_v_x_i16m1(0, 8);
        v2514 = v2515;
        vint16m1_t v2516;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2517 = __riscv_vmv_v_x_i16m1(0, 8);
        v2516 = v2517;
        vint16m1_t v2518;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2519 = __riscv_vmv_v_x_i16m1(0, 8);
        v2518 = v2519;
        vint16m1_t v2520;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2521 = __riscv_vmv_v_x_i16m1(0, 8);
        v2520 = v2521;
        vint16m1_t v2522;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2523 = __riscv_vmv_v_x_i16m1(0, 8);
        v2522 = v2523;
        vint16m1_t v2524;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2525 = __riscv_vmv_v_x_i16m1(0, 8);
        v2524 = v2525;
        vint16m1_t v2526;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2527 = __riscv_vmv_v_x_i16m1(0, 8);
        v2526 = v2527;
        vint16m1_t v2528;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2529 = __riscv_vmv_v_x_i16m1(0, 8);
        v2528 = v2529;
        vint16m1_t v2530;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2531 = __riscv_vmv_v_x_i16m1(0, 8);
        v2530 = v2531;
        vint16m1_t v2532;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2533 = __riscv_vmv_v_x_i16m1(0, 8);
        v2532 = v2533;
        vint16m1_t v2534;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2535 = __riscv_vmv_v_x_i16m1(0, 8);
        v2534 = v2535;
        vint16m1_t v2536;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2537 = __riscv_vmv_v_x_i16m1(0, 8);
        v2536 = v2537;
        vint16m1_t v2538;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2539 = __riscv_vmv_v_x_i16m1(0, 8);
        v2538 = v2539;
        vint16m1_t v2540;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2541 = __riscv_vmv_v_x_i16m1(0, 8);
        v2540 = v2541;
        for (size_t v2542 = 0; v2542 < 8; v2542 += 1) {
          size_t v2543 = v2542 * 16;
          const uint8_t* v2544 = v38 + v2543;
          size_t v2545 = v2542 * 4;
          const uint8_t* v2546 = v40 + v2545;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v2547 = v2544 + 2720;
          const uint8_t* v2548 = (const uint8_t*) v2547;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v2549 = __riscv_vle8_v_u8mf2(v2548, 8);
          const uint8_t* v2550 = v2544 + 3232;
          const uint8_t* v2551 = (const uint8_t*) v2550;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v2552 = __riscv_vle8_v_u8mf2(v2551, 8);
          const uint8_t* v2553 = v2544 + 1184;
          const uint8_t* v2554 = (const uint8_t*) v2553;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v2555 = __riscv_vle8_v_u8mf2(v2554, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2556 = __riscv_vand_vx_u8mf2(v2549, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2557 = __riscv_vand_vx_u8mf2(v2555, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2558 = __riscv_vsll_vx_u8mf2(v2557, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2559 = __riscv_vor_vv_u8mf2(v2556, v2558, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2560 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2559);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2561 = __riscv_vsub_vx_i8mf2(v2560, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2562 = __riscv_vand_vx_u8mf2(v2552, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2563 = __riscv_vsrl_vx_u8mf2(v2555, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2564 = __riscv_vand_vx_u8mf2(v2563, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2565 = __riscv_vsll_vx_u8mf2(v2564, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2566 = __riscv_vor_vv_u8mf2(v2562, v2565, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2567 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2566);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2568 = __riscv_vsub_vx_i8mf2(v2567, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2569 = __riscv_vsrl_vx_u8mf2(v2549, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2570 = __riscv_vsrl_vx_u8mf2(v2555, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2571 = __riscv_vand_vx_u8mf2(v2570, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2572 = __riscv_vsll_vx_u8mf2(v2571, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2573 = __riscv_vor_vv_u8mf2(v2569, v2572, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2574 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2573);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2575 = __riscv_vsub_vx_i8mf2(v2574, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2576 = __riscv_vsrl_vx_u8mf2(v2552, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2577 = __riscv_vsrl_vx_u8mf2(v2555, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2578 = __riscv_vand_vx_u8mf2(v2577, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2579 = __riscv_vsll_vx_u8mf2(v2578, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2580 = __riscv_vor_vv_u8mf2(v2576, v2579, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2581 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2580);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2582 = __riscv_vsub_vx_i8mf2(v2581, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_ql_qh_addr
          const uint8_t* v2583 = v2544 + 2728;
          const uint8_t* v2584 = (const uint8_t*) v2583;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v2585 = __riscv_vle8_v_u8mf2(v2584, 8);
          const uint8_t* v2586 = v2544 + 3240;
          const uint8_t* v2587 = (const uint8_t*) v2586;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v2588 = __riscv_vle8_v_u8mf2(v2587, 8);
          const uint8_t* v2589 = v2544 + 1192;
          const uint8_t* v2590 = (const uint8_t*) v2589;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v2591 = __riscv_vle8_v_u8mf2(v2590, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2592 = __riscv_vand_vx_u8mf2(v2585, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2593 = __riscv_vand_vx_u8mf2(v2591, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2594 = __riscv_vsll_vx_u8mf2(v2593, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2595 = __riscv_vor_vv_u8mf2(v2592, v2594, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2596 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2595);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2597 = __riscv_vsub_vx_i8mf2(v2596, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2598 = __riscv_vand_vx_u8mf2(v2588, 0x0F, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2599 = __riscv_vsrl_vx_u8mf2(v2591, 2, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2600 = __riscv_vand_vx_u8mf2(v2599, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2601 = __riscv_vsll_vx_u8mf2(v2600, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2602 = __riscv_vor_vv_u8mf2(v2598, v2601, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2603 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2602);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2604 = __riscv_vsub_vx_i8mf2(v2603, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2605 = __riscv_vsrl_vx_u8mf2(v2585, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2606 = __riscv_vsrl_vx_u8mf2(v2591, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2607 = __riscv_vand_vx_u8mf2(v2606, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2608 = __riscv_vsll_vx_u8mf2(v2607, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2609 = __riscv_vor_vv_u8mf2(v2605, v2608, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2610 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2609);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2611 = __riscv_vsub_vx_i8mf2(v2610, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2612 = __riscv_vsrl_vx_u8mf2(v2588, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v2613 = __riscv_vsrl_vx_u8mf2(v2591, 6, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v2614 = __riscv_vand_vx_u8mf2(v2613, 0x03, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v2615 = __riscv_vsll_vx_u8mf2(v2614, 4, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v2616 = __riscv_vor_vv_u8mf2(v2612, v2615, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v2617 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2616);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
          vint8mf2_t v2618 = __riscv_vsub_vx_i8mf2(v2617, 32, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v2619 = v2546 + 624;
          const int8_t* v2620 = (const int8_t*) v2619;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2621 = *(const int8_t *)(v2620);
          vint16m1_t v2622 = v2478;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2623 = __riscv_vwmacc_vx_i16m1(v2622, v2621, v2561, 8);
          v2478 = v2623;
          vint16m1_t v2624 = v2486;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2625 = __riscv_vwmacc_vx_i16m1(v2624, v2621, v2597, 8);
          v2486 = v2625;
          const uint8_t* v2626 = v2546 + 752;
          const int8_t* v2627 = (const int8_t*) v2626;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2628 = *(const int8_t *)(v2627);
          vint16m1_t v2629 = v2480;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2630 = __riscv_vwmacc_vx_i16m1(v2629, v2628, v2568, 8);
          v2480 = v2630;
          vint16m1_t v2631 = v2488;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2632 = __riscv_vwmacc_vx_i16m1(v2631, v2628, v2604, 8);
          v2488 = v2632;
          const uint8_t* v2633 = v2546 + 880;
          const int8_t* v2634 = (const int8_t*) v2633;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2635 = *(const int8_t *)(v2634);
          vint16m1_t v2636 = v2482;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2637 = __riscv_vwmacc_vx_i16m1(v2636, v2635, v2575, 8);
          v2482 = v2637;
          vint16m1_t v2638 = v2490;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2639 = __riscv_vwmacc_vx_i16m1(v2638, v2635, v2611, 8);
          v2490 = v2639;
          const uint8_t* v2640 = v2546 + 1008;
          const int8_t* v2641 = (const int8_t*) v2640;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2642 = *(const int8_t *)(v2641);
          vint16m1_t v2643 = v2484;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2644 = __riscv_vwmacc_vx_i16m1(v2643, v2642, v2582, 8);
          v2484 = v2644;
          vint16m1_t v2645 = v2492;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2646 = __riscv_vwmacc_vx_i16m1(v2645, v2642, v2618, 8);
          v2492 = v2646;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v2647 = v2546 + 625;
          const int8_t* v2648 = (const int8_t*) v2647;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2649 = *(const int8_t *)(v2648);
          vint16m1_t v2650 = v2494;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2651 = __riscv_vwmacc_vx_i16m1(v2650, v2649, v2561, 8);
          v2494 = v2651;
          vint16m1_t v2652 = v2502;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2653 = __riscv_vwmacc_vx_i16m1(v2652, v2649, v2597, 8);
          v2502 = v2653;
          const uint8_t* v2654 = v2546 + 753;
          const int8_t* v2655 = (const int8_t*) v2654;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2656 = *(const int8_t *)(v2655);
          vint16m1_t v2657 = v2496;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2658 = __riscv_vwmacc_vx_i16m1(v2657, v2656, v2568, 8);
          v2496 = v2658;
          vint16m1_t v2659 = v2504;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2660 = __riscv_vwmacc_vx_i16m1(v2659, v2656, v2604, 8);
          v2504 = v2660;
          const uint8_t* v2661 = v2546 + 881;
          const int8_t* v2662 = (const int8_t*) v2661;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2663 = *(const int8_t *)(v2662);
          vint16m1_t v2664 = v2498;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2665 = __riscv_vwmacc_vx_i16m1(v2664, v2663, v2575, 8);
          v2498 = v2665;
          vint16m1_t v2666 = v2506;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2667 = __riscv_vwmacc_vx_i16m1(v2666, v2663, v2611, 8);
          v2506 = v2667;
          const uint8_t* v2668 = v2546 + 1009;
          const int8_t* v2669 = (const int8_t*) v2668;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2670 = *(const int8_t *)(v2669);
          vint16m1_t v2671 = v2500;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2672 = __riscv_vwmacc_vx_i16m1(v2671, v2670, v2582, 8);
          v2500 = v2672;
          vint16m1_t v2673 = v2508;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2674 = __riscv_vwmacc_vx_i16m1(v2673, v2670, v2618, 8);
          v2508 = v2674;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v2675 = v2546 + 626;
          const int8_t* v2676 = (const int8_t*) v2675;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2677 = *(const int8_t *)(v2676);
          vint16m1_t v2678 = v2510;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2679 = __riscv_vwmacc_vx_i16m1(v2678, v2677, v2561, 8);
          v2510 = v2679;
          vint16m1_t v2680 = v2518;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2681 = __riscv_vwmacc_vx_i16m1(v2680, v2677, v2597, 8);
          v2518 = v2681;
          const uint8_t* v2682 = v2546 + 754;
          const int8_t* v2683 = (const int8_t*) v2682;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2684 = *(const int8_t *)(v2683);
          vint16m1_t v2685 = v2512;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2686 = __riscv_vwmacc_vx_i16m1(v2685, v2684, v2568, 8);
          v2512 = v2686;
          vint16m1_t v2687 = v2520;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2688 = __riscv_vwmacc_vx_i16m1(v2687, v2684, v2604, 8);
          v2520 = v2688;
          const uint8_t* v2689 = v2546 + 882;
          const int8_t* v2690 = (const int8_t*) v2689;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2691 = *(const int8_t *)(v2690);
          vint16m1_t v2692 = v2514;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2693 = __riscv_vwmacc_vx_i16m1(v2692, v2691, v2575, 8);
          v2514 = v2693;
          vint16m1_t v2694 = v2522;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2695 = __riscv_vwmacc_vx_i16m1(v2694, v2691, v2611, 8);
          v2522 = v2695;
          const uint8_t* v2696 = v2546 + 1010;
          const int8_t* v2697 = (const int8_t*) v2696;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2698 = *(const int8_t *)(v2697);
          vint16m1_t v2699 = v2516;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2700 = __riscv_vwmacc_vx_i16m1(v2699, v2698, v2582, 8);
          v2516 = v2700;
          vint16m1_t v2701 = v2524;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2702 = __riscv_vwmacc_vx_i16m1(v2701, v2698, v2618, 8);
          v2524 = v2702;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          const uint8_t* v2703 = v2546 + 627;
          const int8_t* v2704 = (const int8_t*) v2703;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2705 = *(const int8_t *)(v2704);
          vint16m1_t v2706 = v2526;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2707 = __riscv_vwmacc_vx_i16m1(v2706, v2705, v2561, 8);
          v2526 = v2707;
          vint16m1_t v2708 = v2534;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2709 = __riscv_vwmacc_vx_i16m1(v2708, v2705, v2597, 8);
          v2534 = v2709;
          const uint8_t* v2710 = v2546 + 755;
          const int8_t* v2711 = (const int8_t*) v2710;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2712 = *(const int8_t *)(v2711);
          vint16m1_t v2713 = v2528;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2714 = __riscv_vwmacc_vx_i16m1(v2713, v2712, v2568, 8);
          v2528 = v2714;
          vint16m1_t v2715 = v2536;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2716 = __riscv_vwmacc_vx_i16m1(v2715, v2712, v2604, 8);
          v2536 = v2716;
          const uint8_t* v2717 = v2546 + 883;
          const int8_t* v2718 = (const int8_t*) v2717;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2719 = *(const int8_t *)(v2718);
          vint16m1_t v2720 = v2530;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2721 = __riscv_vwmacc_vx_i16m1(v2720, v2719, v2575, 8);
          v2530 = v2721;
          vint16m1_t v2722 = v2538;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2723 = __riscv_vwmacc_vx_i16m1(v2722, v2719, v2611, 8);
          v2538 = v2723;
          const uint8_t* v2724 = v2546 + 1011;
          const int8_t* v2725 = (const int8_t*) v2724;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v2726 = *(const int8_t *)(v2725);
          vint16m1_t v2727 = v2532;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2728 = __riscv_vwmacc_vx_i16m1(v2727, v2726, v2582, 8);
          v2532 = v2728;
          vint16m1_t v2729 = v2540;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v2730 = __riscv_vwmacc_vx_i16m1(v2729, v2726, v2618, 8);
          v2540 = v2730;
        }
        vint16m1_t v2731 = v2478;
        vint16m1_t v2732 = v2480;
        vint16m1_t v2733 = v2482;
        vint16m1_t v2734 = v2484;
        vint16m1_t v2735 = v2486;
        vint16m1_t v2736 = v2488;
        vint16m1_t v2737 = v2490;
        vint16m1_t v2738 = v2492;
        vint16m1_t v2739 = v2494;
        vint16m1_t v2740 = v2496;
        vint16m1_t v2741 = v2498;
        vint16m1_t v2742 = v2500;
        vint16m1_t v2743 = v2502;
        vint16m1_t v2744 = v2504;
        vint16m1_t v2745 = v2506;
        vint16m1_t v2746 = v2508;
        vint16m1_t v2747 = v2510;
        vint16m1_t v2748 = v2512;
        vint16m1_t v2749 = v2514;
        vint16m1_t v2750 = v2516;
        vint16m1_t v2751 = v2518;
        vint16m1_t v2752 = v2520;
        vint16m1_t v2753 = v2522;
        vint16m1_t v2754 = v2524;
        vint16m1_t v2755 = v2526;
        vint16m1_t v2756 = v2528;
        vint16m1_t v2757 = v2530;
        vint16m1_t v2758 = v2532;
        vint16m1_t v2759 = v2534;
        vint16m1_t v2760 = v2536;
        vint16m1_t v2761 = v2538;
        vint16m1_t v2762 = v2540;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        vint32m2_t v2763 = v59;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2764 = __riscv_vwmacc_vv_i32m2(v2763, v2124, v2731, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2765 = __riscv_vwmacc_vv_i32m2(v2764, v2128, v2732, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2766 = __riscv_vwmacc_vv_i32m2(v2765, v2132, v2733, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2767 = __riscv_vwmacc_vv_i32m2(v2766, v2136, v2734, 8);
        v59 = v2767;
        vint32m2_t v2768 = v61;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2769 = __riscv_vwmacc_vv_i32m2(v2768, v2140, v2735, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2770 = __riscv_vwmacc_vv_i32m2(v2769, v2144, v2736, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2771 = __riscv_vwmacc_vv_i32m2(v2770, v2148, v2737, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2772 = __riscv_vwmacc_vv_i32m2(v2771, v2152, v2738, 8);
        v61 = v2772;
        vint32m2_t v2773 = v63;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2774 = __riscv_vwmacc_vv_i32m2(v2773, v2124, v2739, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2775 = __riscv_vwmacc_vv_i32m2(v2774, v2128, v2740, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2776 = __riscv_vwmacc_vv_i32m2(v2775, v2132, v2741, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2777 = __riscv_vwmacc_vv_i32m2(v2776, v2136, v2742, 8);
        v63 = v2777;
        vint32m2_t v2778 = v65;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2779 = __riscv_vwmacc_vv_i32m2(v2778, v2140, v2743, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2780 = __riscv_vwmacc_vv_i32m2(v2779, v2144, v2744, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2781 = __riscv_vwmacc_vv_i32m2(v2780, v2148, v2745, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2782 = __riscv_vwmacc_vv_i32m2(v2781, v2152, v2746, 8);
        v65 = v2782;
        vint32m2_t v2783 = v67;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2784 = __riscv_vwmacc_vv_i32m2(v2783, v2124, v2747, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2785 = __riscv_vwmacc_vv_i32m2(v2784, v2128, v2748, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2786 = __riscv_vwmacc_vv_i32m2(v2785, v2132, v2749, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2787 = __riscv_vwmacc_vv_i32m2(v2786, v2136, v2750, 8);
        v67 = v2787;
        vint32m2_t v2788 = v69;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2789 = __riscv_vwmacc_vv_i32m2(v2788, v2140, v2751, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2790 = __riscv_vwmacc_vv_i32m2(v2789, v2144, v2752, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2791 = __riscv_vwmacc_vv_i32m2(v2790, v2148, v2753, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2792 = __riscv_vwmacc_vv_i32m2(v2791, v2152, v2754, 8);
        v69 = v2792;
        vint32m2_t v2793 = v71;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2794 = __riscv_vwmacc_vv_i32m2(v2793, v2124, v2755, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2795 = __riscv_vwmacc_vv_i32m2(v2794, v2128, v2756, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2796 = __riscv_vwmacc_vv_i32m2(v2795, v2132, v2757, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2797 = __riscv_vwmacc_vv_i32m2(v2796, v2136, v2758, 8);
        v71 = v2797;
        vint32m2_t v2798 = v73;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2799 = __riscv_vwmacc_vv_i32m2(v2798, v2140, v2759, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2800 = __riscv_vwmacc_vv_i32m2(v2799, v2144, v2760, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2801 = __riscv_vwmacc_vv_i32m2(v2800, v2148, v2761, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2802 = __riscv_vwmacc_vv_i32m2(v2801, v2152, v2762, 8);
        v73 = v2802;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v2803 = __riscv_vfmul_vf_f32m2(v54, v42, 8);
        vint32m2_t v2804 = v59;
        vfloat32m2_t v2805 = v20;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v2806 = __riscv_vfcvt_f_x_v_f32m2(v2804, 8);
        vfloat32m2_t v2807 = __riscv_vfmacc_vv_f32m2(v2805, v2806, v2803, 8);
        v20 = v2807;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v2808 = __riscv_vfmul_vf_f32m2(v58, v42, 8);
        vint32m2_t v2809 = v61;
        vfloat32m2_t v2810 = v22;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v2811 = __riscv_vfcvt_f_x_v_f32m2(v2809, 8);
        vfloat32m2_t v2812 = __riscv_vfmacc_vv_f32m2(v2810, v2811, v2808, 8);
        v22 = v2812;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v2813 = __riscv_vfmul_vf_f32m2(v54, v45, 8);
        vint32m2_t v2814 = v63;
        vfloat32m2_t v2815 = v24;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v2816 = __riscv_vfcvt_f_x_v_f32m2(v2814, 8);
        vfloat32m2_t v2817 = __riscv_vfmacc_vv_f32m2(v2815, v2816, v2813, 8);
        v24 = v2817;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v2818 = __riscv_vfmul_vf_f32m2(v58, v45, 8);
        vint32m2_t v2819 = v65;
        vfloat32m2_t v2820 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v2821 = __riscv_vfcvt_f_x_v_f32m2(v2819, 8);
        vfloat32m2_t v2822 = __riscv_vfmacc_vv_f32m2(v2820, v2821, v2818, 8);
        v26 = v2822;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v2823 = __riscv_vfmul_vf_f32m2(v54, v48, 8);
        vint32m2_t v2824 = v67;
        vfloat32m2_t v2825 = v28;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v2826 = __riscv_vfcvt_f_x_v_f32m2(v2824, 8);
        vfloat32m2_t v2827 = __riscv_vfmacc_vv_f32m2(v2825, v2826, v2823, 8);
        v28 = v2827;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v2828 = __riscv_vfmul_vf_f32m2(v58, v48, 8);
        vint32m2_t v2829 = v69;
        vfloat32m2_t v2830 = v30;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v2831 = __riscv_vfcvt_f_x_v_f32m2(v2829, 8);
        vfloat32m2_t v2832 = __riscv_vfmacc_vv_f32m2(v2830, v2831, v2828, 8);
        v30 = v2832;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v2833 = __riscv_vfmul_vf_f32m2(v54, v51, 8);
        vint32m2_t v2834 = v71;
        vfloat32m2_t v2835 = v32;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v2836 = __riscv_vfcvt_f_x_v_f32m2(v2834, 8);
        vfloat32m2_t v2837 = __riscv_vfmacc_vv_f32m2(v2835, v2836, v2833, 8);
        v32 = v2837;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v2838 = __riscv_vfmul_vf_f32m2(v58, v51, 8);
        vint32m2_t v2839 = v73;
        vfloat32m2_t v2840 = v34;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v2841 = __riscv_vfcvt_f_x_v_f32m2(v2839, 8);
        vfloat32m2_t v2842 = __riscv_vfmacc_vv_f32m2(v2840, v2841, v2838, 8);
        v34 = v2842;
      }
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v2843 = v12 * 4;
      size_t v2844 = v2843 + 0;
      size_t v2845 = v2844 * v7;
      size_t v2846 = v16 * 16;
      size_t v2847 = v2845 + v2846;
      float* v2848 = v2 + v2847;
      vfloat32m2_t v2849 = v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v2848, v2849, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v2850 = v12 * 4;
      size_t v2851 = v2850 + 0;
      size_t v2852 = v2851 * v7;
      size_t v2853 = v16 * 16;
      size_t v2854 = v2852 + v2853;
      size_t v2855 = v2854 + 8;
      float* v2856 = v2 + v2855;
      vfloat32m2_t v2857 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v2856, v2857, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v2858 = v12 * 4;
      size_t v2859 = v2858 + 1;
      size_t v2860 = v2859 * v7;
      size_t v2861 = v16 * 16;
      size_t v2862 = v2860 + v2861;
      float* v2863 = v2 + v2862;
      vfloat32m2_t v2864 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v2863, v2864, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v2865 = v12 * 4;
      size_t v2866 = v2865 + 1;
      size_t v2867 = v2866 * v7;
      size_t v2868 = v16 * 16;
      size_t v2869 = v2867 + v2868;
      size_t v2870 = v2869 + 8;
      float* v2871 = v2 + v2870;
      vfloat32m2_t v2872 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v2871, v2872, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v2873 = v12 * 4;
      size_t v2874 = v2873 + 2;
      size_t v2875 = v2874 * v7;
      size_t v2876 = v16 * 16;
      size_t v2877 = v2875 + v2876;
      float* v2878 = v2 + v2877;
      vfloat32m2_t v2879 = v28;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v2878, v2879, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v2880 = v12 * 4;
      size_t v2881 = v2880 + 2;
      size_t v2882 = v2881 * v7;
      size_t v2883 = v16 * 16;
      size_t v2884 = v2882 + v2883;
      size_t v2885 = v2884 + 8;
      float* v2886 = v2 + v2885;
      vfloat32m2_t v2887 = v30;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v2886, v2887, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v2888 = v12 * 4;
      size_t v2889 = v2888 + 3;
      size_t v2890 = v2889 * v7;
      size_t v2891 = v16 * 16;
      size_t v2892 = v2890 + v2891;
      float* v2893 = v2 + v2892;
      vfloat32m2_t v2894 = v32;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v2893, v2894, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v2895 = v12 * 4;
      size_t v2896 = v2895 + 3;
      size_t v2897 = v2896 * v7;
      size_t v2898 = v16 * 16;
      size_t v2899 = v2897 + v2898;
      size_t v2900 = v2899 + 8;
      float* v2901 = v2 + v2900;
      vfloat32m2_t v2902 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v2901, v2902, 8);
    }
  }
  return;
}


