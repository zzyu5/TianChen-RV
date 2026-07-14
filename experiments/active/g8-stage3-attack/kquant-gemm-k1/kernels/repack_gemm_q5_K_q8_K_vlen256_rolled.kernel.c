#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5, size_t v6, size_t v7) {
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
      size_t v18 = v17 * 2816;
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
        size_t v29 = v28 * 2816;
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
        int16_t v44[64];
        int16_t v45[64];
        int32_t v46[64];
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
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
        const uint8_t* v55 = v30 + 64;
        const uint8_t* v56 = (const uint8_t*) v55;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v57 = __riscv_vle8_v_u8mf2(v56, 16);
        const uint8_t* v58 = v30 + 192;
        const uint8_t* v59 = (const uint8_t*) v58;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v60 = __riscv_vle8_v_u8mf2(v59, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v61 = __riscv_vand_vx_u8mf2(v57, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v62 = __riscv_vsrl_vx_u8mf2(v57, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v63 = __riscv_vand_vx_u8mf2(v60, 0x03, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v64 = __riscv_vsll_vx_u8mf2(v63, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v65 = __riscv_vand_vx_u8mf2(v60, 0x0C, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v66 = __riscv_vsll_vx_u8mf2(v65, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v67 = __riscv_vor_vv_u8mf2(v64, v61, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v68 = __riscv_vor_vv_u8mf2(v66, v62, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v69 = __riscv_vzext_vf2_u16m1(v67, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v70 = __riscv_vreinterpret_v_u16m1_i16m1(v69);
        int16_t* v71 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v71, v70, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v72 = __riscv_vzext_vf2_u16m1(v68, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v73 = __riscv_vreinterpret_v_u16m1_i16m1(v72);
        int16_t* v74 = &v45[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v74, v73, 16);
        const uint8_t* v75 = v30 + 80;
        const uint8_t* v76 = (const uint8_t*) v75;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v77 = __riscv_vle8_v_u8mf2(v76, 16);
        const uint8_t* v78 = v30 + 208;
        const uint8_t* v79 = (const uint8_t*) v78;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v80 = __riscv_vle8_v_u8mf2(v79, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v81 = __riscv_vand_vx_u8mf2(v77, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v82 = __riscv_vsrl_vx_u8mf2(v77, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v83 = __riscv_vand_vx_u8mf2(v80, 0x03, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v84 = __riscv_vsll_vx_u8mf2(v83, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v85 = __riscv_vand_vx_u8mf2(v80, 0x0C, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v86 = __riscv_vsll_vx_u8mf2(v85, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v87 = __riscv_vor_vv_u8mf2(v84, v81, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v88 = __riscv_vor_vv_u8mf2(v86, v82, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v89 = __riscv_vzext_vf2_u16m1(v87, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v90 = __riscv_vreinterpret_v_u16m1_i16m1(v89);
        int16_t* v91 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v91, v90, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v92 = __riscv_vzext_vf2_u16m1(v88, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v93 = __riscv_vreinterpret_v_u16m1_i16m1(v92);
        int16_t* v94 = &v45[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v94, v93, 16);
        const uint8_t* v95 = v30 + 96;
        const uint8_t* v96 = (const uint8_t*) v95;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v97 = __riscv_vle8_v_u8mf2(v96, 16);
        const uint8_t* v98 = v30 + 224;
        const uint8_t* v99 = (const uint8_t*) v98;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v100 = __riscv_vle8_v_u8mf2(v99, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v101 = __riscv_vand_vx_u8mf2(v97, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v102 = __riscv_vsrl_vx_u8mf2(v97, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v103 = __riscv_vand_vx_u8mf2(v100, 0x03, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v104 = __riscv_vsll_vx_u8mf2(v103, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v105 = __riscv_vand_vx_u8mf2(v100, 0x0C, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v106 = __riscv_vsll_vx_u8mf2(v105, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v107 = __riscv_vor_vv_u8mf2(v104, v101, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v108 = __riscv_vor_vv_u8mf2(v106, v102, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v109 = __riscv_vzext_vf2_u16m1(v107, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v110 = __riscv_vreinterpret_v_u16m1_i16m1(v109);
        int16_t* v111 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v111, v110, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v112 = __riscv_vzext_vf2_u16m1(v108, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v113 = __riscv_vreinterpret_v_u16m1_i16m1(v112);
        int16_t* v114 = &v45[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v114, v113, 16);
        const uint8_t* v115 = v30 + 112;
        const uint8_t* v116 = (const uint8_t*) v115;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v117 = __riscv_vle8_v_u8mf2(v116, 16);
        const uint8_t* v118 = v30 + 240;
        const uint8_t* v119 = (const uint8_t*) v118;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v120 = __riscv_vle8_v_u8mf2(v119, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v121 = __riscv_vand_vx_u8mf2(v117, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v122 = __riscv_vsrl_vx_u8mf2(v117, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v123 = __riscv_vand_vx_u8mf2(v120, 0x03, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v124 = __riscv_vsll_vx_u8mf2(v123, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v125 = __riscv_vand_vx_u8mf2(v120, 0x0C, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v126 = __riscv_vsll_vx_u8mf2(v125, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v127 = __riscv_vor_vv_u8mf2(v124, v121, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v128 = __riscv_vor_vv_u8mf2(v126, v122, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v129 = __riscv_vzext_vf2_u16m1(v127, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v130 = __riscv_vreinterpret_v_u16m1_i16m1(v129);
        int16_t* v131 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v131, v130, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v132 = __riscv_vzext_vf2_u16m1(v128, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v133 = __riscv_vreinterpret_v_u16m1_i16m1(v132);
        int16_t* v134 = &v45[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v134, v133, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold
        int16_t* v135 = &v45[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v136 = __riscv_vle16_v_i16m1(v135, 16);
        const uint8_t* v137 = v32 + 1040;
        const int16_t* v138 = (const int16_t*) v137;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v139 = *(const int16_t *)(v138);
        const uint8_t* v140 = v32 + 1048;
        const int16_t* v141 = (const int16_t*) v140;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v142 = *(const int16_t *)(v141);
        int32_t v143 = v139 + v142;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v144 = __riscv_vmv_v_x_i32m2(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v145 = __riscv_vwmacc_vx_i32m2(v144, v143, v136, 16);
        int32_t* v146 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v146, v145, 16);
        const uint8_t* v147 = v32 + 1042;
        const int16_t* v148 = (const int16_t*) v147;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v149 = *(const int16_t *)(v148);
        const uint8_t* v150 = v32 + 1050;
        const int16_t* v151 = (const int16_t*) v150;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v152 = *(const int16_t *)(v151);
        int32_t v153 = v149 + v152;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v154 = __riscv_vmv_v_x_i32m2(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v155 = __riscv_vwmacc_vx_i32m2(v154, v153, v136, 16);
        int32_t* v156 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v156, v155, 16);
        const uint8_t* v157 = v32 + 1044;
        const int16_t* v158 = (const int16_t*) v157;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v159 = *(const int16_t *)(v158);
        const uint8_t* v160 = v32 + 1052;
        const int16_t* v161 = (const int16_t*) v160;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v162 = *(const int16_t *)(v161);
        int32_t v163 = v159 + v162;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v164 = __riscv_vmv_v_x_i32m2(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v165 = __riscv_vwmacc_vx_i32m2(v164, v163, v136, 16);
        int32_t* v166 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v166, v165, 16);
        const uint8_t* v167 = v32 + 1046;
        const int16_t* v168 = (const int16_t*) v167;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v169 = *(const int16_t *)(v168);
        const uint8_t* v170 = v32 + 1054;
        const int16_t* v171 = (const int16_t*) v170;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v172 = *(const int16_t *)(v171);
        int32_t v173 = v169 + v172;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v174 = __riscv_vmv_v_x_i32m2(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v175 = __riscv_vwmacc_vx_i32m2(v174, v173, v136, 16);
        int32_t* v176 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v176, v175, 16);
        int16_t* v177 = &v45[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v178 = __riscv_vle16_v_i16m1(v177, 16);
        const uint8_t* v179 = v32 + 1056;
        const int16_t* v180 = (const int16_t*) v179;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v181 = *(const int16_t *)(v180);
        const uint8_t* v182 = v32 + 1064;
        const int16_t* v183 = (const int16_t*) v182;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v184 = *(const int16_t *)(v183);
        int32_t v185 = v181 + v184;
        int32_t* v186 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v187 = __riscv_vle32_v_i32m2(v186, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v188 = __riscv_vwmacc_vx_i32m2(v187, v185, v178, 16);
        int32_t* v189 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v189, v188, 16);
        const uint8_t* v190 = v32 + 1058;
        const int16_t* v191 = (const int16_t*) v190;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v192 = *(const int16_t *)(v191);
        const uint8_t* v193 = v32 + 1066;
        const int16_t* v194 = (const int16_t*) v193;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v195 = *(const int16_t *)(v194);
        int32_t v196 = v192 + v195;
        int32_t* v197 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v198 = __riscv_vle32_v_i32m2(v197, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v199 = __riscv_vwmacc_vx_i32m2(v198, v196, v178, 16);
        int32_t* v200 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v200, v199, 16);
        const uint8_t* v201 = v32 + 1060;
        const int16_t* v202 = (const int16_t*) v201;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v203 = *(const int16_t *)(v202);
        const uint8_t* v204 = v32 + 1068;
        const int16_t* v205 = (const int16_t*) v204;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v206 = *(const int16_t *)(v205);
        int32_t v207 = v203 + v206;
        int32_t* v208 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v209 = __riscv_vle32_v_i32m2(v208, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v210 = __riscv_vwmacc_vx_i32m2(v209, v207, v178, 16);
        int32_t* v211 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v211, v210, 16);
        const uint8_t* v212 = v32 + 1062;
        const int16_t* v213 = (const int16_t*) v212;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v214 = *(const int16_t *)(v213);
        const uint8_t* v215 = v32 + 1070;
        const int16_t* v216 = (const int16_t*) v215;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v217 = *(const int16_t *)(v216);
        int32_t v218 = v214 + v217;
        int32_t* v219 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v220 = __riscv_vle32_v_i32m2(v219, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v221 = __riscv_vwmacc_vx_i32m2(v220, v218, v178, 16);
        int32_t* v222 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v222, v221, 16);
        int16_t* v223 = &v45[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v224 = __riscv_vle16_v_i16m1(v223, 16);
        const uint8_t* v225 = v32 + 1072;
        const int16_t* v226 = (const int16_t*) v225;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v227 = *(const int16_t *)(v226);
        const uint8_t* v228 = v32 + 1080;
        const int16_t* v229 = (const int16_t*) v228;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v230 = *(const int16_t *)(v229);
        int32_t v231 = v227 + v230;
        int32_t* v232 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v233 = __riscv_vle32_v_i32m2(v232, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v234 = __riscv_vwmacc_vx_i32m2(v233, v231, v224, 16);
        int32_t* v235 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v235, v234, 16);
        const uint8_t* v236 = v32 + 1074;
        const int16_t* v237 = (const int16_t*) v236;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v238 = *(const int16_t *)(v237);
        const uint8_t* v239 = v32 + 1082;
        const int16_t* v240 = (const int16_t*) v239;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v241 = *(const int16_t *)(v240);
        int32_t v242 = v238 + v241;
        int32_t* v243 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v244 = __riscv_vle32_v_i32m2(v243, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v245 = __riscv_vwmacc_vx_i32m2(v244, v242, v224, 16);
        int32_t* v246 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v246, v245, 16);
        const uint8_t* v247 = v32 + 1076;
        const int16_t* v248 = (const int16_t*) v247;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v249 = *(const int16_t *)(v248);
        const uint8_t* v250 = v32 + 1084;
        const int16_t* v251 = (const int16_t*) v250;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v252 = *(const int16_t *)(v251);
        int32_t v253 = v249 + v252;
        int32_t* v254 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v255 = __riscv_vle32_v_i32m2(v254, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v256 = __riscv_vwmacc_vx_i32m2(v255, v253, v224, 16);
        int32_t* v257 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v257, v256, 16);
        const uint8_t* v258 = v32 + 1078;
        const int16_t* v259 = (const int16_t*) v258;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v260 = *(const int16_t *)(v259);
        const uint8_t* v261 = v32 + 1086;
        const int16_t* v262 = (const int16_t*) v261;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v263 = *(const int16_t *)(v262);
        int32_t v264 = v260 + v263;
        int32_t* v265 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v266 = __riscv_vle32_v_i32m2(v265, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v267 = __riscv_vwmacc_vx_i32m2(v266, v264, v224, 16);
        int32_t* v268 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v268, v267, 16);
        int16_t* v269 = &v45[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v270 = __riscv_vle16_v_i16m1(v269, 16);
        const uint8_t* v271 = v32 + 1088;
        const int16_t* v272 = (const int16_t*) v271;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v273 = *(const int16_t *)(v272);
        const uint8_t* v274 = v32 + 1096;
        const int16_t* v275 = (const int16_t*) v274;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v276 = *(const int16_t *)(v275);
        int32_t v277 = v273 + v276;
        int32_t* v278 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v279 = __riscv_vle32_v_i32m2(v278, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v280 = __riscv_vwmacc_vx_i32m2(v279, v277, v270, 16);
        int32_t* v281 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v281, v280, 16);
        const uint8_t* v282 = v32 + 1090;
        const int16_t* v283 = (const int16_t*) v282;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v284 = *(const int16_t *)(v283);
        const uint8_t* v285 = v32 + 1098;
        const int16_t* v286 = (const int16_t*) v285;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v287 = *(const int16_t *)(v286);
        int32_t v288 = v284 + v287;
        int32_t* v289 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v290 = __riscv_vle32_v_i32m2(v289, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v291 = __riscv_vwmacc_vx_i32m2(v290, v288, v270, 16);
        int32_t* v292 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v292, v291, 16);
        const uint8_t* v293 = v32 + 1092;
        const int16_t* v294 = (const int16_t*) v293;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v295 = *(const int16_t *)(v294);
        const uint8_t* v296 = v32 + 1100;
        const int16_t* v297 = (const int16_t*) v296;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v298 = *(const int16_t *)(v297);
        int32_t v299 = v295 + v298;
        int32_t* v300 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v301 = __riscv_vle32_v_i32m2(v300, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v302 = __riscv_vwmacc_vx_i32m2(v301, v299, v270, 16);
        int32_t* v303 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v303, v302, 16);
        const uint8_t* v304 = v32 + 1094;
        const int16_t* v305 = (const int16_t*) v304;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v306 = *(const int16_t *)(v305);
        const uint8_t* v307 = v32 + 1102;
        const int16_t* v308 = (const int16_t*) v307;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v309 = *(const int16_t *)(v308);
        int32_t v310 = v306 + v309;
        int32_t* v311 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v312 = __riscv_vle32_v_i32m2(v311, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v313 = __riscv_vwmacc_vx_i32m2(v312, v310, v270, 16);
        int32_t* v314 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v314, v313, 16);
        vint16m1_t v315;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v316 = __riscv_vmv_v_x_i16m1(0, 16);
        v315 = v316;
        vint16m1_t v317;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v318 = __riscv_vmv_v_x_i16m1(0, 16);
        v317 = v318;
        vint16m1_t v319;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v320 = __riscv_vmv_v_x_i16m1(0, 16);
        v319 = v320;
        vint16m1_t v321;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v322 = __riscv_vmv_v_x_i16m1(0, 16);
        v321 = v322;
        vint16m1_t v323;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v324 = __riscv_vmv_v_x_i16m1(0, 16);
        v323 = v324;
        vint16m1_t v325;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v326 = __riscv_vmv_v_x_i16m1(0, 16);
        v325 = v326;
        vint16m1_t v327;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v328 = __riscv_vmv_v_x_i16m1(0, 16);
        v327 = v328;
        vint16m1_t v329;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v330 = __riscv_vmv_v_x_i16m1(0, 16);
        v329 = v330;
        for (size_t v331 = 0; v331 < 16; v331 += 1) {
          size_t v332 = v331 * 16;
          size_t v333 = v331 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v334 = 768 + v332;
          const uint8_t* v335 = v30 + v334;
          const uint8_t* v336 = (const uint8_t*) v335;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v337 = __riscv_vle8_v_u8mf2(v336, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v338 = __riscv_vand_vx_u8mf2(v337, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v339 = __riscv_vsrl_vx_u8mf2(v337, 4, 16);
          size_t v340 = 256 + v332;
          const uint8_t* v341 = v30 + v340;
          const uint8_t* v342 = (const uint8_t*) v341;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v343 = __riscv_vle8_v_u8mf2(v342, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v344 = __riscv_vand_vx_u8mf2(v343, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v345 = __riscv_vsll_vx_u8mf2(v344, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v346 = __riscv_vsrl_vx_u8mf2(v343, 1, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v347 = __riscv_vand_vx_u8mf2(v346, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v348 = __riscv_vsll_vx_u8mf2(v347, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v349 = __riscv_vor_vv_u8mf2(v338, v345, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v350 = __riscv_vreinterpret_v_u8mf2_i8mf2(v349);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v351 = __riscv_vor_vv_u8mf2(v339, v348, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v352 = __riscv_vreinterpret_v_u8mf2_i8mf2(v351);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v353 = 16 + v333;
          size_t v354 = 144 + v333;
          const uint8_t* v355 = v32 + v353;
          const int8_t* v356 = (const int8_t*) v355;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v357 = *(const int8_t *)(v356);
          const uint8_t* v358 = v32 + v354;
          const int8_t* v359 = (const int8_t*) v358;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v360 = *(const int8_t *)(v359);
          vint16m1_t v361 = v315;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v362 = __riscv_vwmacc_vx_i16m1(v361, v357, v350, 16);
          v315 = v362;
          vint16m1_t v363 = v317;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v364 = __riscv_vwmacc_vx_i16m1(v363, v360, v352, 16);
          v317 = v364;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v365 = 17 + v333;
          size_t v366 = 145 + v333;
          const uint8_t* v367 = v32 + v365;
          const int8_t* v368 = (const int8_t*) v367;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v369 = *(const int8_t *)(v368);
          const uint8_t* v370 = v32 + v366;
          const int8_t* v371 = (const int8_t*) v370;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v372 = *(const int8_t *)(v371);
          vint16m1_t v373 = v319;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v374 = __riscv_vwmacc_vx_i16m1(v373, v369, v350, 16);
          v319 = v374;
          vint16m1_t v375 = v321;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v376 = __riscv_vwmacc_vx_i16m1(v375, v372, v352, 16);
          v321 = v376;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v377 = 18 + v333;
          size_t v378 = 146 + v333;
          const uint8_t* v379 = v32 + v377;
          const int8_t* v380 = (const int8_t*) v379;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v381 = *(const int8_t *)(v380);
          const uint8_t* v382 = v32 + v378;
          const int8_t* v383 = (const int8_t*) v382;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v384 = *(const int8_t *)(v383);
          vint16m1_t v385 = v323;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v386 = __riscv_vwmacc_vx_i16m1(v385, v381, v350, 16);
          v323 = v386;
          vint16m1_t v387 = v325;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v388 = __riscv_vwmacc_vx_i16m1(v387, v384, v352, 16);
          v325 = v388;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v389 = 19 + v333;
          size_t v390 = 147 + v333;
          const uint8_t* v391 = v32 + v389;
          const int8_t* v392 = (const int8_t*) v391;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v393 = *(const int8_t *)(v392);
          const uint8_t* v394 = v32 + v390;
          const int8_t* v395 = (const int8_t*) v394;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v396 = *(const int8_t *)(v395);
          vint16m1_t v397 = v327;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v398 = __riscv_vwmacc_vx_i16m1(v397, v393, v350, 16);
          v327 = v398;
          vint16m1_t v399 = v329;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v400 = __riscv_vwmacc_vx_i16m1(v399, v396, v352, 16);
          v329 = v400;
        }
        vint16m1_t v401 = v315;
        vint16m1_t v402 = v317;
        vint16m1_t v403 = v319;
        vint16m1_t v404 = v321;
        vint16m1_t v405 = v323;
        vint16m1_t v406 = v325;
        vint16m1_t v407 = v327;
        vint16m1_t v408 = v329;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v409 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v410 = __riscv_vle16_v_i16m1(v409, 16);
        int16_t* v411 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v412 = __riscv_vle16_v_i16m1(v411, 16);
        vint32m2_t v413 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v414 = __riscv_vwmacc_vv_i32m2(v413, v410, v401, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v415 = __riscv_vwmacc_vv_i32m2(v414, v412, v402, 16);
        v47 = v415;
        vint32m2_t v416 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v417 = __riscv_vwmacc_vv_i32m2(v416, v410, v403, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v418 = __riscv_vwmacc_vv_i32m2(v417, v412, v404, 16);
        v49 = v418;
        vint32m2_t v419 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v420 = __riscv_vwmacc_vv_i32m2(v419, v410, v405, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v421 = __riscv_vwmacc_vv_i32m2(v420, v412, v406, 16);
        v51 = v421;
        vint32m2_t v422 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v423 = __riscv_vwmacc_vv_i32m2(v422, v410, v407, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v424 = __riscv_vwmacc_vv_i32m2(v423, v412, v408, 16);
        v53 = v424;
        vint16m1_t v425;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v426 = __riscv_vmv_v_x_i16m1(0, 16);
        v425 = v426;
        vint16m1_t v427;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v428 = __riscv_vmv_v_x_i16m1(0, 16);
        v427 = v428;
        vint16m1_t v429;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v430 = __riscv_vmv_v_x_i16m1(0, 16);
        v429 = v430;
        vint16m1_t v431;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v432 = __riscv_vmv_v_x_i16m1(0, 16);
        v431 = v432;
        vint16m1_t v433;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v434 = __riscv_vmv_v_x_i16m1(0, 16);
        v433 = v434;
        vint16m1_t v435;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v436 = __riscv_vmv_v_x_i16m1(0, 16);
        v435 = v436;
        vint16m1_t v437;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v438 = __riscv_vmv_v_x_i16m1(0, 16);
        v437 = v438;
        vint16m1_t v439;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v440 = __riscv_vmv_v_x_i16m1(0, 16);
        v439 = v440;
        for (size_t v441 = 0; v441 < 16; v441 += 1) {
          size_t v442 = v441 * 16;
          size_t v443 = v441 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v444 = 1024 + v442;
          const uint8_t* v445 = v30 + v444;
          const uint8_t* v446 = (const uint8_t*) v445;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v447 = __riscv_vle8_v_u8mf2(v446, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v448 = __riscv_vand_vx_u8mf2(v447, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v449 = __riscv_vsrl_vx_u8mf2(v447, 4, 16);
          size_t v450 = 512 + v442;
          const uint8_t* v451 = v30 + v450;
          const uint8_t* v452 = (const uint8_t*) v451;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v453 = __riscv_vle8_v_u8mf2(v452, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v454 = __riscv_vand_vx_u8mf2(v453, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v455 = __riscv_vsll_vx_u8mf2(v454, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v456 = __riscv_vsrl_vx_u8mf2(v453, 1, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v457 = __riscv_vand_vx_u8mf2(v456, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v458 = __riscv_vsll_vx_u8mf2(v457, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v459 = __riscv_vor_vv_u8mf2(v448, v455, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v460 = __riscv_vreinterpret_v_u8mf2_i8mf2(v459);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v461 = __riscv_vor_vv_u8mf2(v449, v458, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v462 = __riscv_vreinterpret_v_u8mf2_i8mf2(v461);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v463 = 80 + v443;
          size_t v464 = 208 + v443;
          const uint8_t* v465 = v32 + v463;
          const int8_t* v466 = (const int8_t*) v465;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v467 = *(const int8_t *)(v466);
          const uint8_t* v468 = v32 + v464;
          const int8_t* v469 = (const int8_t*) v468;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v470 = *(const int8_t *)(v469);
          vint16m1_t v471 = v425;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v472 = __riscv_vwmacc_vx_i16m1(v471, v467, v460, 16);
          v425 = v472;
          vint16m1_t v473 = v427;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v474 = __riscv_vwmacc_vx_i16m1(v473, v470, v462, 16);
          v427 = v474;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v475 = 81 + v443;
          size_t v476 = 209 + v443;
          const uint8_t* v477 = v32 + v475;
          const int8_t* v478 = (const int8_t*) v477;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v479 = *(const int8_t *)(v478);
          const uint8_t* v480 = v32 + v476;
          const int8_t* v481 = (const int8_t*) v480;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v482 = *(const int8_t *)(v481);
          vint16m1_t v483 = v429;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v484 = __riscv_vwmacc_vx_i16m1(v483, v479, v460, 16);
          v429 = v484;
          vint16m1_t v485 = v431;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v486 = __riscv_vwmacc_vx_i16m1(v485, v482, v462, 16);
          v431 = v486;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v487 = 82 + v443;
          size_t v488 = 210 + v443;
          const uint8_t* v489 = v32 + v487;
          const int8_t* v490 = (const int8_t*) v489;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v491 = *(const int8_t *)(v490);
          const uint8_t* v492 = v32 + v488;
          const int8_t* v493 = (const int8_t*) v492;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v494 = *(const int8_t *)(v493);
          vint16m1_t v495 = v433;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v496 = __riscv_vwmacc_vx_i16m1(v495, v491, v460, 16);
          v433 = v496;
          vint16m1_t v497 = v435;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v498 = __riscv_vwmacc_vx_i16m1(v497, v494, v462, 16);
          v435 = v498;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v499 = 83 + v443;
          size_t v500 = 211 + v443;
          const uint8_t* v501 = v32 + v499;
          const int8_t* v502 = (const int8_t*) v501;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v503 = *(const int8_t *)(v502);
          const uint8_t* v504 = v32 + v500;
          const int8_t* v505 = (const int8_t*) v504;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v506 = *(const int8_t *)(v505);
          vint16m1_t v507 = v437;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v508 = __riscv_vwmacc_vx_i16m1(v507, v503, v460, 16);
          v437 = v508;
          vint16m1_t v509 = v439;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v510 = __riscv_vwmacc_vx_i16m1(v509, v506, v462, 16);
          v439 = v510;
        }
        vint16m1_t v511 = v425;
        vint16m1_t v512 = v427;
        vint16m1_t v513 = v429;
        vint16m1_t v514 = v431;
        vint16m1_t v515 = v433;
        vint16m1_t v516 = v435;
        vint16m1_t v517 = v437;
        vint16m1_t v518 = v439;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v519 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v520 = __riscv_vle16_v_i16m1(v519, 16);
        int16_t* v521 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v522 = __riscv_vle16_v_i16m1(v521, 16);
        vint32m2_t v523 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v524 = __riscv_vwmacc_vv_i32m2(v523, v520, v511, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v525 = __riscv_vwmacc_vv_i32m2(v524, v522, v512, 16);
        v47 = v525;
        vint32m2_t v526 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v527 = __riscv_vwmacc_vv_i32m2(v526, v520, v513, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v528 = __riscv_vwmacc_vv_i32m2(v527, v522, v514, 16);
        v49 = v528;
        vint32m2_t v529 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v530 = __riscv_vwmacc_vv_i32m2(v529, v520, v515, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v531 = __riscv_vwmacc_vv_i32m2(v530, v522, v516, 16);
        v51 = v531;
        vint32m2_t v532 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v533 = __riscv_vwmacc_vv_i32m2(v532, v520, v517, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v534 = __riscv_vwmacc_vv_i32m2(v533, v522, v518, 16);
        v53 = v534;
        vint16m1_t v535;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v536 = __riscv_vmv_v_x_i16m1(0, 16);
        v535 = v536;
        vint16m1_t v537;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v538 = __riscv_vmv_v_x_i16m1(0, 16);
        v537 = v538;
        vint16m1_t v539;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v540 = __riscv_vmv_v_x_i16m1(0, 16);
        v539 = v540;
        vint16m1_t v541;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v542 = __riscv_vmv_v_x_i16m1(0, 16);
        v541 = v542;
        vint16m1_t v543;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v544 = __riscv_vmv_v_x_i16m1(0, 16);
        v543 = v544;
        vint16m1_t v545;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v546 = __riscv_vmv_v_x_i16m1(0, 16);
        v545 = v546;
        vint16m1_t v547;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v548 = __riscv_vmv_v_x_i16m1(0, 16);
        v547 = v548;
        vint16m1_t v549;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v550 = __riscv_vmv_v_x_i16m1(0, 16);
        v549 = v550;
        for (size_t v551 = 0; v551 < 16; v551 += 1) {
          size_t v552 = v551 * 16;
          size_t v553 = v551 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v554 = 1280 + v552;
          const uint8_t* v555 = v30 + v554;
          const uint8_t* v556 = (const uint8_t*) v555;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v557 = __riscv_vle8_v_u8mf2(v556, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v558 = __riscv_vand_vx_u8mf2(v557, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v559 = __riscv_vsrl_vx_u8mf2(v557, 4, 16);
          size_t v560 = 256 + v552;
          const uint8_t* v561 = v30 + v560;
          const uint8_t* v562 = (const uint8_t*) v561;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v563 = __riscv_vle8_v_u8mf2(v562, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v564 = __riscv_vsrl_vx_u8mf2(v563, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v565 = __riscv_vand_vx_u8mf2(v564, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v566 = __riscv_vsll_vx_u8mf2(v565, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v567 = __riscv_vsrl_vx_u8mf2(v563, 3, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v568 = __riscv_vand_vx_u8mf2(v567, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v569 = __riscv_vsll_vx_u8mf2(v568, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v570 = __riscv_vor_vv_u8mf2(v558, v566, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v571 = __riscv_vreinterpret_v_u8mf2_i8mf2(v570);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v572 = __riscv_vor_vv_u8mf2(v559, v569, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v573 = __riscv_vreinterpret_v_u8mf2_i8mf2(v572);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v574 = 272 + v553;
          size_t v575 = 400 + v553;
          const uint8_t* v576 = v32 + v574;
          const int8_t* v577 = (const int8_t*) v576;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v578 = *(const int8_t *)(v577);
          const uint8_t* v579 = v32 + v575;
          const int8_t* v580 = (const int8_t*) v579;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v581 = *(const int8_t *)(v580);
          vint16m1_t v582 = v535;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v583 = __riscv_vwmacc_vx_i16m1(v582, v578, v571, 16);
          v535 = v583;
          vint16m1_t v584 = v537;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v585 = __riscv_vwmacc_vx_i16m1(v584, v581, v573, 16);
          v537 = v585;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v586 = 273 + v553;
          size_t v587 = 401 + v553;
          const uint8_t* v588 = v32 + v586;
          const int8_t* v589 = (const int8_t*) v588;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v590 = *(const int8_t *)(v589);
          const uint8_t* v591 = v32 + v587;
          const int8_t* v592 = (const int8_t*) v591;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v593 = *(const int8_t *)(v592);
          vint16m1_t v594 = v539;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v595 = __riscv_vwmacc_vx_i16m1(v594, v590, v571, 16);
          v539 = v595;
          vint16m1_t v596 = v541;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v597 = __riscv_vwmacc_vx_i16m1(v596, v593, v573, 16);
          v541 = v597;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v598 = 274 + v553;
          size_t v599 = 402 + v553;
          const uint8_t* v600 = v32 + v598;
          const int8_t* v601 = (const int8_t*) v600;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v602 = *(const int8_t *)(v601);
          const uint8_t* v603 = v32 + v599;
          const int8_t* v604 = (const int8_t*) v603;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v605 = *(const int8_t *)(v604);
          vint16m1_t v606 = v543;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v607 = __riscv_vwmacc_vx_i16m1(v606, v602, v571, 16);
          v543 = v607;
          vint16m1_t v608 = v545;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v609 = __riscv_vwmacc_vx_i16m1(v608, v605, v573, 16);
          v545 = v609;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v610 = 275 + v553;
          size_t v611 = 403 + v553;
          const uint8_t* v612 = v32 + v610;
          const int8_t* v613 = (const int8_t*) v612;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v614 = *(const int8_t *)(v613);
          const uint8_t* v615 = v32 + v611;
          const int8_t* v616 = (const int8_t*) v615;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v617 = *(const int8_t *)(v616);
          vint16m1_t v618 = v547;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v619 = __riscv_vwmacc_vx_i16m1(v618, v614, v571, 16);
          v547 = v619;
          vint16m1_t v620 = v549;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v621 = __riscv_vwmacc_vx_i16m1(v620, v617, v573, 16);
          v549 = v621;
        }
        vint16m1_t v622 = v535;
        vint16m1_t v623 = v537;
        vint16m1_t v624 = v539;
        vint16m1_t v625 = v541;
        vint16m1_t v626 = v543;
        vint16m1_t v627 = v545;
        vint16m1_t v628 = v547;
        vint16m1_t v629 = v549;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v630 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v631 = __riscv_vle16_v_i16m1(v630, 16);
        int16_t* v632 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v633 = __riscv_vle16_v_i16m1(v632, 16);
        vint32m2_t v634 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v635 = __riscv_vwmacc_vv_i32m2(v634, v631, v622, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v636 = __riscv_vwmacc_vv_i32m2(v635, v633, v623, 16);
        v47 = v636;
        vint32m2_t v637 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v638 = __riscv_vwmacc_vv_i32m2(v637, v631, v624, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v639 = __riscv_vwmacc_vv_i32m2(v638, v633, v625, 16);
        v49 = v639;
        vint32m2_t v640 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v641 = __riscv_vwmacc_vv_i32m2(v640, v631, v626, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v642 = __riscv_vwmacc_vv_i32m2(v641, v633, v627, 16);
        v51 = v642;
        vint32m2_t v643 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v644 = __riscv_vwmacc_vv_i32m2(v643, v631, v628, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v645 = __riscv_vwmacc_vv_i32m2(v644, v633, v629, 16);
        v53 = v645;
        vint16m1_t v646;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v647 = __riscv_vmv_v_x_i16m1(0, 16);
        v646 = v647;
        vint16m1_t v648;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v649 = __riscv_vmv_v_x_i16m1(0, 16);
        v648 = v649;
        vint16m1_t v650;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v651 = __riscv_vmv_v_x_i16m1(0, 16);
        v650 = v651;
        vint16m1_t v652;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v653 = __riscv_vmv_v_x_i16m1(0, 16);
        v652 = v653;
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
        for (size_t v662 = 0; v662 < 16; v662 += 1) {
          size_t v663 = v662 * 16;
          size_t v664 = v662 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v665 = 1536 + v663;
          const uint8_t* v666 = v30 + v665;
          const uint8_t* v667 = (const uint8_t*) v666;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v668 = __riscv_vle8_v_u8mf2(v667, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v669 = __riscv_vand_vx_u8mf2(v668, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v670 = __riscv_vsrl_vx_u8mf2(v668, 4, 16);
          size_t v671 = 512 + v663;
          const uint8_t* v672 = v30 + v671;
          const uint8_t* v673 = (const uint8_t*) v672;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v674 = __riscv_vle8_v_u8mf2(v673, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v675 = __riscv_vsrl_vx_u8mf2(v674, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v676 = __riscv_vand_vx_u8mf2(v675, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v677 = __riscv_vsll_vx_u8mf2(v676, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v678 = __riscv_vsrl_vx_u8mf2(v674, 3, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v679 = __riscv_vand_vx_u8mf2(v678, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v680 = __riscv_vsll_vx_u8mf2(v679, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v681 = __riscv_vor_vv_u8mf2(v669, v677, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v682 = __riscv_vreinterpret_v_u8mf2_i8mf2(v681);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v683 = __riscv_vor_vv_u8mf2(v670, v680, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v684 = __riscv_vreinterpret_v_u8mf2_i8mf2(v683);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v685 = 336 + v664;
          size_t v686 = 464 + v664;
          const uint8_t* v687 = v32 + v685;
          const int8_t* v688 = (const int8_t*) v687;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v689 = *(const int8_t *)(v688);
          const uint8_t* v690 = v32 + v686;
          const int8_t* v691 = (const int8_t*) v690;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v692 = *(const int8_t *)(v691);
          vint16m1_t v693 = v646;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v694 = __riscv_vwmacc_vx_i16m1(v693, v689, v682, 16);
          v646 = v694;
          vint16m1_t v695 = v648;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v696 = __riscv_vwmacc_vx_i16m1(v695, v692, v684, 16);
          v648 = v696;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v697 = 337 + v664;
          size_t v698 = 465 + v664;
          const uint8_t* v699 = v32 + v697;
          const int8_t* v700 = (const int8_t*) v699;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v701 = *(const int8_t *)(v700);
          const uint8_t* v702 = v32 + v698;
          const int8_t* v703 = (const int8_t*) v702;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v704 = *(const int8_t *)(v703);
          vint16m1_t v705 = v650;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v706 = __riscv_vwmacc_vx_i16m1(v705, v701, v682, 16);
          v650 = v706;
          vint16m1_t v707 = v652;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v708 = __riscv_vwmacc_vx_i16m1(v707, v704, v684, 16);
          v652 = v708;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v709 = 338 + v664;
          size_t v710 = 466 + v664;
          const uint8_t* v711 = v32 + v709;
          const int8_t* v712 = (const int8_t*) v711;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v713 = *(const int8_t *)(v712);
          const uint8_t* v714 = v32 + v710;
          const int8_t* v715 = (const int8_t*) v714;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v716 = *(const int8_t *)(v715);
          vint16m1_t v717 = v654;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v718 = __riscv_vwmacc_vx_i16m1(v717, v713, v682, 16);
          v654 = v718;
          vint16m1_t v719 = v656;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v720 = __riscv_vwmacc_vx_i16m1(v719, v716, v684, 16);
          v656 = v720;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v721 = 339 + v664;
          size_t v722 = 467 + v664;
          const uint8_t* v723 = v32 + v721;
          const int8_t* v724 = (const int8_t*) v723;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v725 = *(const int8_t *)(v724);
          const uint8_t* v726 = v32 + v722;
          const int8_t* v727 = (const int8_t*) v726;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v728 = *(const int8_t *)(v727);
          vint16m1_t v729 = v658;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v730 = __riscv_vwmacc_vx_i16m1(v729, v725, v682, 16);
          v658 = v730;
          vint16m1_t v731 = v660;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v732 = __riscv_vwmacc_vx_i16m1(v731, v728, v684, 16);
          v660 = v732;
        }
        vint16m1_t v733 = v646;
        vint16m1_t v734 = v648;
        vint16m1_t v735 = v650;
        vint16m1_t v736 = v652;
        vint16m1_t v737 = v654;
        vint16m1_t v738 = v656;
        vint16m1_t v739 = v658;
        vint16m1_t v740 = v660;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v741 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v742 = __riscv_vle16_v_i16m1(v741, 16);
        int16_t* v743 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v744 = __riscv_vle16_v_i16m1(v743, 16);
        vint32m2_t v745 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v746 = __riscv_vwmacc_vv_i32m2(v745, v742, v733, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v747 = __riscv_vwmacc_vv_i32m2(v746, v744, v734, 16);
        v47 = v747;
        vint32m2_t v748 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v749 = __riscv_vwmacc_vv_i32m2(v748, v742, v735, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v750 = __riscv_vwmacc_vv_i32m2(v749, v744, v736, 16);
        v49 = v750;
        vint32m2_t v751 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v752 = __riscv_vwmacc_vv_i32m2(v751, v742, v737, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v753 = __riscv_vwmacc_vv_i32m2(v752, v744, v738, 16);
        v51 = v753;
        vint32m2_t v754 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v755 = __riscv_vwmacc_vv_i32m2(v754, v742, v739, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v756 = __riscv_vwmacc_vv_i32m2(v755, v744, v740, 16);
        v53 = v756;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
        const uint8_t* v757 = v30 + 128;
        const uint8_t* v758 = (const uint8_t*) v757;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v759 = __riscv_vle8_v_u8mf2(v758, 16);
        const uint8_t* v760 = v30 + 192;
        const uint8_t* v761 = (const uint8_t*) v760;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v762 = __riscv_vle8_v_u8mf2(v761, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v763 = __riscv_vand_vx_u8mf2(v759, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v764 = __riscv_vsrl_vx_u8mf2(v759, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v765 = __riscv_vand_vx_u8mf2(v762, 0x30, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v766 = __riscv_vand_vx_u8mf2(v762, 0xC0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v767 = __riscv_vsrl_vx_u8mf2(v766, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v768 = __riscv_vor_vv_u8mf2(v765, v763, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v769 = __riscv_vor_vv_u8mf2(v767, v764, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v770 = __riscv_vzext_vf2_u16m1(v768, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v771 = __riscv_vreinterpret_v_u16m1_i16m1(v770);
        int16_t* v772 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v772, v771, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v773 = __riscv_vzext_vf2_u16m1(v769, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v774 = __riscv_vreinterpret_v_u16m1_i16m1(v773);
        int16_t* v775 = &v45[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v775, v774, 16);
        const uint8_t* v776 = v30 + 144;
        const uint8_t* v777 = (const uint8_t*) v776;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v778 = __riscv_vle8_v_u8mf2(v777, 16);
        const uint8_t* v779 = v30 + 208;
        const uint8_t* v780 = (const uint8_t*) v779;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v781 = __riscv_vle8_v_u8mf2(v780, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v782 = __riscv_vand_vx_u8mf2(v778, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v783 = __riscv_vsrl_vx_u8mf2(v778, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v784 = __riscv_vand_vx_u8mf2(v781, 0x30, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v785 = __riscv_vand_vx_u8mf2(v781, 0xC0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v786 = __riscv_vsrl_vx_u8mf2(v785, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v787 = __riscv_vor_vv_u8mf2(v784, v782, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v788 = __riscv_vor_vv_u8mf2(v786, v783, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v789 = __riscv_vzext_vf2_u16m1(v787, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v790 = __riscv_vreinterpret_v_u16m1_i16m1(v789);
        int16_t* v791 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v791, v790, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v792 = __riscv_vzext_vf2_u16m1(v788, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v793 = __riscv_vreinterpret_v_u16m1_i16m1(v792);
        int16_t* v794 = &v45[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v794, v793, 16);
        const uint8_t* v795 = v30 + 160;
        const uint8_t* v796 = (const uint8_t*) v795;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v797 = __riscv_vle8_v_u8mf2(v796, 16);
        const uint8_t* v798 = v30 + 224;
        const uint8_t* v799 = (const uint8_t*) v798;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v800 = __riscv_vle8_v_u8mf2(v799, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v801 = __riscv_vand_vx_u8mf2(v797, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v802 = __riscv_vsrl_vx_u8mf2(v797, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v803 = __riscv_vand_vx_u8mf2(v800, 0x30, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v804 = __riscv_vand_vx_u8mf2(v800, 0xC0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v805 = __riscv_vsrl_vx_u8mf2(v804, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v806 = __riscv_vor_vv_u8mf2(v803, v801, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v807 = __riscv_vor_vv_u8mf2(v805, v802, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v808 = __riscv_vzext_vf2_u16m1(v806, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v809 = __riscv_vreinterpret_v_u16m1_i16m1(v808);
        int16_t* v810 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v810, v809, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v811 = __riscv_vzext_vf2_u16m1(v807, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v812 = __riscv_vreinterpret_v_u16m1_i16m1(v811);
        int16_t* v813 = &v45[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v813, v812, 16);
        const uint8_t* v814 = v30 + 176;
        const uint8_t* v815 = (const uint8_t*) v814;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v816 = __riscv_vle8_v_u8mf2(v815, 16);
        const uint8_t* v817 = v30 + 240;
        const uint8_t* v818 = (const uint8_t*) v817;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v819 = __riscv_vle8_v_u8mf2(v818, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v820 = __riscv_vand_vx_u8mf2(v816, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v821 = __riscv_vsrl_vx_u8mf2(v816, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v822 = __riscv_vand_vx_u8mf2(v819, 0x30, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v823 = __riscv_vand_vx_u8mf2(v819, 0xC0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v824 = __riscv_vsrl_vx_u8mf2(v823, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v825 = __riscv_vor_vv_u8mf2(v822, v820, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v826 = __riscv_vor_vv_u8mf2(v824, v821, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v827 = __riscv_vzext_vf2_u16m1(v825, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v828 = __riscv_vreinterpret_v_u16m1_i16m1(v827);
        int16_t* v829 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v829, v828, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v830 = __riscv_vzext_vf2_u16m1(v826, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v831 = __riscv_vreinterpret_v_u16m1_i16m1(v830);
        int16_t* v832 = &v45[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v832, v831, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold
        int16_t* v833 = &v45[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v834 = __riscv_vle16_v_i16m1(v833, 16);
        const uint8_t* v835 = v32 + 1104;
        const int16_t* v836 = (const int16_t*) v835;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v837 = *(const int16_t *)(v836);
        const uint8_t* v838 = v32 + 1112;
        const int16_t* v839 = (const int16_t*) v838;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v840 = *(const int16_t *)(v839);
        int32_t v841 = v837 + v840;
        int32_t* v842 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v843 = __riscv_vle32_v_i32m2(v842, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v844 = __riscv_vwmacc_vx_i32m2(v843, v841, v834, 16);
        int32_t* v845 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v845, v844, 16);
        const uint8_t* v846 = v32 + 1106;
        const int16_t* v847 = (const int16_t*) v846;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v848 = *(const int16_t *)(v847);
        const uint8_t* v849 = v32 + 1114;
        const int16_t* v850 = (const int16_t*) v849;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v851 = *(const int16_t *)(v850);
        int32_t v852 = v848 + v851;
        int32_t* v853 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v854 = __riscv_vle32_v_i32m2(v853, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v855 = __riscv_vwmacc_vx_i32m2(v854, v852, v834, 16);
        int32_t* v856 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v856, v855, 16);
        const uint8_t* v857 = v32 + 1108;
        const int16_t* v858 = (const int16_t*) v857;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v859 = *(const int16_t *)(v858);
        const uint8_t* v860 = v32 + 1116;
        const int16_t* v861 = (const int16_t*) v860;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v862 = *(const int16_t *)(v861);
        int32_t v863 = v859 + v862;
        int32_t* v864 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v865 = __riscv_vle32_v_i32m2(v864, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v866 = __riscv_vwmacc_vx_i32m2(v865, v863, v834, 16);
        int32_t* v867 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v867, v866, 16);
        const uint8_t* v868 = v32 + 1110;
        const int16_t* v869 = (const int16_t*) v868;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v870 = *(const int16_t *)(v869);
        const uint8_t* v871 = v32 + 1118;
        const int16_t* v872 = (const int16_t*) v871;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v873 = *(const int16_t *)(v872);
        int32_t v874 = v870 + v873;
        int32_t* v875 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v876 = __riscv_vle32_v_i32m2(v875, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v877 = __riscv_vwmacc_vx_i32m2(v876, v874, v834, 16);
        int32_t* v878 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v878, v877, 16);
        int16_t* v879 = &v45[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v880 = __riscv_vle16_v_i16m1(v879, 16);
        const uint8_t* v881 = v32 + 1120;
        const int16_t* v882 = (const int16_t*) v881;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v883 = *(const int16_t *)(v882);
        const uint8_t* v884 = v32 + 1128;
        const int16_t* v885 = (const int16_t*) v884;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v886 = *(const int16_t *)(v885);
        int32_t v887 = v883 + v886;
        int32_t* v888 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v889 = __riscv_vle32_v_i32m2(v888, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v890 = __riscv_vwmacc_vx_i32m2(v889, v887, v880, 16);
        int32_t* v891 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v891, v890, 16);
        const uint8_t* v892 = v32 + 1122;
        const int16_t* v893 = (const int16_t*) v892;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v894 = *(const int16_t *)(v893);
        const uint8_t* v895 = v32 + 1130;
        const int16_t* v896 = (const int16_t*) v895;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v897 = *(const int16_t *)(v896);
        int32_t v898 = v894 + v897;
        int32_t* v899 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v900 = __riscv_vle32_v_i32m2(v899, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v901 = __riscv_vwmacc_vx_i32m2(v900, v898, v880, 16);
        int32_t* v902 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v902, v901, 16);
        const uint8_t* v903 = v32 + 1124;
        const int16_t* v904 = (const int16_t*) v903;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v905 = *(const int16_t *)(v904);
        const uint8_t* v906 = v32 + 1132;
        const int16_t* v907 = (const int16_t*) v906;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v908 = *(const int16_t *)(v907);
        int32_t v909 = v905 + v908;
        int32_t* v910 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v911 = __riscv_vle32_v_i32m2(v910, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v912 = __riscv_vwmacc_vx_i32m2(v911, v909, v880, 16);
        int32_t* v913 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v913, v912, 16);
        const uint8_t* v914 = v32 + 1126;
        const int16_t* v915 = (const int16_t*) v914;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v916 = *(const int16_t *)(v915);
        const uint8_t* v917 = v32 + 1134;
        const int16_t* v918 = (const int16_t*) v917;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v919 = *(const int16_t *)(v918);
        int32_t v920 = v916 + v919;
        int32_t* v921 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v922 = __riscv_vle32_v_i32m2(v921, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v923 = __riscv_vwmacc_vx_i32m2(v922, v920, v880, 16);
        int32_t* v924 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v924, v923, 16);
        int16_t* v925 = &v45[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v926 = __riscv_vle16_v_i16m1(v925, 16);
        const uint8_t* v927 = v32 + 1136;
        const int16_t* v928 = (const int16_t*) v927;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v929 = *(const int16_t *)(v928);
        const uint8_t* v930 = v32 + 1144;
        const int16_t* v931 = (const int16_t*) v930;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v932 = *(const int16_t *)(v931);
        int32_t v933 = v929 + v932;
        int32_t* v934 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v935 = __riscv_vle32_v_i32m2(v934, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v936 = __riscv_vwmacc_vx_i32m2(v935, v933, v926, 16);
        int32_t* v937 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v937, v936, 16);
        const uint8_t* v938 = v32 + 1138;
        const int16_t* v939 = (const int16_t*) v938;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v940 = *(const int16_t *)(v939);
        const uint8_t* v941 = v32 + 1146;
        const int16_t* v942 = (const int16_t*) v941;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v943 = *(const int16_t *)(v942);
        int32_t v944 = v940 + v943;
        int32_t* v945 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v946 = __riscv_vle32_v_i32m2(v945, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v947 = __riscv_vwmacc_vx_i32m2(v946, v944, v926, 16);
        int32_t* v948 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v948, v947, 16);
        const uint8_t* v949 = v32 + 1140;
        const int16_t* v950 = (const int16_t*) v949;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v951 = *(const int16_t *)(v950);
        const uint8_t* v952 = v32 + 1148;
        const int16_t* v953 = (const int16_t*) v952;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v954 = *(const int16_t *)(v953);
        int32_t v955 = v951 + v954;
        int32_t* v956 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v957 = __riscv_vle32_v_i32m2(v956, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v958 = __riscv_vwmacc_vx_i32m2(v957, v955, v926, 16);
        int32_t* v959 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v959, v958, 16);
        const uint8_t* v960 = v32 + 1142;
        const int16_t* v961 = (const int16_t*) v960;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v962 = *(const int16_t *)(v961);
        const uint8_t* v963 = v32 + 1150;
        const int16_t* v964 = (const int16_t*) v963;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v965 = *(const int16_t *)(v964);
        int32_t v966 = v962 + v965;
        int32_t* v967 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v968 = __riscv_vle32_v_i32m2(v967, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v969 = __riscv_vwmacc_vx_i32m2(v968, v966, v926, 16);
        int32_t* v970 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v970, v969, 16);
        int16_t* v971 = &v45[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v972 = __riscv_vle16_v_i16m1(v971, 16);
        const uint8_t* v973 = v32 + 1152;
        const int16_t* v974 = (const int16_t*) v973;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v975 = *(const int16_t *)(v974);
        const uint8_t* v976 = v32 + 1160;
        const int16_t* v977 = (const int16_t*) v976;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v978 = *(const int16_t *)(v977);
        int32_t v979 = v975 + v978;
        int32_t* v980 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v981 = __riscv_vle32_v_i32m2(v980, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v982 = __riscv_vwmacc_vx_i32m2(v981, v979, v972, 16);
        int32_t* v983 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v983, v982, 16);
        const uint8_t* v984 = v32 + 1154;
        const int16_t* v985 = (const int16_t*) v984;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v986 = *(const int16_t *)(v985);
        const uint8_t* v987 = v32 + 1162;
        const int16_t* v988 = (const int16_t*) v987;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v989 = *(const int16_t *)(v988);
        int32_t v990 = v986 + v989;
        int32_t* v991 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v992 = __riscv_vle32_v_i32m2(v991, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v993 = __riscv_vwmacc_vx_i32m2(v992, v990, v972, 16);
        int32_t* v994 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v994, v993, 16);
        const uint8_t* v995 = v32 + 1156;
        const int16_t* v996 = (const int16_t*) v995;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v997 = *(const int16_t *)(v996);
        const uint8_t* v998 = v32 + 1164;
        const int16_t* v999 = (const int16_t*) v998;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1000 = *(const int16_t *)(v999);
        int32_t v1001 = v997 + v1000;
        int32_t* v1002 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1003 = __riscv_vle32_v_i32m2(v1002, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1004 = __riscv_vwmacc_vx_i32m2(v1003, v1001, v972, 16);
        int32_t* v1005 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1005, v1004, 16);
        const uint8_t* v1006 = v32 + 1158;
        const int16_t* v1007 = (const int16_t*) v1006;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1008 = *(const int16_t *)(v1007);
        const uint8_t* v1009 = v32 + 1166;
        const int16_t* v1010 = (const int16_t*) v1009;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1011 = *(const int16_t *)(v1010);
        int32_t v1012 = v1008 + v1011;
        int32_t* v1013 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1014 = __riscv_vle32_v_i32m2(v1013, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1015 = __riscv_vwmacc_vx_i32m2(v1014, v1012, v972, 16);
        int32_t* v1016 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1016, v1015, 16);
        vint16m1_t v1017;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1018 = __riscv_vmv_v_x_i16m1(0, 16);
        v1017 = v1018;
        vint16m1_t v1019;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1020 = __riscv_vmv_v_x_i16m1(0, 16);
        v1019 = v1020;
        vint16m1_t v1021;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1022 = __riscv_vmv_v_x_i16m1(0, 16);
        v1021 = v1022;
        vint16m1_t v1023;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1024 = __riscv_vmv_v_x_i16m1(0, 16);
        v1023 = v1024;
        vint16m1_t v1025;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1026 = __riscv_vmv_v_x_i16m1(0, 16);
        v1025 = v1026;
        vint16m1_t v1027;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1028 = __riscv_vmv_v_x_i16m1(0, 16);
        v1027 = v1028;
        vint16m1_t v1029;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1030 = __riscv_vmv_v_x_i16m1(0, 16);
        v1029 = v1030;
        vint16m1_t v1031;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1032 = __riscv_vmv_v_x_i16m1(0, 16);
        v1031 = v1032;
        for (size_t v1033 = 0; v1033 < 16; v1033 += 1) {
          size_t v1034 = v1033 * 16;
          size_t v1035 = v1033 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v1036 = 1792 + v1034;
          const uint8_t* v1037 = v30 + v1036;
          const uint8_t* v1038 = (const uint8_t*) v1037;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1039 = __riscv_vle8_v_u8mf2(v1038, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1040 = __riscv_vand_vx_u8mf2(v1039, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1041 = __riscv_vsrl_vx_u8mf2(v1039, 4, 16);
          size_t v1042 = 256 + v1034;
          const uint8_t* v1043 = v30 + v1042;
          const uint8_t* v1044 = (const uint8_t*) v1043;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1045 = __riscv_vle8_v_u8mf2(v1044, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1046 = __riscv_vsrl_vx_u8mf2(v1045, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1047 = __riscv_vand_vx_u8mf2(v1046, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1048 = __riscv_vsll_vx_u8mf2(v1047, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1049 = __riscv_vsrl_vx_u8mf2(v1045, 5, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1050 = __riscv_vand_vx_u8mf2(v1049, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1051 = __riscv_vsll_vx_u8mf2(v1050, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1052 = __riscv_vor_vv_u8mf2(v1040, v1048, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1053 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1052);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1054 = __riscv_vor_vv_u8mf2(v1041, v1051, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1055 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1054);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1056 = 528 + v1035;
          size_t v1057 = 656 + v1035;
          const uint8_t* v1058 = v32 + v1056;
          const int8_t* v1059 = (const int8_t*) v1058;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1060 = *(const int8_t *)(v1059);
          const uint8_t* v1061 = v32 + v1057;
          const int8_t* v1062 = (const int8_t*) v1061;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1063 = *(const int8_t *)(v1062);
          vint16m1_t v1064 = v1017;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1065 = __riscv_vwmacc_vx_i16m1(v1064, v1060, v1053, 16);
          v1017 = v1065;
          vint16m1_t v1066 = v1019;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1067 = __riscv_vwmacc_vx_i16m1(v1066, v1063, v1055, 16);
          v1019 = v1067;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1068 = 529 + v1035;
          size_t v1069 = 657 + v1035;
          const uint8_t* v1070 = v32 + v1068;
          const int8_t* v1071 = (const int8_t*) v1070;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1072 = *(const int8_t *)(v1071);
          const uint8_t* v1073 = v32 + v1069;
          const int8_t* v1074 = (const int8_t*) v1073;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1075 = *(const int8_t *)(v1074);
          vint16m1_t v1076 = v1021;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1077 = __riscv_vwmacc_vx_i16m1(v1076, v1072, v1053, 16);
          v1021 = v1077;
          vint16m1_t v1078 = v1023;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1079 = __riscv_vwmacc_vx_i16m1(v1078, v1075, v1055, 16);
          v1023 = v1079;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1080 = 530 + v1035;
          size_t v1081 = 658 + v1035;
          const uint8_t* v1082 = v32 + v1080;
          const int8_t* v1083 = (const int8_t*) v1082;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1084 = *(const int8_t *)(v1083);
          const uint8_t* v1085 = v32 + v1081;
          const int8_t* v1086 = (const int8_t*) v1085;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1087 = *(const int8_t *)(v1086);
          vint16m1_t v1088 = v1025;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1089 = __riscv_vwmacc_vx_i16m1(v1088, v1084, v1053, 16);
          v1025 = v1089;
          vint16m1_t v1090 = v1027;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1091 = __riscv_vwmacc_vx_i16m1(v1090, v1087, v1055, 16);
          v1027 = v1091;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1092 = 531 + v1035;
          size_t v1093 = 659 + v1035;
          const uint8_t* v1094 = v32 + v1092;
          const int8_t* v1095 = (const int8_t*) v1094;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1096 = *(const int8_t *)(v1095);
          const uint8_t* v1097 = v32 + v1093;
          const int8_t* v1098 = (const int8_t*) v1097;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1099 = *(const int8_t *)(v1098);
          vint16m1_t v1100 = v1029;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1101 = __riscv_vwmacc_vx_i16m1(v1100, v1096, v1053, 16);
          v1029 = v1101;
          vint16m1_t v1102 = v1031;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1103 = __riscv_vwmacc_vx_i16m1(v1102, v1099, v1055, 16);
          v1031 = v1103;
        }
        vint16m1_t v1104 = v1017;
        vint16m1_t v1105 = v1019;
        vint16m1_t v1106 = v1021;
        vint16m1_t v1107 = v1023;
        vint16m1_t v1108 = v1025;
        vint16m1_t v1109 = v1027;
        vint16m1_t v1110 = v1029;
        vint16m1_t v1111 = v1031;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v1112 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1113 = __riscv_vle16_v_i16m1(v1112, 16);
        int16_t* v1114 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1115 = __riscv_vle16_v_i16m1(v1114, 16);
        vint32m2_t v1116 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1117 = __riscv_vwmacc_vv_i32m2(v1116, v1113, v1104, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1118 = __riscv_vwmacc_vv_i32m2(v1117, v1115, v1105, 16);
        v47 = v1118;
        vint32m2_t v1119 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1120 = __riscv_vwmacc_vv_i32m2(v1119, v1113, v1106, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1121 = __riscv_vwmacc_vv_i32m2(v1120, v1115, v1107, 16);
        v49 = v1121;
        vint32m2_t v1122 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1123 = __riscv_vwmacc_vv_i32m2(v1122, v1113, v1108, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1124 = __riscv_vwmacc_vv_i32m2(v1123, v1115, v1109, 16);
        v51 = v1124;
        vint32m2_t v1125 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1126 = __riscv_vwmacc_vv_i32m2(v1125, v1113, v1110, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1127 = __riscv_vwmacc_vv_i32m2(v1126, v1115, v1111, 16);
        v53 = v1127;
        vint16m1_t v1128;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1129 = __riscv_vmv_v_x_i16m1(0, 16);
        v1128 = v1129;
        vint16m1_t v1130;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1131 = __riscv_vmv_v_x_i16m1(0, 16);
        v1130 = v1131;
        vint16m1_t v1132;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1133 = __riscv_vmv_v_x_i16m1(0, 16);
        v1132 = v1133;
        vint16m1_t v1134;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1135 = __riscv_vmv_v_x_i16m1(0, 16);
        v1134 = v1135;
        vint16m1_t v1136;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1137 = __riscv_vmv_v_x_i16m1(0, 16);
        v1136 = v1137;
        vint16m1_t v1138;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1139 = __riscv_vmv_v_x_i16m1(0, 16);
        v1138 = v1139;
        vint16m1_t v1140;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1141 = __riscv_vmv_v_x_i16m1(0, 16);
        v1140 = v1141;
        vint16m1_t v1142;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1143 = __riscv_vmv_v_x_i16m1(0, 16);
        v1142 = v1143;
        for (size_t v1144 = 0; v1144 < 16; v1144 += 1) {
          size_t v1145 = v1144 * 16;
          size_t v1146 = v1144 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v1147 = 2048 + v1145;
          const uint8_t* v1148 = v30 + v1147;
          const uint8_t* v1149 = (const uint8_t*) v1148;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1150 = __riscv_vle8_v_u8mf2(v1149, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1151 = __riscv_vand_vx_u8mf2(v1150, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1152 = __riscv_vsrl_vx_u8mf2(v1150, 4, 16);
          size_t v1153 = 512 + v1145;
          const uint8_t* v1154 = v30 + v1153;
          const uint8_t* v1155 = (const uint8_t*) v1154;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1156 = __riscv_vle8_v_u8mf2(v1155, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1157 = __riscv_vsrl_vx_u8mf2(v1156, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1158 = __riscv_vand_vx_u8mf2(v1157, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1159 = __riscv_vsll_vx_u8mf2(v1158, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1160 = __riscv_vsrl_vx_u8mf2(v1156, 5, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1161 = __riscv_vand_vx_u8mf2(v1160, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1162 = __riscv_vsll_vx_u8mf2(v1161, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1163 = __riscv_vor_vv_u8mf2(v1151, v1159, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1164 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1163);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1165 = __riscv_vor_vv_u8mf2(v1152, v1162, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1166 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1165);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1167 = 592 + v1146;
          size_t v1168 = 720 + v1146;
          const uint8_t* v1169 = v32 + v1167;
          const int8_t* v1170 = (const int8_t*) v1169;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1171 = *(const int8_t *)(v1170);
          const uint8_t* v1172 = v32 + v1168;
          const int8_t* v1173 = (const int8_t*) v1172;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1174 = *(const int8_t *)(v1173);
          vint16m1_t v1175 = v1128;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1176 = __riscv_vwmacc_vx_i16m1(v1175, v1171, v1164, 16);
          v1128 = v1176;
          vint16m1_t v1177 = v1130;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1178 = __riscv_vwmacc_vx_i16m1(v1177, v1174, v1166, 16);
          v1130 = v1178;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1179 = 593 + v1146;
          size_t v1180 = 721 + v1146;
          const uint8_t* v1181 = v32 + v1179;
          const int8_t* v1182 = (const int8_t*) v1181;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1183 = *(const int8_t *)(v1182);
          const uint8_t* v1184 = v32 + v1180;
          const int8_t* v1185 = (const int8_t*) v1184;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1186 = *(const int8_t *)(v1185);
          vint16m1_t v1187 = v1132;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1188 = __riscv_vwmacc_vx_i16m1(v1187, v1183, v1164, 16);
          v1132 = v1188;
          vint16m1_t v1189 = v1134;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1190 = __riscv_vwmacc_vx_i16m1(v1189, v1186, v1166, 16);
          v1134 = v1190;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1191 = 594 + v1146;
          size_t v1192 = 722 + v1146;
          const uint8_t* v1193 = v32 + v1191;
          const int8_t* v1194 = (const int8_t*) v1193;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1195 = *(const int8_t *)(v1194);
          const uint8_t* v1196 = v32 + v1192;
          const int8_t* v1197 = (const int8_t*) v1196;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1198 = *(const int8_t *)(v1197);
          vint16m1_t v1199 = v1136;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1200 = __riscv_vwmacc_vx_i16m1(v1199, v1195, v1164, 16);
          v1136 = v1200;
          vint16m1_t v1201 = v1138;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1202 = __riscv_vwmacc_vx_i16m1(v1201, v1198, v1166, 16);
          v1138 = v1202;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1203 = 595 + v1146;
          size_t v1204 = 723 + v1146;
          const uint8_t* v1205 = v32 + v1203;
          const int8_t* v1206 = (const int8_t*) v1205;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1207 = *(const int8_t *)(v1206);
          const uint8_t* v1208 = v32 + v1204;
          const int8_t* v1209 = (const int8_t*) v1208;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1210 = *(const int8_t *)(v1209);
          vint16m1_t v1211 = v1140;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1212 = __riscv_vwmacc_vx_i16m1(v1211, v1207, v1164, 16);
          v1140 = v1212;
          vint16m1_t v1213 = v1142;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1214 = __riscv_vwmacc_vx_i16m1(v1213, v1210, v1166, 16);
          v1142 = v1214;
        }
        vint16m1_t v1215 = v1128;
        vint16m1_t v1216 = v1130;
        vint16m1_t v1217 = v1132;
        vint16m1_t v1218 = v1134;
        vint16m1_t v1219 = v1136;
        vint16m1_t v1220 = v1138;
        vint16m1_t v1221 = v1140;
        vint16m1_t v1222 = v1142;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v1223 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1224 = __riscv_vle16_v_i16m1(v1223, 16);
        int16_t* v1225 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1226 = __riscv_vle16_v_i16m1(v1225, 16);
        vint32m2_t v1227 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1228 = __riscv_vwmacc_vv_i32m2(v1227, v1224, v1215, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1229 = __riscv_vwmacc_vv_i32m2(v1228, v1226, v1216, 16);
        v47 = v1229;
        vint32m2_t v1230 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1231 = __riscv_vwmacc_vv_i32m2(v1230, v1224, v1217, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1232 = __riscv_vwmacc_vv_i32m2(v1231, v1226, v1218, 16);
        v49 = v1232;
        vint32m2_t v1233 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1234 = __riscv_vwmacc_vv_i32m2(v1233, v1224, v1219, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1235 = __riscv_vwmacc_vv_i32m2(v1234, v1226, v1220, 16);
        v51 = v1235;
        vint32m2_t v1236 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1237 = __riscv_vwmacc_vv_i32m2(v1236, v1224, v1221, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1238 = __riscv_vwmacc_vv_i32m2(v1237, v1226, v1222, 16);
        v53 = v1238;
        vint16m1_t v1239;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1240 = __riscv_vmv_v_x_i16m1(0, 16);
        v1239 = v1240;
        vint16m1_t v1241;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1242 = __riscv_vmv_v_x_i16m1(0, 16);
        v1241 = v1242;
        vint16m1_t v1243;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1244 = __riscv_vmv_v_x_i16m1(0, 16);
        v1243 = v1244;
        vint16m1_t v1245;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1246 = __riscv_vmv_v_x_i16m1(0, 16);
        v1245 = v1246;
        vint16m1_t v1247;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1248 = __riscv_vmv_v_x_i16m1(0, 16);
        v1247 = v1248;
        vint16m1_t v1249;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1250 = __riscv_vmv_v_x_i16m1(0, 16);
        v1249 = v1250;
        vint16m1_t v1251;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1252 = __riscv_vmv_v_x_i16m1(0, 16);
        v1251 = v1252;
        vint16m1_t v1253;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1254 = __riscv_vmv_v_x_i16m1(0, 16);
        v1253 = v1254;
        for (size_t v1255 = 0; v1255 < 16; v1255 += 1) {
          size_t v1256 = v1255 * 16;
          size_t v1257 = v1255 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v1258 = 2304 + v1256;
          const uint8_t* v1259 = v30 + v1258;
          const uint8_t* v1260 = (const uint8_t*) v1259;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1261 = __riscv_vle8_v_u8mf2(v1260, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1262 = __riscv_vand_vx_u8mf2(v1261, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1263 = __riscv_vsrl_vx_u8mf2(v1261, 4, 16);
          size_t v1264 = 256 + v1256;
          const uint8_t* v1265 = v30 + v1264;
          const uint8_t* v1266 = (const uint8_t*) v1265;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1267 = __riscv_vle8_v_u8mf2(v1266, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1268 = __riscv_vsrl_vx_u8mf2(v1267, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1269 = __riscv_vand_vx_u8mf2(v1268, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1270 = __riscv_vsll_vx_u8mf2(v1269, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1271 = __riscv_vsrl_vx_u8mf2(v1267, 7, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1272 = __riscv_vand_vx_u8mf2(v1271, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1273 = __riscv_vsll_vx_u8mf2(v1272, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1274 = __riscv_vor_vv_u8mf2(v1262, v1270, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1275 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1274);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1276 = __riscv_vor_vv_u8mf2(v1263, v1273, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1277 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1276);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1278 = 784 + v1257;
          size_t v1279 = 912 + v1257;
          const uint8_t* v1280 = v32 + v1278;
          const int8_t* v1281 = (const int8_t*) v1280;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1282 = *(const int8_t *)(v1281);
          const uint8_t* v1283 = v32 + v1279;
          const int8_t* v1284 = (const int8_t*) v1283;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1285 = *(const int8_t *)(v1284);
          vint16m1_t v1286 = v1239;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1287 = __riscv_vwmacc_vx_i16m1(v1286, v1282, v1275, 16);
          v1239 = v1287;
          vint16m1_t v1288 = v1241;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1289 = __riscv_vwmacc_vx_i16m1(v1288, v1285, v1277, 16);
          v1241 = v1289;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1290 = 785 + v1257;
          size_t v1291 = 913 + v1257;
          const uint8_t* v1292 = v32 + v1290;
          const int8_t* v1293 = (const int8_t*) v1292;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1294 = *(const int8_t *)(v1293);
          const uint8_t* v1295 = v32 + v1291;
          const int8_t* v1296 = (const int8_t*) v1295;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1297 = *(const int8_t *)(v1296);
          vint16m1_t v1298 = v1243;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1299 = __riscv_vwmacc_vx_i16m1(v1298, v1294, v1275, 16);
          v1243 = v1299;
          vint16m1_t v1300 = v1245;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1301 = __riscv_vwmacc_vx_i16m1(v1300, v1297, v1277, 16);
          v1245 = v1301;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1302 = 786 + v1257;
          size_t v1303 = 914 + v1257;
          const uint8_t* v1304 = v32 + v1302;
          const int8_t* v1305 = (const int8_t*) v1304;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1306 = *(const int8_t *)(v1305);
          const uint8_t* v1307 = v32 + v1303;
          const int8_t* v1308 = (const int8_t*) v1307;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1309 = *(const int8_t *)(v1308);
          vint16m1_t v1310 = v1247;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1311 = __riscv_vwmacc_vx_i16m1(v1310, v1306, v1275, 16);
          v1247 = v1311;
          vint16m1_t v1312 = v1249;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1313 = __riscv_vwmacc_vx_i16m1(v1312, v1309, v1277, 16);
          v1249 = v1313;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1314 = 787 + v1257;
          size_t v1315 = 915 + v1257;
          const uint8_t* v1316 = v32 + v1314;
          const int8_t* v1317 = (const int8_t*) v1316;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1318 = *(const int8_t *)(v1317);
          const uint8_t* v1319 = v32 + v1315;
          const int8_t* v1320 = (const int8_t*) v1319;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1321 = *(const int8_t *)(v1320);
          vint16m1_t v1322 = v1251;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1323 = __riscv_vwmacc_vx_i16m1(v1322, v1318, v1275, 16);
          v1251 = v1323;
          vint16m1_t v1324 = v1253;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1325 = __riscv_vwmacc_vx_i16m1(v1324, v1321, v1277, 16);
          v1253 = v1325;
        }
        vint16m1_t v1326 = v1239;
        vint16m1_t v1327 = v1241;
        vint16m1_t v1328 = v1243;
        vint16m1_t v1329 = v1245;
        vint16m1_t v1330 = v1247;
        vint16m1_t v1331 = v1249;
        vint16m1_t v1332 = v1251;
        vint16m1_t v1333 = v1253;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v1334 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1335 = __riscv_vle16_v_i16m1(v1334, 16);
        int16_t* v1336 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1337 = __riscv_vle16_v_i16m1(v1336, 16);
        vint32m2_t v1338 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1339 = __riscv_vwmacc_vv_i32m2(v1338, v1335, v1326, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1340 = __riscv_vwmacc_vv_i32m2(v1339, v1337, v1327, 16);
        v47 = v1340;
        vint32m2_t v1341 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1342 = __riscv_vwmacc_vv_i32m2(v1341, v1335, v1328, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1343 = __riscv_vwmacc_vv_i32m2(v1342, v1337, v1329, 16);
        v49 = v1343;
        vint32m2_t v1344 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1345 = __riscv_vwmacc_vv_i32m2(v1344, v1335, v1330, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1346 = __riscv_vwmacc_vv_i32m2(v1345, v1337, v1331, 16);
        v51 = v1346;
        vint32m2_t v1347 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1348 = __riscv_vwmacc_vv_i32m2(v1347, v1335, v1332, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1349 = __riscv_vwmacc_vv_i32m2(v1348, v1337, v1333, 16);
        v53 = v1349;
        vint16m1_t v1350;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1351 = __riscv_vmv_v_x_i16m1(0, 16);
        v1350 = v1351;
        vint16m1_t v1352;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1353 = __riscv_vmv_v_x_i16m1(0, 16);
        v1352 = v1353;
        vint16m1_t v1354;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1355 = __riscv_vmv_v_x_i16m1(0, 16);
        v1354 = v1355;
        vint16m1_t v1356;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1357 = __riscv_vmv_v_x_i16m1(0, 16);
        v1356 = v1357;
        vint16m1_t v1358;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1359 = __riscv_vmv_v_x_i16m1(0, 16);
        v1358 = v1359;
        vint16m1_t v1360;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1361 = __riscv_vmv_v_x_i16m1(0, 16);
        v1360 = v1361;
        vint16m1_t v1362;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1363 = __riscv_vmv_v_x_i16m1(0, 16);
        v1362 = v1363;
        vint16m1_t v1364;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1365 = __riscv_vmv_v_x_i16m1(0, 16);
        v1364 = v1365;
        for (size_t v1366 = 0; v1366 < 16; v1366 += 1) {
          size_t v1367 = v1366 * 16;
          size_t v1368 = v1366 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v1369 = 2560 + v1367;
          const uint8_t* v1370 = v30 + v1369;
          const uint8_t* v1371 = (const uint8_t*) v1370;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1372 = __riscv_vle8_v_u8mf2(v1371, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1373 = __riscv_vand_vx_u8mf2(v1372, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1374 = __riscv_vsrl_vx_u8mf2(v1372, 4, 16);
          size_t v1375 = 512 + v1367;
          const uint8_t* v1376 = v30 + v1375;
          const uint8_t* v1377 = (const uint8_t*) v1376;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1378 = __riscv_vle8_v_u8mf2(v1377, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1379 = __riscv_vsrl_vx_u8mf2(v1378, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1380 = __riscv_vand_vx_u8mf2(v1379, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1381 = __riscv_vsll_vx_u8mf2(v1380, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1382 = __riscv_vsrl_vx_u8mf2(v1378, 7, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1383 = __riscv_vand_vx_u8mf2(v1382, 0x01, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
          vuint8mf2_t v1384 = __riscv_vsll_vx_u8mf2(v1383, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1385 = __riscv_vor_vv_u8mf2(v1373, v1381, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1386 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1385);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
          vuint8mf2_t v1387 = __riscv_vor_vv_u8mf2(v1374, v1384, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1388 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1387);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1389 = 848 + v1368;
          size_t v1390 = 976 + v1368;
          const uint8_t* v1391 = v32 + v1389;
          const int8_t* v1392 = (const int8_t*) v1391;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1393 = *(const int8_t *)(v1392);
          const uint8_t* v1394 = v32 + v1390;
          const int8_t* v1395 = (const int8_t*) v1394;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1396 = *(const int8_t *)(v1395);
          vint16m1_t v1397 = v1350;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1398 = __riscv_vwmacc_vx_i16m1(v1397, v1393, v1386, 16);
          v1350 = v1398;
          vint16m1_t v1399 = v1352;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1400 = __riscv_vwmacc_vx_i16m1(v1399, v1396, v1388, 16);
          v1352 = v1400;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1401 = 849 + v1368;
          size_t v1402 = 977 + v1368;
          const uint8_t* v1403 = v32 + v1401;
          const int8_t* v1404 = (const int8_t*) v1403;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1405 = *(const int8_t *)(v1404);
          const uint8_t* v1406 = v32 + v1402;
          const int8_t* v1407 = (const int8_t*) v1406;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1408 = *(const int8_t *)(v1407);
          vint16m1_t v1409 = v1354;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1410 = __riscv_vwmacc_vx_i16m1(v1409, v1405, v1386, 16);
          v1354 = v1410;
          vint16m1_t v1411 = v1356;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1412 = __riscv_vwmacc_vx_i16m1(v1411, v1408, v1388, 16);
          v1356 = v1412;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1413 = 850 + v1368;
          size_t v1414 = 978 + v1368;
          const uint8_t* v1415 = v32 + v1413;
          const int8_t* v1416 = (const int8_t*) v1415;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1417 = *(const int8_t *)(v1416);
          const uint8_t* v1418 = v32 + v1414;
          const int8_t* v1419 = (const int8_t*) v1418;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1420 = *(const int8_t *)(v1419);
          vint16m1_t v1421 = v1358;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1422 = __riscv_vwmacc_vx_i16m1(v1421, v1417, v1386, 16);
          v1358 = v1422;
          vint16m1_t v1423 = v1360;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1424 = __riscv_vwmacc_vx_i16m1(v1423, v1420, v1388, 16);
          v1360 = v1424;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1425 = 851 + v1368;
          size_t v1426 = 979 + v1368;
          const uint8_t* v1427 = v32 + v1425;
          const int8_t* v1428 = (const int8_t*) v1427;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1429 = *(const int8_t *)(v1428);
          const uint8_t* v1430 = v32 + v1426;
          const int8_t* v1431 = (const int8_t*) v1430;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1432 = *(const int8_t *)(v1431);
          vint16m1_t v1433 = v1362;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1434 = __riscv_vwmacc_vx_i16m1(v1433, v1429, v1386, 16);
          v1362 = v1434;
          vint16m1_t v1435 = v1364;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1436 = __riscv_vwmacc_vx_i16m1(v1435, v1432, v1388, 16);
          v1364 = v1436;
        }
        vint16m1_t v1437 = v1350;
        vint16m1_t v1438 = v1352;
        vint16m1_t v1439 = v1354;
        vint16m1_t v1440 = v1356;
        vint16m1_t v1441 = v1358;
        vint16m1_t v1442 = v1360;
        vint16m1_t v1443 = v1362;
        vint16m1_t v1444 = v1364;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v1445 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1446 = __riscv_vle16_v_i16m1(v1445, 16);
        int16_t* v1447 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1448 = __riscv_vle16_v_i16m1(v1447, 16);
        vint32m2_t v1449 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1450 = __riscv_vwmacc_vv_i32m2(v1449, v1446, v1437, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1451 = __riscv_vwmacc_vv_i32m2(v1450, v1448, v1438, 16);
        v47 = v1451;
        vint32m2_t v1452 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1453 = __riscv_vwmacc_vv_i32m2(v1452, v1446, v1439, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1454 = __riscv_vwmacc_vv_i32m2(v1453, v1448, v1440, 16);
        v49 = v1454;
        vint32m2_t v1455 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1456 = __riscv_vwmacc_vv_i32m2(v1455, v1446, v1441, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1457 = __riscv_vwmacc_vv_i32m2(v1456, v1448, v1442, 16);
        v51 = v1457;
        vint32m2_t v1458 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1459 = __riscv_vwmacc_vv_i32m2(v1458, v1446, v1443, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1460 = __riscv_vwmacc_vv_i32m2(v1459, v1448, v1444, 16);
        v53 = v1460;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const uint8_t* v1461 = v30 + 32;
        const _Float16* v1462 = (const _Float16*) v1461;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v1463 = __riscv_vle16_v_f16m1(v1462, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v1464 = __riscv_vfwcvt_f_f_v_f32m2(v1463, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const _Float16* v1465 = (const _Float16*) v30;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v1466 = __riscv_vle16_v_f16m1(v1465, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v1467 = __riscv_vfwcvt_f_f_v_f32m2(v1466, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1468 = __riscv_vfmul_vf_f32m2(v1467, v34, 16);
        vint32m2_t v1469 = v47;
        vfloat32m2_t v1470 = v20;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1471 = __riscv_vfcvt_f_x_v_f32m2(v1469, 16);
        vfloat32m2_t v1472 = __riscv_vfmacc_vv_f32m2(v1470, v1471, v1468, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1473 = __riscv_vfmul_vf_f32m2(v1464, v34, 16);
        int32_t* v1474 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1475 = __riscv_vle32_v_i32m2(v1474, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1476 = __riscv_vfcvt_f_x_v_f32m2(v1475, 16);
        vfloat32m2_t v1477 = __riscv_vfnmsac_vv_f32m2(v1472, v1473, v1476, 16);
        v20 = v1477;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1478 = __riscv_vfmul_vf_f32m2(v1467, v37, 16);
        vint32m2_t v1479 = v49;
        vfloat32m2_t v1480 = v22;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1481 = __riscv_vfcvt_f_x_v_f32m2(v1479, 16);
        vfloat32m2_t v1482 = __riscv_vfmacc_vv_f32m2(v1480, v1481, v1478, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1483 = __riscv_vfmul_vf_f32m2(v1464, v37, 16);
        int32_t* v1484 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1485 = __riscv_vle32_v_i32m2(v1484, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1486 = __riscv_vfcvt_f_x_v_f32m2(v1485, 16);
        vfloat32m2_t v1487 = __riscv_vfnmsac_vv_f32m2(v1482, v1483, v1486, 16);
        v22 = v1487;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1488 = __riscv_vfmul_vf_f32m2(v1467, v40, 16);
        vint32m2_t v1489 = v51;
        vfloat32m2_t v1490 = v24;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1491 = __riscv_vfcvt_f_x_v_f32m2(v1489, 16);
        vfloat32m2_t v1492 = __riscv_vfmacc_vv_f32m2(v1490, v1491, v1488, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1493 = __riscv_vfmul_vf_f32m2(v1464, v40, 16);
        int32_t* v1494 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1495 = __riscv_vle32_v_i32m2(v1494, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1496 = __riscv_vfcvt_f_x_v_f32m2(v1495, 16);
        vfloat32m2_t v1497 = __riscv_vfnmsac_vv_f32m2(v1492, v1493, v1496, 16);
        v24 = v1497;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1498 = __riscv_vfmul_vf_f32m2(v1467, v43, 16);
        vint32m2_t v1499 = v53;
        vfloat32m2_t v1500 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1501 = __riscv_vfcvt_f_x_v_f32m2(v1499, 16);
        vfloat32m2_t v1502 = __riscv_vfmacc_vv_f32m2(v1500, v1501, v1498, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1503 = __riscv_vfmul_vf_f32m2(v1464, v43, 16);
        int32_t* v1504 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1505 = __riscv_vle32_v_i32m2(v1504, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1506 = __riscv_vfcvt_f_x_v_f32m2(v1505, 16);
        vfloat32m2_t v1507 = __riscv_vfnmsac_vv_f32m2(v1502, v1503, v1506, 16);
        v26 = v1507;
      }
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1508 = v12 * 4;
      size_t v1509 = v1508 + 0;
      size_t v1510 = v1509 * v7;
      size_t v1511 = v16 * 16;
      size_t v1512 = v1510 + v1511;
      float* v1513 = v2 + v1512;
      vfloat32m2_t v1514 = v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1513, v1514, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1515 = v12 * 4;
      size_t v1516 = v1515 + 1;
      size_t v1517 = v1516 * v7;
      size_t v1518 = v16 * 16;
      size_t v1519 = v1517 + v1518;
      float* v1520 = v2 + v1519;
      vfloat32m2_t v1521 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1520, v1521, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1522 = v12 * 4;
      size_t v1523 = v1522 + 2;
      size_t v1524 = v1523 * v7;
      size_t v1525 = v16 * 16;
      size_t v1526 = v1524 + v1525;
      float* v1527 = v2 + v1526;
      vfloat32m2_t v1528 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1527, v1528, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1529 = v12 * 4;
      size_t v1530 = v1529 + 3;
      size_t v1531 = v1530 * v7;
      size_t v1532 = v16 * 16;
      size_t v1533 = v1531 + v1532;
      float* v1534 = v2 + v1533;
      vfloat32m2_t v1535 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1534, v1535, 16);
    }
  }
  return;
}


