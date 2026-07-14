#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5, size_t v6, size_t v7) {
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
  for (size_t v12 = 0; v12 < v11; v12 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_group_base
    size_t v13 = v12 * v9;
    size_t v14 = v13 * 2304;
    const uint8_t* v15 = v3 + v14;
    for (size_t v16 = 0; v16 < v10; v16 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_group_base
      size_t v17 = v16 * v9;
      size_t v18 = v17 * 1168;
      const uint8_t* v19 = v4 + v18;
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
        size_t v29 = v28 * 2304;
        const uint8_t* v30 = v15 + v29;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
        size_t v31 = v28 * 1168;
        const uint8_t* v32 = v19 + v31;
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
          size_t v334 = 256 + v332;
          const uint8_t* v335 = v30 + v334;
          const uint8_t* v336 = (const uint8_t*) v335;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v337 = __riscv_vle8_v_u8mf2(v336, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v338 = __riscv_vand_vx_u8mf2(v337, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v339 = __riscv_vreinterpret_v_u8mf2_i8mf2(v338);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v340 = __riscv_vsrl_vx_u8mf2(v337, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v341 = __riscv_vreinterpret_v_u8mf2_i8mf2(v340);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v342 = 16 + v333;
          size_t v343 = 144 + v333;
          const uint8_t* v344 = v32 + v342;
          const int8_t* v345 = (const int8_t*) v344;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v346 = *(const int8_t *)(v345);
          const uint8_t* v347 = v32 + v343;
          const int8_t* v348 = (const int8_t*) v347;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v349 = *(const int8_t *)(v348);
          vint16m1_t v350 = v315;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v351 = __riscv_vwmacc_vx_i16m1(v350, v346, v339, 16);
          v315 = v351;
          vint16m1_t v352 = v317;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v353 = __riscv_vwmacc_vx_i16m1(v352, v349, v341, 16);
          v317 = v353;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v354 = 17 + v333;
          size_t v355 = 145 + v333;
          const uint8_t* v356 = v32 + v354;
          const int8_t* v357 = (const int8_t*) v356;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v358 = *(const int8_t *)(v357);
          const uint8_t* v359 = v32 + v355;
          const int8_t* v360 = (const int8_t*) v359;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v361 = *(const int8_t *)(v360);
          vint16m1_t v362 = v319;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v363 = __riscv_vwmacc_vx_i16m1(v362, v358, v339, 16);
          v319 = v363;
          vint16m1_t v364 = v321;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v365 = __riscv_vwmacc_vx_i16m1(v364, v361, v341, 16);
          v321 = v365;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v366 = 18 + v333;
          size_t v367 = 146 + v333;
          const uint8_t* v368 = v32 + v366;
          const int8_t* v369 = (const int8_t*) v368;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v370 = *(const int8_t *)(v369);
          const uint8_t* v371 = v32 + v367;
          const int8_t* v372 = (const int8_t*) v371;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v373 = *(const int8_t *)(v372);
          vint16m1_t v374 = v323;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v375 = __riscv_vwmacc_vx_i16m1(v374, v370, v339, 16);
          v323 = v375;
          vint16m1_t v376 = v325;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v377 = __riscv_vwmacc_vx_i16m1(v376, v373, v341, 16);
          v325 = v377;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v378 = 19 + v333;
          size_t v379 = 147 + v333;
          const uint8_t* v380 = v32 + v378;
          const int8_t* v381 = (const int8_t*) v380;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v382 = *(const int8_t *)(v381);
          const uint8_t* v383 = v32 + v379;
          const int8_t* v384 = (const int8_t*) v383;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v385 = *(const int8_t *)(v384);
          vint16m1_t v386 = v327;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v387 = __riscv_vwmacc_vx_i16m1(v386, v382, v339, 16);
          v327 = v387;
          vint16m1_t v388 = v329;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v389 = __riscv_vwmacc_vx_i16m1(v388, v385, v341, 16);
          v329 = v389;
        }
        vint16m1_t v390 = v315;
        vint16m1_t v391 = v317;
        vint16m1_t v392 = v319;
        vint16m1_t v393 = v321;
        vint16m1_t v394 = v323;
        vint16m1_t v395 = v325;
        vint16m1_t v396 = v327;
        vint16m1_t v397 = v329;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v398 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v399 = __riscv_vle16_v_i16m1(v398, 16);
        int16_t* v400 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v401 = __riscv_vle16_v_i16m1(v400, 16);
        vint32m2_t v402 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v403 = __riscv_vwmacc_vv_i32m2(v402, v399, v390, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v404 = __riscv_vwmacc_vv_i32m2(v403, v401, v391, 16);
        v47 = v404;
        vint32m2_t v405 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v406 = __riscv_vwmacc_vv_i32m2(v405, v399, v392, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v407 = __riscv_vwmacc_vv_i32m2(v406, v401, v393, 16);
        v49 = v407;
        vint32m2_t v408 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v409 = __riscv_vwmacc_vv_i32m2(v408, v399, v394, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v410 = __riscv_vwmacc_vv_i32m2(v409, v401, v395, 16);
        v51 = v410;
        vint32m2_t v411 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v412 = __riscv_vwmacc_vv_i32m2(v411, v399, v396, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v413 = __riscv_vwmacc_vv_i32m2(v412, v401, v397, 16);
        v53 = v413;
        vint16m1_t v414;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v415 = __riscv_vmv_v_x_i16m1(0, 16);
        v414 = v415;
        vint16m1_t v416;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v417 = __riscv_vmv_v_x_i16m1(0, 16);
        v416 = v417;
        vint16m1_t v418;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v419 = __riscv_vmv_v_x_i16m1(0, 16);
        v418 = v419;
        vint16m1_t v420;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v421 = __riscv_vmv_v_x_i16m1(0, 16);
        v420 = v421;
        vint16m1_t v422;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v423 = __riscv_vmv_v_x_i16m1(0, 16);
        v422 = v423;
        vint16m1_t v424;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v425 = __riscv_vmv_v_x_i16m1(0, 16);
        v424 = v425;
        vint16m1_t v426;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v427 = __riscv_vmv_v_x_i16m1(0, 16);
        v426 = v427;
        vint16m1_t v428;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v429 = __riscv_vmv_v_x_i16m1(0, 16);
        v428 = v429;
        for (size_t v430 = 0; v430 < 16; v430 += 1) {
          size_t v431 = v430 * 16;
          size_t v432 = v430 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v433 = 512 + v431;
          const uint8_t* v434 = v30 + v433;
          const uint8_t* v435 = (const uint8_t*) v434;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v436 = __riscv_vle8_v_u8mf2(v435, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v437 = __riscv_vand_vx_u8mf2(v436, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v438 = __riscv_vreinterpret_v_u8mf2_i8mf2(v437);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v439 = __riscv_vsrl_vx_u8mf2(v436, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v440 = __riscv_vreinterpret_v_u8mf2_i8mf2(v439);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v441 = 80 + v432;
          size_t v442 = 208 + v432;
          const uint8_t* v443 = v32 + v441;
          const int8_t* v444 = (const int8_t*) v443;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v445 = *(const int8_t *)(v444);
          const uint8_t* v446 = v32 + v442;
          const int8_t* v447 = (const int8_t*) v446;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v448 = *(const int8_t *)(v447);
          vint16m1_t v449 = v414;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v450 = __riscv_vwmacc_vx_i16m1(v449, v445, v438, 16);
          v414 = v450;
          vint16m1_t v451 = v416;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v452 = __riscv_vwmacc_vx_i16m1(v451, v448, v440, 16);
          v416 = v452;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v453 = 81 + v432;
          size_t v454 = 209 + v432;
          const uint8_t* v455 = v32 + v453;
          const int8_t* v456 = (const int8_t*) v455;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v457 = *(const int8_t *)(v456);
          const uint8_t* v458 = v32 + v454;
          const int8_t* v459 = (const int8_t*) v458;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v460 = *(const int8_t *)(v459);
          vint16m1_t v461 = v418;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v462 = __riscv_vwmacc_vx_i16m1(v461, v457, v438, 16);
          v418 = v462;
          vint16m1_t v463 = v420;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v464 = __riscv_vwmacc_vx_i16m1(v463, v460, v440, 16);
          v420 = v464;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v465 = 82 + v432;
          size_t v466 = 210 + v432;
          const uint8_t* v467 = v32 + v465;
          const int8_t* v468 = (const int8_t*) v467;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v469 = *(const int8_t *)(v468);
          const uint8_t* v470 = v32 + v466;
          const int8_t* v471 = (const int8_t*) v470;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v472 = *(const int8_t *)(v471);
          vint16m1_t v473 = v422;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v474 = __riscv_vwmacc_vx_i16m1(v473, v469, v438, 16);
          v422 = v474;
          vint16m1_t v475 = v424;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v476 = __riscv_vwmacc_vx_i16m1(v475, v472, v440, 16);
          v424 = v476;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v477 = 83 + v432;
          size_t v478 = 211 + v432;
          const uint8_t* v479 = v32 + v477;
          const int8_t* v480 = (const int8_t*) v479;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v481 = *(const int8_t *)(v480);
          const uint8_t* v482 = v32 + v478;
          const int8_t* v483 = (const int8_t*) v482;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v484 = *(const int8_t *)(v483);
          vint16m1_t v485 = v426;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v486 = __riscv_vwmacc_vx_i16m1(v485, v481, v438, 16);
          v426 = v486;
          vint16m1_t v487 = v428;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v488 = __riscv_vwmacc_vx_i16m1(v487, v484, v440, 16);
          v428 = v488;
        }
        vint16m1_t v489 = v414;
        vint16m1_t v490 = v416;
        vint16m1_t v491 = v418;
        vint16m1_t v492 = v420;
        vint16m1_t v493 = v422;
        vint16m1_t v494 = v424;
        vint16m1_t v495 = v426;
        vint16m1_t v496 = v428;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v497 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v498 = __riscv_vle16_v_i16m1(v497, 16);
        int16_t* v499 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v500 = __riscv_vle16_v_i16m1(v499, 16);
        vint32m2_t v501 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v502 = __riscv_vwmacc_vv_i32m2(v501, v498, v489, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v503 = __riscv_vwmacc_vv_i32m2(v502, v500, v490, 16);
        v47 = v503;
        vint32m2_t v504 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v505 = __riscv_vwmacc_vv_i32m2(v504, v498, v491, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v506 = __riscv_vwmacc_vv_i32m2(v505, v500, v492, 16);
        v49 = v506;
        vint32m2_t v507 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v508 = __riscv_vwmacc_vv_i32m2(v507, v498, v493, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v509 = __riscv_vwmacc_vv_i32m2(v508, v500, v494, 16);
        v51 = v509;
        vint32m2_t v510 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v511 = __riscv_vwmacc_vv_i32m2(v510, v498, v495, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v512 = __riscv_vwmacc_vv_i32m2(v511, v500, v496, 16);
        v53 = v512;
        vint16m1_t v513;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v514 = __riscv_vmv_v_x_i16m1(0, 16);
        v513 = v514;
        vint16m1_t v515;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v516 = __riscv_vmv_v_x_i16m1(0, 16);
        v515 = v516;
        vint16m1_t v517;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v518 = __riscv_vmv_v_x_i16m1(0, 16);
        v517 = v518;
        vint16m1_t v519;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v520 = __riscv_vmv_v_x_i16m1(0, 16);
        v519 = v520;
        vint16m1_t v521;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v522 = __riscv_vmv_v_x_i16m1(0, 16);
        v521 = v522;
        vint16m1_t v523;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v524 = __riscv_vmv_v_x_i16m1(0, 16);
        v523 = v524;
        vint16m1_t v525;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v526 = __riscv_vmv_v_x_i16m1(0, 16);
        v525 = v526;
        vint16m1_t v527;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v528 = __riscv_vmv_v_x_i16m1(0, 16);
        v527 = v528;
        for (size_t v529 = 0; v529 < 16; v529 += 1) {
          size_t v530 = v529 * 16;
          size_t v531 = v529 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v532 = 768 + v530;
          const uint8_t* v533 = v30 + v532;
          const uint8_t* v534 = (const uint8_t*) v533;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v535 = __riscv_vle8_v_u8mf2(v534, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v536 = __riscv_vand_vx_u8mf2(v535, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v537 = __riscv_vreinterpret_v_u8mf2_i8mf2(v536);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v538 = __riscv_vsrl_vx_u8mf2(v535, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v539 = __riscv_vreinterpret_v_u8mf2_i8mf2(v538);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v540 = 272 + v531;
          size_t v541 = 400 + v531;
          const uint8_t* v542 = v32 + v540;
          const int8_t* v543 = (const int8_t*) v542;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v544 = *(const int8_t *)(v543);
          const uint8_t* v545 = v32 + v541;
          const int8_t* v546 = (const int8_t*) v545;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v547 = *(const int8_t *)(v546);
          vint16m1_t v548 = v513;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v549 = __riscv_vwmacc_vx_i16m1(v548, v544, v537, 16);
          v513 = v549;
          vint16m1_t v550 = v515;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v551 = __riscv_vwmacc_vx_i16m1(v550, v547, v539, 16);
          v515 = v551;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v552 = 273 + v531;
          size_t v553 = 401 + v531;
          const uint8_t* v554 = v32 + v552;
          const int8_t* v555 = (const int8_t*) v554;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v556 = *(const int8_t *)(v555);
          const uint8_t* v557 = v32 + v553;
          const int8_t* v558 = (const int8_t*) v557;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v559 = *(const int8_t *)(v558);
          vint16m1_t v560 = v517;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v561 = __riscv_vwmacc_vx_i16m1(v560, v556, v537, 16);
          v517 = v561;
          vint16m1_t v562 = v519;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v563 = __riscv_vwmacc_vx_i16m1(v562, v559, v539, 16);
          v519 = v563;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v564 = 274 + v531;
          size_t v565 = 402 + v531;
          const uint8_t* v566 = v32 + v564;
          const int8_t* v567 = (const int8_t*) v566;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v568 = *(const int8_t *)(v567);
          const uint8_t* v569 = v32 + v565;
          const int8_t* v570 = (const int8_t*) v569;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v571 = *(const int8_t *)(v570);
          vint16m1_t v572 = v521;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v573 = __riscv_vwmacc_vx_i16m1(v572, v568, v537, 16);
          v521 = v573;
          vint16m1_t v574 = v523;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v575 = __riscv_vwmacc_vx_i16m1(v574, v571, v539, 16);
          v523 = v575;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v576 = 275 + v531;
          size_t v577 = 403 + v531;
          const uint8_t* v578 = v32 + v576;
          const int8_t* v579 = (const int8_t*) v578;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v580 = *(const int8_t *)(v579);
          const uint8_t* v581 = v32 + v577;
          const int8_t* v582 = (const int8_t*) v581;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v583 = *(const int8_t *)(v582);
          vint16m1_t v584 = v525;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v585 = __riscv_vwmacc_vx_i16m1(v584, v580, v537, 16);
          v525 = v585;
          vint16m1_t v586 = v527;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v587 = __riscv_vwmacc_vx_i16m1(v586, v583, v539, 16);
          v527 = v587;
        }
        vint16m1_t v588 = v513;
        vint16m1_t v589 = v515;
        vint16m1_t v590 = v517;
        vint16m1_t v591 = v519;
        vint16m1_t v592 = v521;
        vint16m1_t v593 = v523;
        vint16m1_t v594 = v525;
        vint16m1_t v595 = v527;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v596 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v597 = __riscv_vle16_v_i16m1(v596, 16);
        int16_t* v598 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v599 = __riscv_vle16_v_i16m1(v598, 16);
        vint32m2_t v600 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v601 = __riscv_vwmacc_vv_i32m2(v600, v597, v588, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v602 = __riscv_vwmacc_vv_i32m2(v601, v599, v589, 16);
        v47 = v602;
        vint32m2_t v603 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v604 = __riscv_vwmacc_vv_i32m2(v603, v597, v590, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v605 = __riscv_vwmacc_vv_i32m2(v604, v599, v591, 16);
        v49 = v605;
        vint32m2_t v606 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v607 = __riscv_vwmacc_vv_i32m2(v606, v597, v592, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v608 = __riscv_vwmacc_vv_i32m2(v607, v599, v593, 16);
        v51 = v608;
        vint32m2_t v609 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v610 = __riscv_vwmacc_vv_i32m2(v609, v597, v594, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v611 = __riscv_vwmacc_vv_i32m2(v610, v599, v595, 16);
        v53 = v611;
        vint16m1_t v612;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v613 = __riscv_vmv_v_x_i16m1(0, 16);
        v612 = v613;
        vint16m1_t v614;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v615 = __riscv_vmv_v_x_i16m1(0, 16);
        v614 = v615;
        vint16m1_t v616;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v617 = __riscv_vmv_v_x_i16m1(0, 16);
        v616 = v617;
        vint16m1_t v618;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v619 = __riscv_vmv_v_x_i16m1(0, 16);
        v618 = v619;
        vint16m1_t v620;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v621 = __riscv_vmv_v_x_i16m1(0, 16);
        v620 = v621;
        vint16m1_t v622;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v623 = __riscv_vmv_v_x_i16m1(0, 16);
        v622 = v623;
        vint16m1_t v624;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v625 = __riscv_vmv_v_x_i16m1(0, 16);
        v624 = v625;
        vint16m1_t v626;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v627 = __riscv_vmv_v_x_i16m1(0, 16);
        v626 = v627;
        for (size_t v628 = 0; v628 < 16; v628 += 1) {
          size_t v629 = v628 * 16;
          size_t v630 = v628 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v631 = 1024 + v629;
          const uint8_t* v632 = v30 + v631;
          const uint8_t* v633 = (const uint8_t*) v632;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v634 = __riscv_vle8_v_u8mf2(v633, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v635 = __riscv_vand_vx_u8mf2(v634, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v636 = __riscv_vreinterpret_v_u8mf2_i8mf2(v635);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v637 = __riscv_vsrl_vx_u8mf2(v634, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v638 = __riscv_vreinterpret_v_u8mf2_i8mf2(v637);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v639 = 336 + v630;
          size_t v640 = 464 + v630;
          const uint8_t* v641 = v32 + v639;
          const int8_t* v642 = (const int8_t*) v641;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v643 = *(const int8_t *)(v642);
          const uint8_t* v644 = v32 + v640;
          const int8_t* v645 = (const int8_t*) v644;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v646 = *(const int8_t *)(v645);
          vint16m1_t v647 = v612;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v648 = __riscv_vwmacc_vx_i16m1(v647, v643, v636, 16);
          v612 = v648;
          vint16m1_t v649 = v614;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v650 = __riscv_vwmacc_vx_i16m1(v649, v646, v638, 16);
          v614 = v650;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v651 = 337 + v630;
          size_t v652 = 465 + v630;
          const uint8_t* v653 = v32 + v651;
          const int8_t* v654 = (const int8_t*) v653;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v655 = *(const int8_t *)(v654);
          const uint8_t* v656 = v32 + v652;
          const int8_t* v657 = (const int8_t*) v656;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v658 = *(const int8_t *)(v657);
          vint16m1_t v659 = v616;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v660 = __riscv_vwmacc_vx_i16m1(v659, v655, v636, 16);
          v616 = v660;
          vint16m1_t v661 = v618;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v662 = __riscv_vwmacc_vx_i16m1(v661, v658, v638, 16);
          v618 = v662;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v663 = 338 + v630;
          size_t v664 = 466 + v630;
          const uint8_t* v665 = v32 + v663;
          const int8_t* v666 = (const int8_t*) v665;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v667 = *(const int8_t *)(v666);
          const uint8_t* v668 = v32 + v664;
          const int8_t* v669 = (const int8_t*) v668;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v670 = *(const int8_t *)(v669);
          vint16m1_t v671 = v620;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v672 = __riscv_vwmacc_vx_i16m1(v671, v667, v636, 16);
          v620 = v672;
          vint16m1_t v673 = v622;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v674 = __riscv_vwmacc_vx_i16m1(v673, v670, v638, 16);
          v622 = v674;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v675 = 339 + v630;
          size_t v676 = 467 + v630;
          const uint8_t* v677 = v32 + v675;
          const int8_t* v678 = (const int8_t*) v677;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v679 = *(const int8_t *)(v678);
          const uint8_t* v680 = v32 + v676;
          const int8_t* v681 = (const int8_t*) v680;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v682 = *(const int8_t *)(v681);
          vint16m1_t v683 = v624;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v684 = __riscv_vwmacc_vx_i16m1(v683, v679, v636, 16);
          v624 = v684;
          vint16m1_t v685 = v626;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v686 = __riscv_vwmacc_vx_i16m1(v685, v682, v638, 16);
          v626 = v686;
        }
        vint16m1_t v687 = v612;
        vint16m1_t v688 = v614;
        vint16m1_t v689 = v616;
        vint16m1_t v690 = v618;
        vint16m1_t v691 = v620;
        vint16m1_t v692 = v622;
        vint16m1_t v693 = v624;
        vint16m1_t v694 = v626;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v695 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v696 = __riscv_vle16_v_i16m1(v695, 16);
        int16_t* v697 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v698 = __riscv_vle16_v_i16m1(v697, 16);
        vint32m2_t v699 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v700 = __riscv_vwmacc_vv_i32m2(v699, v696, v687, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v701 = __riscv_vwmacc_vv_i32m2(v700, v698, v688, 16);
        v47 = v701;
        vint32m2_t v702 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v703 = __riscv_vwmacc_vv_i32m2(v702, v696, v689, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v704 = __riscv_vwmacc_vv_i32m2(v703, v698, v690, 16);
        v49 = v704;
        vint32m2_t v705 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v706 = __riscv_vwmacc_vv_i32m2(v705, v696, v691, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v707 = __riscv_vwmacc_vv_i32m2(v706, v698, v692, 16);
        v51 = v707;
        vint32m2_t v708 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v709 = __riscv_vwmacc_vv_i32m2(v708, v696, v693, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v710 = __riscv_vwmacc_vv_i32m2(v709, v698, v694, 16);
        v53 = v710;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
        const uint8_t* v711 = v30 + 128;
        const uint8_t* v712 = (const uint8_t*) v711;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v713 = __riscv_vle8_v_u8mf2(v712, 16);
        const uint8_t* v714 = v30 + 192;
        const uint8_t* v715 = (const uint8_t*) v714;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v716 = __riscv_vle8_v_u8mf2(v715, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v717 = __riscv_vand_vx_u8mf2(v713, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v718 = __riscv_vsrl_vx_u8mf2(v713, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v719 = __riscv_vand_vx_u8mf2(v716, 0x30, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v720 = __riscv_vand_vx_u8mf2(v716, 0xC0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v721 = __riscv_vsrl_vx_u8mf2(v720, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v722 = __riscv_vor_vv_u8mf2(v719, v717, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v723 = __riscv_vor_vv_u8mf2(v721, v718, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v724 = __riscv_vzext_vf2_u16m1(v722, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v725 = __riscv_vreinterpret_v_u16m1_i16m1(v724);
        int16_t* v726 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v726, v725, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v727 = __riscv_vzext_vf2_u16m1(v723, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v728 = __riscv_vreinterpret_v_u16m1_i16m1(v727);
        int16_t* v729 = &v45[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v729, v728, 16);
        const uint8_t* v730 = v30 + 144;
        const uint8_t* v731 = (const uint8_t*) v730;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v732 = __riscv_vle8_v_u8mf2(v731, 16);
        const uint8_t* v733 = v30 + 208;
        const uint8_t* v734 = (const uint8_t*) v733;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v735 = __riscv_vle8_v_u8mf2(v734, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v736 = __riscv_vand_vx_u8mf2(v732, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v737 = __riscv_vsrl_vx_u8mf2(v732, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v738 = __riscv_vand_vx_u8mf2(v735, 0x30, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v739 = __riscv_vand_vx_u8mf2(v735, 0xC0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v740 = __riscv_vsrl_vx_u8mf2(v739, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v741 = __riscv_vor_vv_u8mf2(v738, v736, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v742 = __riscv_vor_vv_u8mf2(v740, v737, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v743 = __riscv_vzext_vf2_u16m1(v741, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v744 = __riscv_vreinterpret_v_u16m1_i16m1(v743);
        int16_t* v745 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v745, v744, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v746 = __riscv_vzext_vf2_u16m1(v742, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v747 = __riscv_vreinterpret_v_u16m1_i16m1(v746);
        int16_t* v748 = &v45[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v748, v747, 16);
        const uint8_t* v749 = v30 + 160;
        const uint8_t* v750 = (const uint8_t*) v749;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v751 = __riscv_vle8_v_u8mf2(v750, 16);
        const uint8_t* v752 = v30 + 224;
        const uint8_t* v753 = (const uint8_t*) v752;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v754 = __riscv_vle8_v_u8mf2(v753, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v755 = __riscv_vand_vx_u8mf2(v751, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v756 = __riscv_vsrl_vx_u8mf2(v751, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v757 = __riscv_vand_vx_u8mf2(v754, 0x30, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v758 = __riscv_vand_vx_u8mf2(v754, 0xC0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v759 = __riscv_vsrl_vx_u8mf2(v758, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v760 = __riscv_vor_vv_u8mf2(v757, v755, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v761 = __riscv_vor_vv_u8mf2(v759, v756, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v762 = __riscv_vzext_vf2_u16m1(v760, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v763 = __riscv_vreinterpret_v_u16m1_i16m1(v762);
        int16_t* v764 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v764, v763, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v765 = __riscv_vzext_vf2_u16m1(v761, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v766 = __riscv_vreinterpret_v_u16m1_i16m1(v765);
        int16_t* v767 = &v45[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v767, v766, 16);
        const uint8_t* v768 = v30 + 176;
        const uint8_t* v769 = (const uint8_t*) v768;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v770 = __riscv_vle8_v_u8mf2(v769, 16);
        const uint8_t* v771 = v30 + 240;
        const uint8_t* v772 = (const uint8_t*) v771;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v773 = __riscv_vle8_v_u8mf2(v772, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v774 = __riscv_vand_vx_u8mf2(v770, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v775 = __riscv_vsrl_vx_u8mf2(v770, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v776 = __riscv_vand_vx_u8mf2(v773, 0x30, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v777 = __riscv_vand_vx_u8mf2(v773, 0xC0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v778 = __riscv_vsrl_vx_u8mf2(v777, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v779 = __riscv_vor_vv_u8mf2(v776, v774, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v780 = __riscv_vor_vv_u8mf2(v778, v775, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v781 = __riscv_vzext_vf2_u16m1(v779, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v782 = __riscv_vreinterpret_v_u16m1_i16m1(v781);
        int16_t* v783 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v783, v782, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v784 = __riscv_vzext_vf2_u16m1(v780, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v785 = __riscv_vreinterpret_v_u16m1_i16m1(v784);
        int16_t* v786 = &v45[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v786, v785, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold
        int16_t* v787 = &v45[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v788 = __riscv_vle16_v_i16m1(v787, 16);
        const uint8_t* v789 = v32 + 1104;
        const int16_t* v790 = (const int16_t*) v789;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v791 = *(const int16_t *)(v790);
        const uint8_t* v792 = v32 + 1112;
        const int16_t* v793 = (const int16_t*) v792;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v794 = *(const int16_t *)(v793);
        int32_t v795 = v791 + v794;
        int32_t* v796 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v797 = __riscv_vle32_v_i32m2(v796, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v798 = __riscv_vwmacc_vx_i32m2(v797, v795, v788, 16);
        int32_t* v799 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v799, v798, 16);
        const uint8_t* v800 = v32 + 1106;
        const int16_t* v801 = (const int16_t*) v800;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v802 = *(const int16_t *)(v801);
        const uint8_t* v803 = v32 + 1114;
        const int16_t* v804 = (const int16_t*) v803;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v805 = *(const int16_t *)(v804);
        int32_t v806 = v802 + v805;
        int32_t* v807 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v808 = __riscv_vle32_v_i32m2(v807, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v809 = __riscv_vwmacc_vx_i32m2(v808, v806, v788, 16);
        int32_t* v810 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v810, v809, 16);
        const uint8_t* v811 = v32 + 1108;
        const int16_t* v812 = (const int16_t*) v811;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v813 = *(const int16_t *)(v812);
        const uint8_t* v814 = v32 + 1116;
        const int16_t* v815 = (const int16_t*) v814;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v816 = *(const int16_t *)(v815);
        int32_t v817 = v813 + v816;
        int32_t* v818 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v819 = __riscv_vle32_v_i32m2(v818, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v820 = __riscv_vwmacc_vx_i32m2(v819, v817, v788, 16);
        int32_t* v821 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v821, v820, 16);
        const uint8_t* v822 = v32 + 1110;
        const int16_t* v823 = (const int16_t*) v822;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v824 = *(const int16_t *)(v823);
        const uint8_t* v825 = v32 + 1118;
        const int16_t* v826 = (const int16_t*) v825;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v827 = *(const int16_t *)(v826);
        int32_t v828 = v824 + v827;
        int32_t* v829 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v830 = __riscv_vle32_v_i32m2(v829, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v831 = __riscv_vwmacc_vx_i32m2(v830, v828, v788, 16);
        int32_t* v832 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v832, v831, 16);
        int16_t* v833 = &v45[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v834 = __riscv_vle16_v_i16m1(v833, 16);
        const uint8_t* v835 = v32 + 1120;
        const int16_t* v836 = (const int16_t*) v835;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v837 = *(const int16_t *)(v836);
        const uint8_t* v838 = v32 + 1128;
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
        const uint8_t* v846 = v32 + 1122;
        const int16_t* v847 = (const int16_t*) v846;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v848 = *(const int16_t *)(v847);
        const uint8_t* v849 = v32 + 1130;
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
        const uint8_t* v857 = v32 + 1124;
        const int16_t* v858 = (const int16_t*) v857;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v859 = *(const int16_t *)(v858);
        const uint8_t* v860 = v32 + 1132;
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
        const uint8_t* v868 = v32 + 1126;
        const int16_t* v869 = (const int16_t*) v868;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v870 = *(const int16_t *)(v869);
        const uint8_t* v871 = v32 + 1134;
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
        int16_t* v879 = &v45[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v880 = __riscv_vle16_v_i16m1(v879, 16);
        const uint8_t* v881 = v32 + 1136;
        const int16_t* v882 = (const int16_t*) v881;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v883 = *(const int16_t *)(v882);
        const uint8_t* v884 = v32 + 1144;
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
        const uint8_t* v892 = v32 + 1138;
        const int16_t* v893 = (const int16_t*) v892;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v894 = *(const int16_t *)(v893);
        const uint8_t* v895 = v32 + 1146;
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
        const uint8_t* v903 = v32 + 1140;
        const int16_t* v904 = (const int16_t*) v903;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v905 = *(const int16_t *)(v904);
        const uint8_t* v906 = v32 + 1148;
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
        const uint8_t* v914 = v32 + 1142;
        const int16_t* v915 = (const int16_t*) v914;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v916 = *(const int16_t *)(v915);
        const uint8_t* v917 = v32 + 1150;
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
        int16_t* v925 = &v45[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v926 = __riscv_vle16_v_i16m1(v925, 16);
        const uint8_t* v927 = v32 + 1152;
        const int16_t* v928 = (const int16_t*) v927;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v929 = *(const int16_t *)(v928);
        const uint8_t* v930 = v32 + 1160;
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
        const uint8_t* v938 = v32 + 1154;
        const int16_t* v939 = (const int16_t*) v938;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v940 = *(const int16_t *)(v939);
        const uint8_t* v941 = v32 + 1162;
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
        const uint8_t* v949 = v32 + 1156;
        const int16_t* v950 = (const int16_t*) v949;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v951 = *(const int16_t *)(v950);
        const uint8_t* v952 = v32 + 1164;
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
        const uint8_t* v960 = v32 + 1158;
        const int16_t* v961 = (const int16_t*) v960;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v962 = *(const int16_t *)(v961);
        const uint8_t* v963 = v32 + 1166;
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
        vint16m1_t v971;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v972 = __riscv_vmv_v_x_i16m1(0, 16);
        v971 = v972;
        vint16m1_t v973;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v974 = __riscv_vmv_v_x_i16m1(0, 16);
        v973 = v974;
        vint16m1_t v975;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v976 = __riscv_vmv_v_x_i16m1(0, 16);
        v975 = v976;
        vint16m1_t v977;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v978 = __riscv_vmv_v_x_i16m1(0, 16);
        v977 = v978;
        vint16m1_t v979;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v980 = __riscv_vmv_v_x_i16m1(0, 16);
        v979 = v980;
        vint16m1_t v981;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v982 = __riscv_vmv_v_x_i16m1(0, 16);
        v981 = v982;
        vint16m1_t v983;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v984 = __riscv_vmv_v_x_i16m1(0, 16);
        v983 = v984;
        vint16m1_t v985;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v986 = __riscv_vmv_v_x_i16m1(0, 16);
        v985 = v986;
        for (size_t v987 = 0; v987 < 16; v987 += 1) {
          size_t v988 = v987 * 16;
          size_t v989 = v987 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v990 = 1280 + v988;
          const uint8_t* v991 = v30 + v990;
          const uint8_t* v992 = (const uint8_t*) v991;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v993 = __riscv_vle8_v_u8mf2(v992, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v994 = __riscv_vand_vx_u8mf2(v993, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v995 = __riscv_vreinterpret_v_u8mf2_i8mf2(v994);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v996 = __riscv_vsrl_vx_u8mf2(v993, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v997 = __riscv_vreinterpret_v_u8mf2_i8mf2(v996);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v998 = 528 + v989;
          size_t v999 = 656 + v989;
          const uint8_t* v1000 = v32 + v998;
          const int8_t* v1001 = (const int8_t*) v1000;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1002 = *(const int8_t *)(v1001);
          const uint8_t* v1003 = v32 + v999;
          const int8_t* v1004 = (const int8_t*) v1003;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1005 = *(const int8_t *)(v1004);
          vint16m1_t v1006 = v971;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1007 = __riscv_vwmacc_vx_i16m1(v1006, v1002, v995, 16);
          v971 = v1007;
          vint16m1_t v1008 = v973;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1009 = __riscv_vwmacc_vx_i16m1(v1008, v1005, v997, 16);
          v973 = v1009;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1010 = 529 + v989;
          size_t v1011 = 657 + v989;
          const uint8_t* v1012 = v32 + v1010;
          const int8_t* v1013 = (const int8_t*) v1012;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1014 = *(const int8_t *)(v1013);
          const uint8_t* v1015 = v32 + v1011;
          const int8_t* v1016 = (const int8_t*) v1015;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1017 = *(const int8_t *)(v1016);
          vint16m1_t v1018 = v975;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1019 = __riscv_vwmacc_vx_i16m1(v1018, v1014, v995, 16);
          v975 = v1019;
          vint16m1_t v1020 = v977;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1021 = __riscv_vwmacc_vx_i16m1(v1020, v1017, v997, 16);
          v977 = v1021;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1022 = 530 + v989;
          size_t v1023 = 658 + v989;
          const uint8_t* v1024 = v32 + v1022;
          const int8_t* v1025 = (const int8_t*) v1024;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1026 = *(const int8_t *)(v1025);
          const uint8_t* v1027 = v32 + v1023;
          const int8_t* v1028 = (const int8_t*) v1027;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1029 = *(const int8_t *)(v1028);
          vint16m1_t v1030 = v979;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1031 = __riscv_vwmacc_vx_i16m1(v1030, v1026, v995, 16);
          v979 = v1031;
          vint16m1_t v1032 = v981;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1033 = __riscv_vwmacc_vx_i16m1(v1032, v1029, v997, 16);
          v981 = v1033;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1034 = 531 + v989;
          size_t v1035 = 659 + v989;
          const uint8_t* v1036 = v32 + v1034;
          const int8_t* v1037 = (const int8_t*) v1036;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1038 = *(const int8_t *)(v1037);
          const uint8_t* v1039 = v32 + v1035;
          const int8_t* v1040 = (const int8_t*) v1039;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1041 = *(const int8_t *)(v1040);
          vint16m1_t v1042 = v983;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1043 = __riscv_vwmacc_vx_i16m1(v1042, v1038, v995, 16);
          v983 = v1043;
          vint16m1_t v1044 = v985;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1045 = __riscv_vwmacc_vx_i16m1(v1044, v1041, v997, 16);
          v985 = v1045;
        }
        vint16m1_t v1046 = v971;
        vint16m1_t v1047 = v973;
        vint16m1_t v1048 = v975;
        vint16m1_t v1049 = v977;
        vint16m1_t v1050 = v979;
        vint16m1_t v1051 = v981;
        vint16m1_t v1052 = v983;
        vint16m1_t v1053 = v985;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v1054 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1055 = __riscv_vle16_v_i16m1(v1054, 16);
        int16_t* v1056 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1057 = __riscv_vle16_v_i16m1(v1056, 16);
        vint32m2_t v1058 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1059 = __riscv_vwmacc_vv_i32m2(v1058, v1055, v1046, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1060 = __riscv_vwmacc_vv_i32m2(v1059, v1057, v1047, 16);
        v47 = v1060;
        vint32m2_t v1061 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1062 = __riscv_vwmacc_vv_i32m2(v1061, v1055, v1048, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1063 = __riscv_vwmacc_vv_i32m2(v1062, v1057, v1049, 16);
        v49 = v1063;
        vint32m2_t v1064 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1065 = __riscv_vwmacc_vv_i32m2(v1064, v1055, v1050, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1066 = __riscv_vwmacc_vv_i32m2(v1065, v1057, v1051, 16);
        v51 = v1066;
        vint32m2_t v1067 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1068 = __riscv_vwmacc_vv_i32m2(v1067, v1055, v1052, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1069 = __riscv_vwmacc_vv_i32m2(v1068, v1057, v1053, 16);
        v53 = v1069;
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
        vint16m1_t v1080;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1081 = __riscv_vmv_v_x_i16m1(0, 16);
        v1080 = v1081;
        vint16m1_t v1082;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1083 = __riscv_vmv_v_x_i16m1(0, 16);
        v1082 = v1083;
        vint16m1_t v1084;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1085 = __riscv_vmv_v_x_i16m1(0, 16);
        v1084 = v1085;
        for (size_t v1086 = 0; v1086 < 16; v1086 += 1) {
          size_t v1087 = v1086 * 16;
          size_t v1088 = v1086 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v1089 = 1536 + v1087;
          const uint8_t* v1090 = v30 + v1089;
          const uint8_t* v1091 = (const uint8_t*) v1090;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1092 = __riscv_vle8_v_u8mf2(v1091, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1093 = __riscv_vand_vx_u8mf2(v1092, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1094 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1093);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1095 = __riscv_vsrl_vx_u8mf2(v1092, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1096 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1095);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1097 = 592 + v1088;
          size_t v1098 = 720 + v1088;
          const uint8_t* v1099 = v32 + v1097;
          const int8_t* v1100 = (const int8_t*) v1099;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1101 = *(const int8_t *)(v1100);
          const uint8_t* v1102 = v32 + v1098;
          const int8_t* v1103 = (const int8_t*) v1102;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1104 = *(const int8_t *)(v1103);
          vint16m1_t v1105 = v1070;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1106 = __riscv_vwmacc_vx_i16m1(v1105, v1101, v1094, 16);
          v1070 = v1106;
          vint16m1_t v1107 = v1072;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1108 = __riscv_vwmacc_vx_i16m1(v1107, v1104, v1096, 16);
          v1072 = v1108;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1109 = 593 + v1088;
          size_t v1110 = 721 + v1088;
          const uint8_t* v1111 = v32 + v1109;
          const int8_t* v1112 = (const int8_t*) v1111;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1113 = *(const int8_t *)(v1112);
          const uint8_t* v1114 = v32 + v1110;
          const int8_t* v1115 = (const int8_t*) v1114;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1116 = *(const int8_t *)(v1115);
          vint16m1_t v1117 = v1074;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1118 = __riscv_vwmacc_vx_i16m1(v1117, v1113, v1094, 16);
          v1074 = v1118;
          vint16m1_t v1119 = v1076;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1120 = __riscv_vwmacc_vx_i16m1(v1119, v1116, v1096, 16);
          v1076 = v1120;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1121 = 594 + v1088;
          size_t v1122 = 722 + v1088;
          const uint8_t* v1123 = v32 + v1121;
          const int8_t* v1124 = (const int8_t*) v1123;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1125 = *(const int8_t *)(v1124);
          const uint8_t* v1126 = v32 + v1122;
          const int8_t* v1127 = (const int8_t*) v1126;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1128 = *(const int8_t *)(v1127);
          vint16m1_t v1129 = v1078;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1130 = __riscv_vwmacc_vx_i16m1(v1129, v1125, v1094, 16);
          v1078 = v1130;
          vint16m1_t v1131 = v1080;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1132 = __riscv_vwmacc_vx_i16m1(v1131, v1128, v1096, 16);
          v1080 = v1132;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1133 = 595 + v1088;
          size_t v1134 = 723 + v1088;
          const uint8_t* v1135 = v32 + v1133;
          const int8_t* v1136 = (const int8_t*) v1135;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1137 = *(const int8_t *)(v1136);
          const uint8_t* v1138 = v32 + v1134;
          const int8_t* v1139 = (const int8_t*) v1138;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1140 = *(const int8_t *)(v1139);
          vint16m1_t v1141 = v1082;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1142 = __riscv_vwmacc_vx_i16m1(v1141, v1137, v1094, 16);
          v1082 = v1142;
          vint16m1_t v1143 = v1084;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1144 = __riscv_vwmacc_vx_i16m1(v1143, v1140, v1096, 16);
          v1084 = v1144;
        }
        vint16m1_t v1145 = v1070;
        vint16m1_t v1146 = v1072;
        vint16m1_t v1147 = v1074;
        vint16m1_t v1148 = v1076;
        vint16m1_t v1149 = v1078;
        vint16m1_t v1150 = v1080;
        vint16m1_t v1151 = v1082;
        vint16m1_t v1152 = v1084;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v1153 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1154 = __riscv_vle16_v_i16m1(v1153, 16);
        int16_t* v1155 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1156 = __riscv_vle16_v_i16m1(v1155, 16);
        vint32m2_t v1157 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1158 = __riscv_vwmacc_vv_i32m2(v1157, v1154, v1145, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1159 = __riscv_vwmacc_vv_i32m2(v1158, v1156, v1146, 16);
        v47 = v1159;
        vint32m2_t v1160 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1161 = __riscv_vwmacc_vv_i32m2(v1160, v1154, v1147, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1162 = __riscv_vwmacc_vv_i32m2(v1161, v1156, v1148, 16);
        v49 = v1162;
        vint32m2_t v1163 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1164 = __riscv_vwmacc_vv_i32m2(v1163, v1154, v1149, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1165 = __riscv_vwmacc_vv_i32m2(v1164, v1156, v1150, 16);
        v51 = v1165;
        vint32m2_t v1166 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1167 = __riscv_vwmacc_vv_i32m2(v1166, v1154, v1151, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1168 = __riscv_vwmacc_vv_i32m2(v1167, v1156, v1152, 16);
        v53 = v1168;
        vint16m1_t v1169;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1170 = __riscv_vmv_v_x_i16m1(0, 16);
        v1169 = v1170;
        vint16m1_t v1171;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1172 = __riscv_vmv_v_x_i16m1(0, 16);
        v1171 = v1172;
        vint16m1_t v1173;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1174 = __riscv_vmv_v_x_i16m1(0, 16);
        v1173 = v1174;
        vint16m1_t v1175;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1176 = __riscv_vmv_v_x_i16m1(0, 16);
        v1175 = v1176;
        vint16m1_t v1177;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1178 = __riscv_vmv_v_x_i16m1(0, 16);
        v1177 = v1178;
        vint16m1_t v1179;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1180 = __riscv_vmv_v_x_i16m1(0, 16);
        v1179 = v1180;
        vint16m1_t v1181;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1182 = __riscv_vmv_v_x_i16m1(0, 16);
        v1181 = v1182;
        vint16m1_t v1183;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1184 = __riscv_vmv_v_x_i16m1(0, 16);
        v1183 = v1184;
        for (size_t v1185 = 0; v1185 < 16; v1185 += 1) {
          size_t v1186 = v1185 * 16;
          size_t v1187 = v1185 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v1188 = 1792 + v1186;
          const uint8_t* v1189 = v30 + v1188;
          const uint8_t* v1190 = (const uint8_t*) v1189;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1191 = __riscv_vle8_v_u8mf2(v1190, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1192 = __riscv_vand_vx_u8mf2(v1191, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1193 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1192);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1194 = __riscv_vsrl_vx_u8mf2(v1191, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1195 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1194);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1196 = 784 + v1187;
          size_t v1197 = 912 + v1187;
          const uint8_t* v1198 = v32 + v1196;
          const int8_t* v1199 = (const int8_t*) v1198;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1200 = *(const int8_t *)(v1199);
          const uint8_t* v1201 = v32 + v1197;
          const int8_t* v1202 = (const int8_t*) v1201;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1203 = *(const int8_t *)(v1202);
          vint16m1_t v1204 = v1169;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1205 = __riscv_vwmacc_vx_i16m1(v1204, v1200, v1193, 16);
          v1169 = v1205;
          vint16m1_t v1206 = v1171;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1207 = __riscv_vwmacc_vx_i16m1(v1206, v1203, v1195, 16);
          v1171 = v1207;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1208 = 785 + v1187;
          size_t v1209 = 913 + v1187;
          const uint8_t* v1210 = v32 + v1208;
          const int8_t* v1211 = (const int8_t*) v1210;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1212 = *(const int8_t *)(v1211);
          const uint8_t* v1213 = v32 + v1209;
          const int8_t* v1214 = (const int8_t*) v1213;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1215 = *(const int8_t *)(v1214);
          vint16m1_t v1216 = v1173;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1217 = __riscv_vwmacc_vx_i16m1(v1216, v1212, v1193, 16);
          v1173 = v1217;
          vint16m1_t v1218 = v1175;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1219 = __riscv_vwmacc_vx_i16m1(v1218, v1215, v1195, 16);
          v1175 = v1219;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1220 = 786 + v1187;
          size_t v1221 = 914 + v1187;
          const uint8_t* v1222 = v32 + v1220;
          const int8_t* v1223 = (const int8_t*) v1222;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1224 = *(const int8_t *)(v1223);
          const uint8_t* v1225 = v32 + v1221;
          const int8_t* v1226 = (const int8_t*) v1225;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1227 = *(const int8_t *)(v1226);
          vint16m1_t v1228 = v1177;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1229 = __riscv_vwmacc_vx_i16m1(v1228, v1224, v1193, 16);
          v1177 = v1229;
          vint16m1_t v1230 = v1179;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1231 = __riscv_vwmacc_vx_i16m1(v1230, v1227, v1195, 16);
          v1179 = v1231;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1232 = 787 + v1187;
          size_t v1233 = 915 + v1187;
          const uint8_t* v1234 = v32 + v1232;
          const int8_t* v1235 = (const int8_t*) v1234;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1236 = *(const int8_t *)(v1235);
          const uint8_t* v1237 = v32 + v1233;
          const int8_t* v1238 = (const int8_t*) v1237;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1239 = *(const int8_t *)(v1238);
          vint16m1_t v1240 = v1181;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1241 = __riscv_vwmacc_vx_i16m1(v1240, v1236, v1193, 16);
          v1181 = v1241;
          vint16m1_t v1242 = v1183;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1243 = __riscv_vwmacc_vx_i16m1(v1242, v1239, v1195, 16);
          v1183 = v1243;
        }
        vint16m1_t v1244 = v1169;
        vint16m1_t v1245 = v1171;
        vint16m1_t v1246 = v1173;
        vint16m1_t v1247 = v1175;
        vint16m1_t v1248 = v1177;
        vint16m1_t v1249 = v1179;
        vint16m1_t v1250 = v1181;
        vint16m1_t v1251 = v1183;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v1252 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1253 = __riscv_vle16_v_i16m1(v1252, 16);
        int16_t* v1254 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1255 = __riscv_vle16_v_i16m1(v1254, 16);
        vint32m2_t v1256 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1257 = __riscv_vwmacc_vv_i32m2(v1256, v1253, v1244, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1258 = __riscv_vwmacc_vv_i32m2(v1257, v1255, v1245, 16);
        v47 = v1258;
        vint32m2_t v1259 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1260 = __riscv_vwmacc_vv_i32m2(v1259, v1253, v1246, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1261 = __riscv_vwmacc_vv_i32m2(v1260, v1255, v1247, 16);
        v49 = v1261;
        vint32m2_t v1262 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1263 = __riscv_vwmacc_vv_i32m2(v1262, v1253, v1248, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1264 = __riscv_vwmacc_vv_i32m2(v1263, v1255, v1249, 16);
        v51 = v1264;
        vint32m2_t v1265 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1266 = __riscv_vwmacc_vv_i32m2(v1265, v1253, v1250, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1267 = __riscv_vwmacc_vv_i32m2(v1266, v1255, v1251, 16);
        v53 = v1267;
        vint16m1_t v1268;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1269 = __riscv_vmv_v_x_i16m1(0, 16);
        v1268 = v1269;
        vint16m1_t v1270;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1271 = __riscv_vmv_v_x_i16m1(0, 16);
        v1270 = v1271;
        vint16m1_t v1272;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1273 = __riscv_vmv_v_x_i16m1(0, 16);
        v1272 = v1273;
        vint16m1_t v1274;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1275 = __riscv_vmv_v_x_i16m1(0, 16);
        v1274 = v1275;
        vint16m1_t v1276;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1277 = __riscv_vmv_v_x_i16m1(0, 16);
        v1276 = v1277;
        vint16m1_t v1278;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1279 = __riscv_vmv_v_x_i16m1(0, 16);
        v1278 = v1279;
        vint16m1_t v1280;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1281 = __riscv_vmv_v_x_i16m1(0, 16);
        v1280 = v1281;
        vint16m1_t v1282;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1283 = __riscv_vmv_v_x_i16m1(0, 16);
        v1282 = v1283;
        for (size_t v1284 = 0; v1284 < 16; v1284 += 1) {
          size_t v1285 = v1284 * 16;
          size_t v1286 = v1284 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
          size_t v1287 = 2048 + v1285;
          const uint8_t* v1288 = v30 + v1287;
          const uint8_t* v1289 = (const uint8_t*) v1288;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1290 = __riscv_vle8_v_u8mf2(v1289, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1291 = __riscv_vand_vx_u8mf2(v1290, 0x0F, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1292 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1291);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1293 = __riscv_vsrl_vx_u8mf2(v1290, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1294 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1293);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1295 = 848 + v1286;
          size_t v1296 = 976 + v1286;
          const uint8_t* v1297 = v32 + v1295;
          const int8_t* v1298 = (const int8_t*) v1297;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1299 = *(const int8_t *)(v1298);
          const uint8_t* v1300 = v32 + v1296;
          const int8_t* v1301 = (const int8_t*) v1300;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1302 = *(const int8_t *)(v1301);
          vint16m1_t v1303 = v1268;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1304 = __riscv_vwmacc_vx_i16m1(v1303, v1299, v1292, 16);
          v1268 = v1304;
          vint16m1_t v1305 = v1270;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1306 = __riscv_vwmacc_vx_i16m1(v1305, v1302, v1294, 16);
          v1270 = v1306;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1307 = 849 + v1286;
          size_t v1308 = 977 + v1286;
          const uint8_t* v1309 = v32 + v1307;
          const int8_t* v1310 = (const int8_t*) v1309;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1311 = *(const int8_t *)(v1310);
          const uint8_t* v1312 = v32 + v1308;
          const int8_t* v1313 = (const int8_t*) v1312;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1314 = *(const int8_t *)(v1313);
          vint16m1_t v1315 = v1272;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1316 = __riscv_vwmacc_vx_i16m1(v1315, v1311, v1292, 16);
          v1272 = v1316;
          vint16m1_t v1317 = v1274;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1318 = __riscv_vwmacc_vx_i16m1(v1317, v1314, v1294, 16);
          v1274 = v1318;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1319 = 850 + v1286;
          size_t v1320 = 978 + v1286;
          const uint8_t* v1321 = v32 + v1319;
          const int8_t* v1322 = (const int8_t*) v1321;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1323 = *(const int8_t *)(v1322);
          const uint8_t* v1324 = v32 + v1320;
          const int8_t* v1325 = (const int8_t*) v1324;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1326 = *(const int8_t *)(v1325);
          vint16m1_t v1327 = v1276;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1328 = __riscv_vwmacc_vx_i16m1(v1327, v1323, v1292, 16);
          v1276 = v1328;
          vint16m1_t v1329 = v1278;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1330 = __riscv_vwmacc_vx_i16m1(v1329, v1326, v1294, 16);
          v1278 = v1330;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1331 = 851 + v1286;
          size_t v1332 = 979 + v1286;
          const uint8_t* v1333 = v32 + v1331;
          const int8_t* v1334 = (const int8_t*) v1333;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1335 = *(const int8_t *)(v1334);
          const uint8_t* v1336 = v32 + v1332;
          const int8_t* v1337 = (const int8_t*) v1336;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1338 = *(const int8_t *)(v1337);
          vint16m1_t v1339 = v1280;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1340 = __riscv_vwmacc_vx_i16m1(v1339, v1335, v1292, 16);
          v1280 = v1340;
          vint16m1_t v1341 = v1282;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1342 = __riscv_vwmacc_vx_i16m1(v1341, v1338, v1294, 16);
          v1282 = v1342;
        }
        vint16m1_t v1343 = v1268;
        vint16m1_t v1344 = v1270;
        vint16m1_t v1345 = v1272;
        vint16m1_t v1346 = v1274;
        vint16m1_t v1347 = v1276;
        vint16m1_t v1348 = v1278;
        vint16m1_t v1349 = v1280;
        vint16m1_t v1350 = v1282;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v1351 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1352 = __riscv_vle16_v_i16m1(v1351, 16);
        int16_t* v1353 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1354 = __riscv_vle16_v_i16m1(v1353, 16);
        vint32m2_t v1355 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1356 = __riscv_vwmacc_vv_i32m2(v1355, v1352, v1343, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1357 = __riscv_vwmacc_vv_i32m2(v1356, v1354, v1344, 16);
        v47 = v1357;
        vint32m2_t v1358 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1359 = __riscv_vwmacc_vv_i32m2(v1358, v1352, v1345, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1360 = __riscv_vwmacc_vv_i32m2(v1359, v1354, v1346, 16);
        v49 = v1360;
        vint32m2_t v1361 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1362 = __riscv_vwmacc_vv_i32m2(v1361, v1352, v1347, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1363 = __riscv_vwmacc_vv_i32m2(v1362, v1354, v1348, 16);
        v51 = v1363;
        vint32m2_t v1364 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1365 = __riscv_vwmacc_vv_i32m2(v1364, v1352, v1349, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1366 = __riscv_vwmacc_vv_i32m2(v1365, v1354, v1350, 16);
        v53 = v1366;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const uint8_t* v1367 = v30 + 32;
        const _Float16* v1368 = (const _Float16*) v1367;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v1369 = __riscv_vle16_v_f16m1(v1368, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v1370 = __riscv_vfwcvt_f_f_v_f32m2(v1369, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const _Float16* v1371 = (const _Float16*) v30;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v1372 = __riscv_vle16_v_f16m1(v1371, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v1373 = __riscv_vfwcvt_f_f_v_f32m2(v1372, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1374 = __riscv_vfmul_vf_f32m2(v1373, v34, 16);
        vint32m2_t v1375 = v47;
        vfloat32m2_t v1376 = v20;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1377 = __riscv_vfcvt_f_x_v_f32m2(v1375, 16);
        vfloat32m2_t v1378 = __riscv_vfmacc_vv_f32m2(v1376, v1377, v1374, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1379 = __riscv_vfmul_vf_f32m2(v1370, v34, 16);
        int32_t* v1380 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1381 = __riscv_vle32_v_i32m2(v1380, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1382 = __riscv_vfcvt_f_x_v_f32m2(v1381, 16);
        vfloat32m2_t v1383 = __riscv_vfnmsac_vv_f32m2(v1378, v1379, v1382, 16);
        v20 = v1383;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1384 = __riscv_vfmul_vf_f32m2(v1373, v37, 16);
        vint32m2_t v1385 = v49;
        vfloat32m2_t v1386 = v22;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1387 = __riscv_vfcvt_f_x_v_f32m2(v1385, 16);
        vfloat32m2_t v1388 = __riscv_vfmacc_vv_f32m2(v1386, v1387, v1384, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1389 = __riscv_vfmul_vf_f32m2(v1370, v37, 16);
        int32_t* v1390 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1391 = __riscv_vle32_v_i32m2(v1390, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1392 = __riscv_vfcvt_f_x_v_f32m2(v1391, 16);
        vfloat32m2_t v1393 = __riscv_vfnmsac_vv_f32m2(v1388, v1389, v1392, 16);
        v22 = v1393;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1394 = __riscv_vfmul_vf_f32m2(v1373, v40, 16);
        vint32m2_t v1395 = v51;
        vfloat32m2_t v1396 = v24;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1397 = __riscv_vfcvt_f_x_v_f32m2(v1395, 16);
        vfloat32m2_t v1398 = __riscv_vfmacc_vv_f32m2(v1396, v1397, v1394, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1399 = __riscv_vfmul_vf_f32m2(v1370, v40, 16);
        int32_t* v1400 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1401 = __riscv_vle32_v_i32m2(v1400, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1402 = __riscv_vfcvt_f_x_v_f32m2(v1401, 16);
        vfloat32m2_t v1403 = __riscv_vfnmsac_vv_f32m2(v1398, v1399, v1402, 16);
        v24 = v1403;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1404 = __riscv_vfmul_vf_f32m2(v1373, v43, 16);
        vint32m2_t v1405 = v53;
        vfloat32m2_t v1406 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1407 = __riscv_vfcvt_f_x_v_f32m2(v1405, 16);
        vfloat32m2_t v1408 = __riscv_vfmacc_vv_f32m2(v1406, v1407, v1404, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1409 = __riscv_vfmul_vf_f32m2(v1370, v43, 16);
        int32_t* v1410 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1411 = __riscv_vle32_v_i32m2(v1410, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1412 = __riscv_vfcvt_f_x_v_f32m2(v1411, 16);
        vfloat32m2_t v1413 = __riscv_vfnmsac_vv_f32m2(v1408, v1409, v1412, 16);
        v26 = v1413;
      }
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1414 = v16 * 4;
      size_t v1415 = v1414 + 0;
      size_t v1416 = v1415 * v7;
      size_t v1417 = v12 * 16;
      size_t v1418 = v1416 + v1417;
      float* v1419 = v2 + v1418;
      vfloat32m2_t v1420 = v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1419, v1420, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1421 = v16 * 4;
      size_t v1422 = v1421 + 1;
      size_t v1423 = v1422 * v7;
      size_t v1424 = v12 * 16;
      size_t v1425 = v1423 + v1424;
      float* v1426 = v2 + v1425;
      vfloat32m2_t v1427 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1426, v1427, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1428 = v16 * 4;
      size_t v1429 = v1428 + 2;
      size_t v1430 = v1429 * v7;
      size_t v1431 = v12 * 16;
      size_t v1432 = v1430 + v1431;
      float* v1433 = v2 + v1432;
      vfloat32m2_t v1434 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1433, v1434, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1435 = v16 * 4;
      size_t v1436 = v1435 + 3;
      size_t v1437 = v1436 * v7;
      size_t v1438 = v12 * 16;
      size_t v1439 = v1437 + v1438;
      float* v1440 = v2 + v1439;
      vfloat32m2_t v1441 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1440, v1441, 16);
    }
  }
  return;
}


