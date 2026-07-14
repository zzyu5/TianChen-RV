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
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v315 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v316 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v317 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v318 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v319 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v320 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v321 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v322 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v323 = v30 + 256;
        const uint8_t* v324 = (const uint8_t*) v323;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v325 = __riscv_vle8_v_u8mf2(v324, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v326 = __riscv_vand_vx_u8mf2(v325, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v327 = __riscv_vreinterpret_v_u8mf2_i8mf2(v326);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v328 = __riscv_vsrl_vx_u8mf2(v325, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v329 = __riscv_vreinterpret_v_u8mf2_i8mf2(v328);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v330 = v32 + 16;
        const int8_t* v331 = (const int8_t*) v330;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v332 = *(const int8_t *)(v331);
        const uint8_t* v333 = v32 + 144;
        const int8_t* v334 = (const int8_t*) v333;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v335 = *(const int8_t *)(v334);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v336 = __riscv_vwmacc_vx_i16m1(v315, v332, v327, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v337 = __riscv_vwmacc_vx_i16m1(v316, v335, v329, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v338 = v32 + 17;
        const int8_t* v339 = (const int8_t*) v338;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v340 = *(const int8_t *)(v339);
        const uint8_t* v341 = v32 + 145;
        const int8_t* v342 = (const int8_t*) v341;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v343 = *(const int8_t *)(v342);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v344 = __riscv_vwmacc_vx_i16m1(v317, v340, v327, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v345 = __riscv_vwmacc_vx_i16m1(v318, v343, v329, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v346 = v32 + 18;
        const int8_t* v347 = (const int8_t*) v346;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v348 = *(const int8_t *)(v347);
        const uint8_t* v349 = v32 + 146;
        const int8_t* v350 = (const int8_t*) v349;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v351 = *(const int8_t *)(v350);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v352 = __riscv_vwmacc_vx_i16m1(v319, v348, v327, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v353 = __riscv_vwmacc_vx_i16m1(v320, v351, v329, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v354 = v32 + 19;
        const int8_t* v355 = (const int8_t*) v354;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v356 = *(const int8_t *)(v355);
        const uint8_t* v357 = v32 + 147;
        const int8_t* v358 = (const int8_t*) v357;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v359 = *(const int8_t *)(v358);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v360 = __riscv_vwmacc_vx_i16m1(v321, v356, v327, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v361 = __riscv_vwmacc_vx_i16m1(v322, v359, v329, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v362 = v30 + 272;
        const uint8_t* v363 = (const uint8_t*) v362;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v364 = __riscv_vle8_v_u8mf2(v363, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v365 = __riscv_vand_vx_u8mf2(v364, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v366 = __riscv_vreinterpret_v_u8mf2_i8mf2(v365);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v367 = __riscv_vsrl_vx_u8mf2(v364, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v368 = __riscv_vreinterpret_v_u8mf2_i8mf2(v367);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v369 = v32 + 20;
        const int8_t* v370 = (const int8_t*) v369;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v371 = *(const int8_t *)(v370);
        const uint8_t* v372 = v32 + 148;
        const int8_t* v373 = (const int8_t*) v372;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v374 = *(const int8_t *)(v373);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v375 = __riscv_vwmacc_vx_i16m1(v336, v371, v366, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v376 = __riscv_vwmacc_vx_i16m1(v337, v374, v368, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v377 = v32 + 21;
        const int8_t* v378 = (const int8_t*) v377;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v379 = *(const int8_t *)(v378);
        const uint8_t* v380 = v32 + 149;
        const int8_t* v381 = (const int8_t*) v380;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v382 = *(const int8_t *)(v381);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v383 = __riscv_vwmacc_vx_i16m1(v344, v379, v366, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v384 = __riscv_vwmacc_vx_i16m1(v345, v382, v368, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v385 = v32 + 22;
        const int8_t* v386 = (const int8_t*) v385;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v387 = *(const int8_t *)(v386);
        const uint8_t* v388 = v32 + 150;
        const int8_t* v389 = (const int8_t*) v388;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v390 = *(const int8_t *)(v389);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v391 = __riscv_vwmacc_vx_i16m1(v352, v387, v366, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v392 = __riscv_vwmacc_vx_i16m1(v353, v390, v368, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v393 = v32 + 23;
        const int8_t* v394 = (const int8_t*) v393;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v395 = *(const int8_t *)(v394);
        const uint8_t* v396 = v32 + 151;
        const int8_t* v397 = (const int8_t*) v396;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v398 = *(const int8_t *)(v397);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v399 = __riscv_vwmacc_vx_i16m1(v360, v395, v366, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v400 = __riscv_vwmacc_vx_i16m1(v361, v398, v368, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v401 = v30 + 288;
        const uint8_t* v402 = (const uint8_t*) v401;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v403 = __riscv_vle8_v_u8mf2(v402, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v404 = __riscv_vand_vx_u8mf2(v403, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v405 = __riscv_vreinterpret_v_u8mf2_i8mf2(v404);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v406 = __riscv_vsrl_vx_u8mf2(v403, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v407 = __riscv_vreinterpret_v_u8mf2_i8mf2(v406);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v408 = v32 + 24;
        const int8_t* v409 = (const int8_t*) v408;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v410 = *(const int8_t *)(v409);
        const uint8_t* v411 = v32 + 152;
        const int8_t* v412 = (const int8_t*) v411;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v413 = *(const int8_t *)(v412);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v414 = __riscv_vwmacc_vx_i16m1(v375, v410, v405, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v415 = __riscv_vwmacc_vx_i16m1(v376, v413, v407, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v416 = v32 + 25;
        const int8_t* v417 = (const int8_t*) v416;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v418 = *(const int8_t *)(v417);
        const uint8_t* v419 = v32 + 153;
        const int8_t* v420 = (const int8_t*) v419;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v421 = *(const int8_t *)(v420);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v422 = __riscv_vwmacc_vx_i16m1(v383, v418, v405, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v423 = __riscv_vwmacc_vx_i16m1(v384, v421, v407, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v424 = v32 + 26;
        const int8_t* v425 = (const int8_t*) v424;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v426 = *(const int8_t *)(v425);
        const uint8_t* v427 = v32 + 154;
        const int8_t* v428 = (const int8_t*) v427;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v429 = *(const int8_t *)(v428);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v430 = __riscv_vwmacc_vx_i16m1(v391, v426, v405, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v431 = __riscv_vwmacc_vx_i16m1(v392, v429, v407, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v432 = v32 + 27;
        const int8_t* v433 = (const int8_t*) v432;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v434 = *(const int8_t *)(v433);
        const uint8_t* v435 = v32 + 155;
        const int8_t* v436 = (const int8_t*) v435;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v437 = *(const int8_t *)(v436);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v438 = __riscv_vwmacc_vx_i16m1(v399, v434, v405, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v439 = __riscv_vwmacc_vx_i16m1(v400, v437, v407, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v440 = v30 + 304;
        const uint8_t* v441 = (const uint8_t*) v440;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v442 = __riscv_vle8_v_u8mf2(v441, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v443 = __riscv_vand_vx_u8mf2(v442, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v444 = __riscv_vreinterpret_v_u8mf2_i8mf2(v443);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v445 = __riscv_vsrl_vx_u8mf2(v442, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v446 = __riscv_vreinterpret_v_u8mf2_i8mf2(v445);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v447 = v32 + 28;
        const int8_t* v448 = (const int8_t*) v447;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v449 = *(const int8_t *)(v448);
        const uint8_t* v450 = v32 + 156;
        const int8_t* v451 = (const int8_t*) v450;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v452 = *(const int8_t *)(v451);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v453 = __riscv_vwmacc_vx_i16m1(v414, v449, v444, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v454 = __riscv_vwmacc_vx_i16m1(v415, v452, v446, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v455 = v32 + 29;
        const int8_t* v456 = (const int8_t*) v455;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v457 = *(const int8_t *)(v456);
        const uint8_t* v458 = v32 + 157;
        const int8_t* v459 = (const int8_t*) v458;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v460 = *(const int8_t *)(v459);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v461 = __riscv_vwmacc_vx_i16m1(v422, v457, v444, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v462 = __riscv_vwmacc_vx_i16m1(v423, v460, v446, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v463 = v32 + 30;
        const int8_t* v464 = (const int8_t*) v463;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v465 = *(const int8_t *)(v464);
        const uint8_t* v466 = v32 + 158;
        const int8_t* v467 = (const int8_t*) v466;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v468 = *(const int8_t *)(v467);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v469 = __riscv_vwmacc_vx_i16m1(v430, v465, v444, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v470 = __riscv_vwmacc_vx_i16m1(v431, v468, v446, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v471 = v32 + 31;
        const int8_t* v472 = (const int8_t*) v471;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v473 = *(const int8_t *)(v472);
        const uint8_t* v474 = v32 + 159;
        const int8_t* v475 = (const int8_t*) v474;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v476 = *(const int8_t *)(v475);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v477 = __riscv_vwmacc_vx_i16m1(v438, v473, v444, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v478 = __riscv_vwmacc_vx_i16m1(v439, v476, v446, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v479 = v30 + 320;
        const uint8_t* v480 = (const uint8_t*) v479;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v481 = __riscv_vle8_v_u8mf2(v480, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v482 = __riscv_vand_vx_u8mf2(v481, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v483 = __riscv_vreinterpret_v_u8mf2_i8mf2(v482);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v484 = __riscv_vsrl_vx_u8mf2(v481, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v485 = __riscv_vreinterpret_v_u8mf2_i8mf2(v484);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v486 = v32 + 32;
        const int8_t* v487 = (const int8_t*) v486;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v488 = *(const int8_t *)(v487);
        const uint8_t* v489 = v32 + 160;
        const int8_t* v490 = (const int8_t*) v489;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v491 = *(const int8_t *)(v490);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v492 = __riscv_vwmacc_vx_i16m1(v453, v488, v483, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v493 = __riscv_vwmacc_vx_i16m1(v454, v491, v485, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v494 = v32 + 33;
        const int8_t* v495 = (const int8_t*) v494;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v496 = *(const int8_t *)(v495);
        const uint8_t* v497 = v32 + 161;
        const int8_t* v498 = (const int8_t*) v497;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v499 = *(const int8_t *)(v498);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v500 = __riscv_vwmacc_vx_i16m1(v461, v496, v483, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v501 = __riscv_vwmacc_vx_i16m1(v462, v499, v485, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v502 = v32 + 34;
        const int8_t* v503 = (const int8_t*) v502;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v504 = *(const int8_t *)(v503);
        const uint8_t* v505 = v32 + 162;
        const int8_t* v506 = (const int8_t*) v505;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v507 = *(const int8_t *)(v506);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v508 = __riscv_vwmacc_vx_i16m1(v469, v504, v483, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v509 = __riscv_vwmacc_vx_i16m1(v470, v507, v485, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v510 = v32 + 35;
        const int8_t* v511 = (const int8_t*) v510;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v512 = *(const int8_t *)(v511);
        const uint8_t* v513 = v32 + 163;
        const int8_t* v514 = (const int8_t*) v513;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v515 = *(const int8_t *)(v514);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v516 = __riscv_vwmacc_vx_i16m1(v477, v512, v483, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v517 = __riscv_vwmacc_vx_i16m1(v478, v515, v485, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v518 = v30 + 336;
        const uint8_t* v519 = (const uint8_t*) v518;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v520 = __riscv_vle8_v_u8mf2(v519, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v521 = __riscv_vand_vx_u8mf2(v520, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v522 = __riscv_vreinterpret_v_u8mf2_i8mf2(v521);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v523 = __riscv_vsrl_vx_u8mf2(v520, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v524 = __riscv_vreinterpret_v_u8mf2_i8mf2(v523);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v525 = v32 + 36;
        const int8_t* v526 = (const int8_t*) v525;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v527 = *(const int8_t *)(v526);
        const uint8_t* v528 = v32 + 164;
        const int8_t* v529 = (const int8_t*) v528;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v530 = *(const int8_t *)(v529);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v531 = __riscv_vwmacc_vx_i16m1(v492, v527, v522, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v532 = __riscv_vwmacc_vx_i16m1(v493, v530, v524, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v533 = v32 + 37;
        const int8_t* v534 = (const int8_t*) v533;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v535 = *(const int8_t *)(v534);
        const uint8_t* v536 = v32 + 165;
        const int8_t* v537 = (const int8_t*) v536;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v538 = *(const int8_t *)(v537);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v539 = __riscv_vwmacc_vx_i16m1(v500, v535, v522, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v540 = __riscv_vwmacc_vx_i16m1(v501, v538, v524, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v541 = v32 + 38;
        const int8_t* v542 = (const int8_t*) v541;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v543 = *(const int8_t *)(v542);
        const uint8_t* v544 = v32 + 166;
        const int8_t* v545 = (const int8_t*) v544;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v546 = *(const int8_t *)(v545);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v547 = __riscv_vwmacc_vx_i16m1(v508, v543, v522, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v548 = __riscv_vwmacc_vx_i16m1(v509, v546, v524, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v549 = v32 + 39;
        const int8_t* v550 = (const int8_t*) v549;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v551 = *(const int8_t *)(v550);
        const uint8_t* v552 = v32 + 167;
        const int8_t* v553 = (const int8_t*) v552;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v554 = *(const int8_t *)(v553);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v555 = __riscv_vwmacc_vx_i16m1(v516, v551, v522, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v556 = __riscv_vwmacc_vx_i16m1(v517, v554, v524, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v557 = v30 + 352;
        const uint8_t* v558 = (const uint8_t*) v557;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v559 = __riscv_vle8_v_u8mf2(v558, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v560 = __riscv_vand_vx_u8mf2(v559, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v561 = __riscv_vreinterpret_v_u8mf2_i8mf2(v560);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v562 = __riscv_vsrl_vx_u8mf2(v559, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v563 = __riscv_vreinterpret_v_u8mf2_i8mf2(v562);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v564 = v32 + 40;
        const int8_t* v565 = (const int8_t*) v564;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v566 = *(const int8_t *)(v565);
        const uint8_t* v567 = v32 + 168;
        const int8_t* v568 = (const int8_t*) v567;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v569 = *(const int8_t *)(v568);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v570 = __riscv_vwmacc_vx_i16m1(v531, v566, v561, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v571 = __riscv_vwmacc_vx_i16m1(v532, v569, v563, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v572 = v32 + 41;
        const int8_t* v573 = (const int8_t*) v572;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v574 = *(const int8_t *)(v573);
        const uint8_t* v575 = v32 + 169;
        const int8_t* v576 = (const int8_t*) v575;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v577 = *(const int8_t *)(v576);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v578 = __riscv_vwmacc_vx_i16m1(v539, v574, v561, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v579 = __riscv_vwmacc_vx_i16m1(v540, v577, v563, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v580 = v32 + 42;
        const int8_t* v581 = (const int8_t*) v580;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v582 = *(const int8_t *)(v581);
        const uint8_t* v583 = v32 + 170;
        const int8_t* v584 = (const int8_t*) v583;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v585 = *(const int8_t *)(v584);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v586 = __riscv_vwmacc_vx_i16m1(v547, v582, v561, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v587 = __riscv_vwmacc_vx_i16m1(v548, v585, v563, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v588 = v32 + 43;
        const int8_t* v589 = (const int8_t*) v588;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v590 = *(const int8_t *)(v589);
        const uint8_t* v591 = v32 + 171;
        const int8_t* v592 = (const int8_t*) v591;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v593 = *(const int8_t *)(v592);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v594 = __riscv_vwmacc_vx_i16m1(v555, v590, v561, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v595 = __riscv_vwmacc_vx_i16m1(v556, v593, v563, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v596 = v30 + 368;
        const uint8_t* v597 = (const uint8_t*) v596;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v598 = __riscv_vle8_v_u8mf2(v597, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v599 = __riscv_vand_vx_u8mf2(v598, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v600 = __riscv_vreinterpret_v_u8mf2_i8mf2(v599);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v601 = __riscv_vsrl_vx_u8mf2(v598, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v602 = __riscv_vreinterpret_v_u8mf2_i8mf2(v601);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v603 = v32 + 44;
        const int8_t* v604 = (const int8_t*) v603;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v605 = *(const int8_t *)(v604);
        const uint8_t* v606 = v32 + 172;
        const int8_t* v607 = (const int8_t*) v606;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v608 = *(const int8_t *)(v607);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v609 = __riscv_vwmacc_vx_i16m1(v570, v605, v600, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v610 = __riscv_vwmacc_vx_i16m1(v571, v608, v602, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v611 = v32 + 45;
        const int8_t* v612 = (const int8_t*) v611;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v613 = *(const int8_t *)(v612);
        const uint8_t* v614 = v32 + 173;
        const int8_t* v615 = (const int8_t*) v614;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v616 = *(const int8_t *)(v615);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v617 = __riscv_vwmacc_vx_i16m1(v578, v613, v600, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v618 = __riscv_vwmacc_vx_i16m1(v579, v616, v602, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v619 = v32 + 46;
        const int8_t* v620 = (const int8_t*) v619;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v621 = *(const int8_t *)(v620);
        const uint8_t* v622 = v32 + 174;
        const int8_t* v623 = (const int8_t*) v622;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v624 = *(const int8_t *)(v623);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v625 = __riscv_vwmacc_vx_i16m1(v586, v621, v600, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v626 = __riscv_vwmacc_vx_i16m1(v587, v624, v602, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v627 = v32 + 47;
        const int8_t* v628 = (const int8_t*) v627;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v629 = *(const int8_t *)(v628);
        const uint8_t* v630 = v32 + 175;
        const int8_t* v631 = (const int8_t*) v630;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v632 = *(const int8_t *)(v631);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v633 = __riscv_vwmacc_vx_i16m1(v594, v629, v600, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v634 = __riscv_vwmacc_vx_i16m1(v595, v632, v602, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v635 = v30 + 384;
        const uint8_t* v636 = (const uint8_t*) v635;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v637 = __riscv_vle8_v_u8mf2(v636, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v638 = __riscv_vand_vx_u8mf2(v637, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v639 = __riscv_vreinterpret_v_u8mf2_i8mf2(v638);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v640 = __riscv_vsrl_vx_u8mf2(v637, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v641 = __riscv_vreinterpret_v_u8mf2_i8mf2(v640);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v642 = v32 + 48;
        const int8_t* v643 = (const int8_t*) v642;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v644 = *(const int8_t *)(v643);
        const uint8_t* v645 = v32 + 176;
        const int8_t* v646 = (const int8_t*) v645;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v647 = *(const int8_t *)(v646);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v648 = __riscv_vwmacc_vx_i16m1(v609, v644, v639, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v649 = __riscv_vwmacc_vx_i16m1(v610, v647, v641, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v650 = v32 + 49;
        const int8_t* v651 = (const int8_t*) v650;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v652 = *(const int8_t *)(v651);
        const uint8_t* v653 = v32 + 177;
        const int8_t* v654 = (const int8_t*) v653;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v655 = *(const int8_t *)(v654);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v656 = __riscv_vwmacc_vx_i16m1(v617, v652, v639, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v657 = __riscv_vwmacc_vx_i16m1(v618, v655, v641, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v658 = v32 + 50;
        const int8_t* v659 = (const int8_t*) v658;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v660 = *(const int8_t *)(v659);
        const uint8_t* v661 = v32 + 178;
        const int8_t* v662 = (const int8_t*) v661;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v663 = *(const int8_t *)(v662);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v664 = __riscv_vwmacc_vx_i16m1(v625, v660, v639, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v665 = __riscv_vwmacc_vx_i16m1(v626, v663, v641, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v666 = v32 + 51;
        const int8_t* v667 = (const int8_t*) v666;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v668 = *(const int8_t *)(v667);
        const uint8_t* v669 = v32 + 179;
        const int8_t* v670 = (const int8_t*) v669;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v671 = *(const int8_t *)(v670);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v672 = __riscv_vwmacc_vx_i16m1(v633, v668, v639, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v673 = __riscv_vwmacc_vx_i16m1(v634, v671, v641, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v674 = v30 + 400;
        const uint8_t* v675 = (const uint8_t*) v674;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v676 = __riscv_vle8_v_u8mf2(v675, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v677 = __riscv_vand_vx_u8mf2(v676, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v678 = __riscv_vreinterpret_v_u8mf2_i8mf2(v677);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v679 = __riscv_vsrl_vx_u8mf2(v676, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v680 = __riscv_vreinterpret_v_u8mf2_i8mf2(v679);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v681 = v32 + 52;
        const int8_t* v682 = (const int8_t*) v681;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v683 = *(const int8_t *)(v682);
        const uint8_t* v684 = v32 + 180;
        const int8_t* v685 = (const int8_t*) v684;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v686 = *(const int8_t *)(v685);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v687 = __riscv_vwmacc_vx_i16m1(v648, v683, v678, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v688 = __riscv_vwmacc_vx_i16m1(v649, v686, v680, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v689 = v32 + 53;
        const int8_t* v690 = (const int8_t*) v689;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v691 = *(const int8_t *)(v690);
        const uint8_t* v692 = v32 + 181;
        const int8_t* v693 = (const int8_t*) v692;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v694 = *(const int8_t *)(v693);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v695 = __riscv_vwmacc_vx_i16m1(v656, v691, v678, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v696 = __riscv_vwmacc_vx_i16m1(v657, v694, v680, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v697 = v32 + 54;
        const int8_t* v698 = (const int8_t*) v697;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v699 = *(const int8_t *)(v698);
        const uint8_t* v700 = v32 + 182;
        const int8_t* v701 = (const int8_t*) v700;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v702 = *(const int8_t *)(v701);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v703 = __riscv_vwmacc_vx_i16m1(v664, v699, v678, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v704 = __riscv_vwmacc_vx_i16m1(v665, v702, v680, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v705 = v32 + 55;
        const int8_t* v706 = (const int8_t*) v705;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v707 = *(const int8_t *)(v706);
        const uint8_t* v708 = v32 + 183;
        const int8_t* v709 = (const int8_t*) v708;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v710 = *(const int8_t *)(v709);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v711 = __riscv_vwmacc_vx_i16m1(v672, v707, v678, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v712 = __riscv_vwmacc_vx_i16m1(v673, v710, v680, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v713 = v30 + 416;
        const uint8_t* v714 = (const uint8_t*) v713;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v715 = __riscv_vle8_v_u8mf2(v714, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v716 = __riscv_vand_vx_u8mf2(v715, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v717 = __riscv_vreinterpret_v_u8mf2_i8mf2(v716);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v718 = __riscv_vsrl_vx_u8mf2(v715, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v719 = __riscv_vreinterpret_v_u8mf2_i8mf2(v718);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v720 = v32 + 56;
        const int8_t* v721 = (const int8_t*) v720;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v722 = *(const int8_t *)(v721);
        const uint8_t* v723 = v32 + 184;
        const int8_t* v724 = (const int8_t*) v723;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v725 = *(const int8_t *)(v724);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v726 = __riscv_vwmacc_vx_i16m1(v687, v722, v717, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v727 = __riscv_vwmacc_vx_i16m1(v688, v725, v719, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v728 = v32 + 57;
        const int8_t* v729 = (const int8_t*) v728;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v730 = *(const int8_t *)(v729);
        const uint8_t* v731 = v32 + 185;
        const int8_t* v732 = (const int8_t*) v731;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v733 = *(const int8_t *)(v732);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v734 = __riscv_vwmacc_vx_i16m1(v695, v730, v717, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v735 = __riscv_vwmacc_vx_i16m1(v696, v733, v719, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v736 = v32 + 58;
        const int8_t* v737 = (const int8_t*) v736;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v738 = *(const int8_t *)(v737);
        const uint8_t* v739 = v32 + 186;
        const int8_t* v740 = (const int8_t*) v739;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v741 = *(const int8_t *)(v740);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v742 = __riscv_vwmacc_vx_i16m1(v703, v738, v717, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v743 = __riscv_vwmacc_vx_i16m1(v704, v741, v719, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v744 = v32 + 59;
        const int8_t* v745 = (const int8_t*) v744;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v746 = *(const int8_t *)(v745);
        const uint8_t* v747 = v32 + 187;
        const int8_t* v748 = (const int8_t*) v747;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v749 = *(const int8_t *)(v748);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v750 = __riscv_vwmacc_vx_i16m1(v711, v746, v717, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v751 = __riscv_vwmacc_vx_i16m1(v712, v749, v719, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v752 = v30 + 432;
        const uint8_t* v753 = (const uint8_t*) v752;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v754 = __riscv_vle8_v_u8mf2(v753, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v755 = __riscv_vand_vx_u8mf2(v754, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v756 = __riscv_vreinterpret_v_u8mf2_i8mf2(v755);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v757 = __riscv_vsrl_vx_u8mf2(v754, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v758 = __riscv_vreinterpret_v_u8mf2_i8mf2(v757);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v759 = v32 + 60;
        const int8_t* v760 = (const int8_t*) v759;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v761 = *(const int8_t *)(v760);
        const uint8_t* v762 = v32 + 188;
        const int8_t* v763 = (const int8_t*) v762;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v764 = *(const int8_t *)(v763);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v765 = __riscv_vwmacc_vx_i16m1(v726, v761, v756, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v766 = __riscv_vwmacc_vx_i16m1(v727, v764, v758, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v767 = v32 + 61;
        const int8_t* v768 = (const int8_t*) v767;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v769 = *(const int8_t *)(v768);
        const uint8_t* v770 = v32 + 189;
        const int8_t* v771 = (const int8_t*) v770;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v772 = *(const int8_t *)(v771);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v773 = __riscv_vwmacc_vx_i16m1(v734, v769, v756, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v774 = __riscv_vwmacc_vx_i16m1(v735, v772, v758, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v775 = v32 + 62;
        const int8_t* v776 = (const int8_t*) v775;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v777 = *(const int8_t *)(v776);
        const uint8_t* v778 = v32 + 190;
        const int8_t* v779 = (const int8_t*) v778;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v780 = *(const int8_t *)(v779);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v781 = __riscv_vwmacc_vx_i16m1(v742, v777, v756, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v782 = __riscv_vwmacc_vx_i16m1(v743, v780, v758, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v783 = v32 + 63;
        const int8_t* v784 = (const int8_t*) v783;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v785 = *(const int8_t *)(v784);
        const uint8_t* v786 = v32 + 191;
        const int8_t* v787 = (const int8_t*) v786;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v788 = *(const int8_t *)(v787);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v789 = __riscv_vwmacc_vx_i16m1(v750, v785, v756, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v790 = __riscv_vwmacc_vx_i16m1(v751, v788, v758, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v791 = v30 + 448;
        const uint8_t* v792 = (const uint8_t*) v791;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v793 = __riscv_vle8_v_u8mf2(v792, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v794 = __riscv_vand_vx_u8mf2(v793, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v795 = __riscv_vreinterpret_v_u8mf2_i8mf2(v794);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v796 = __riscv_vsrl_vx_u8mf2(v793, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v797 = __riscv_vreinterpret_v_u8mf2_i8mf2(v796);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v798 = v32 + 64;
        const int8_t* v799 = (const int8_t*) v798;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v800 = *(const int8_t *)(v799);
        const uint8_t* v801 = v32 + 192;
        const int8_t* v802 = (const int8_t*) v801;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v803 = *(const int8_t *)(v802);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v804 = __riscv_vwmacc_vx_i16m1(v765, v800, v795, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v805 = __riscv_vwmacc_vx_i16m1(v766, v803, v797, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v806 = v32 + 65;
        const int8_t* v807 = (const int8_t*) v806;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v808 = *(const int8_t *)(v807);
        const uint8_t* v809 = v32 + 193;
        const int8_t* v810 = (const int8_t*) v809;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v811 = *(const int8_t *)(v810);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v812 = __riscv_vwmacc_vx_i16m1(v773, v808, v795, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v813 = __riscv_vwmacc_vx_i16m1(v774, v811, v797, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v814 = v32 + 66;
        const int8_t* v815 = (const int8_t*) v814;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v816 = *(const int8_t *)(v815);
        const uint8_t* v817 = v32 + 194;
        const int8_t* v818 = (const int8_t*) v817;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v819 = *(const int8_t *)(v818);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v820 = __riscv_vwmacc_vx_i16m1(v781, v816, v795, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v821 = __riscv_vwmacc_vx_i16m1(v782, v819, v797, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v822 = v32 + 67;
        const int8_t* v823 = (const int8_t*) v822;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v824 = *(const int8_t *)(v823);
        const uint8_t* v825 = v32 + 195;
        const int8_t* v826 = (const int8_t*) v825;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v827 = *(const int8_t *)(v826);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v828 = __riscv_vwmacc_vx_i16m1(v789, v824, v795, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v829 = __riscv_vwmacc_vx_i16m1(v790, v827, v797, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v830 = v30 + 464;
        const uint8_t* v831 = (const uint8_t*) v830;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v832 = __riscv_vle8_v_u8mf2(v831, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v833 = __riscv_vand_vx_u8mf2(v832, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v834 = __riscv_vreinterpret_v_u8mf2_i8mf2(v833);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v835 = __riscv_vsrl_vx_u8mf2(v832, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v836 = __riscv_vreinterpret_v_u8mf2_i8mf2(v835);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v837 = v32 + 68;
        const int8_t* v838 = (const int8_t*) v837;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v839 = *(const int8_t *)(v838);
        const uint8_t* v840 = v32 + 196;
        const int8_t* v841 = (const int8_t*) v840;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v842 = *(const int8_t *)(v841);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v843 = __riscv_vwmacc_vx_i16m1(v804, v839, v834, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v844 = __riscv_vwmacc_vx_i16m1(v805, v842, v836, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v845 = v32 + 69;
        const int8_t* v846 = (const int8_t*) v845;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v847 = *(const int8_t *)(v846);
        const uint8_t* v848 = v32 + 197;
        const int8_t* v849 = (const int8_t*) v848;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v850 = *(const int8_t *)(v849);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v851 = __riscv_vwmacc_vx_i16m1(v812, v847, v834, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v852 = __riscv_vwmacc_vx_i16m1(v813, v850, v836, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v853 = v32 + 70;
        const int8_t* v854 = (const int8_t*) v853;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v855 = *(const int8_t *)(v854);
        const uint8_t* v856 = v32 + 198;
        const int8_t* v857 = (const int8_t*) v856;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v858 = *(const int8_t *)(v857);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v859 = __riscv_vwmacc_vx_i16m1(v820, v855, v834, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v860 = __riscv_vwmacc_vx_i16m1(v821, v858, v836, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v861 = v32 + 71;
        const int8_t* v862 = (const int8_t*) v861;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v863 = *(const int8_t *)(v862);
        const uint8_t* v864 = v32 + 199;
        const int8_t* v865 = (const int8_t*) v864;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v866 = *(const int8_t *)(v865);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v867 = __riscv_vwmacc_vx_i16m1(v828, v863, v834, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v868 = __riscv_vwmacc_vx_i16m1(v829, v866, v836, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v869 = v30 + 480;
        const uint8_t* v870 = (const uint8_t*) v869;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v871 = __riscv_vle8_v_u8mf2(v870, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v872 = __riscv_vand_vx_u8mf2(v871, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v873 = __riscv_vreinterpret_v_u8mf2_i8mf2(v872);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v874 = __riscv_vsrl_vx_u8mf2(v871, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v875 = __riscv_vreinterpret_v_u8mf2_i8mf2(v874);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v876 = v32 + 72;
        const int8_t* v877 = (const int8_t*) v876;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v878 = *(const int8_t *)(v877);
        const uint8_t* v879 = v32 + 200;
        const int8_t* v880 = (const int8_t*) v879;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v881 = *(const int8_t *)(v880);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v882 = __riscv_vwmacc_vx_i16m1(v843, v878, v873, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v883 = __riscv_vwmacc_vx_i16m1(v844, v881, v875, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v884 = v32 + 73;
        const int8_t* v885 = (const int8_t*) v884;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v886 = *(const int8_t *)(v885);
        const uint8_t* v887 = v32 + 201;
        const int8_t* v888 = (const int8_t*) v887;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v889 = *(const int8_t *)(v888);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v890 = __riscv_vwmacc_vx_i16m1(v851, v886, v873, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v891 = __riscv_vwmacc_vx_i16m1(v852, v889, v875, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v892 = v32 + 74;
        const int8_t* v893 = (const int8_t*) v892;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v894 = *(const int8_t *)(v893);
        const uint8_t* v895 = v32 + 202;
        const int8_t* v896 = (const int8_t*) v895;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v897 = *(const int8_t *)(v896);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v898 = __riscv_vwmacc_vx_i16m1(v859, v894, v873, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v899 = __riscv_vwmacc_vx_i16m1(v860, v897, v875, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v900 = v32 + 75;
        const int8_t* v901 = (const int8_t*) v900;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v902 = *(const int8_t *)(v901);
        const uint8_t* v903 = v32 + 203;
        const int8_t* v904 = (const int8_t*) v903;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v905 = *(const int8_t *)(v904);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v906 = __riscv_vwmacc_vx_i16m1(v867, v902, v873, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v907 = __riscv_vwmacc_vx_i16m1(v868, v905, v875, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v908 = v30 + 496;
        const uint8_t* v909 = (const uint8_t*) v908;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v910 = __riscv_vle8_v_u8mf2(v909, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v911 = __riscv_vand_vx_u8mf2(v910, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v912 = __riscv_vreinterpret_v_u8mf2_i8mf2(v911);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v913 = __riscv_vsrl_vx_u8mf2(v910, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v914 = __riscv_vreinterpret_v_u8mf2_i8mf2(v913);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v915 = v32 + 76;
        const int8_t* v916 = (const int8_t*) v915;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v917 = *(const int8_t *)(v916);
        const uint8_t* v918 = v32 + 204;
        const int8_t* v919 = (const int8_t*) v918;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v920 = *(const int8_t *)(v919);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v921 = __riscv_vwmacc_vx_i16m1(v882, v917, v912, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v922 = __riscv_vwmacc_vx_i16m1(v883, v920, v914, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v923 = v32 + 77;
        const int8_t* v924 = (const int8_t*) v923;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v925 = *(const int8_t *)(v924);
        const uint8_t* v926 = v32 + 205;
        const int8_t* v927 = (const int8_t*) v926;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v928 = *(const int8_t *)(v927);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v929 = __riscv_vwmacc_vx_i16m1(v890, v925, v912, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v930 = __riscv_vwmacc_vx_i16m1(v891, v928, v914, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v931 = v32 + 78;
        const int8_t* v932 = (const int8_t*) v931;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v933 = *(const int8_t *)(v932);
        const uint8_t* v934 = v32 + 206;
        const int8_t* v935 = (const int8_t*) v934;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v936 = *(const int8_t *)(v935);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v937 = __riscv_vwmacc_vx_i16m1(v898, v933, v912, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v938 = __riscv_vwmacc_vx_i16m1(v899, v936, v914, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v939 = v32 + 79;
        const int8_t* v940 = (const int8_t*) v939;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v941 = *(const int8_t *)(v940);
        const uint8_t* v942 = v32 + 207;
        const int8_t* v943 = (const int8_t*) v942;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v944 = *(const int8_t *)(v943);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v945 = __riscv_vwmacc_vx_i16m1(v906, v941, v912, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v946 = __riscv_vwmacc_vx_i16m1(v907, v944, v914, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v947 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v948 = __riscv_vle16_v_i16m1(v947, 16);
        int16_t* v949 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v950 = __riscv_vle16_v_i16m1(v949, 16);
        vint32m2_t v951 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v952 = __riscv_vwmacc_vv_i32m2(v951, v948, v921, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v953 = __riscv_vwmacc_vv_i32m2(v952, v950, v922, 16);
        v47 = v953;
        vint32m2_t v954 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v955 = __riscv_vwmacc_vv_i32m2(v954, v948, v929, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v956 = __riscv_vwmacc_vv_i32m2(v955, v950, v930, 16);
        v49 = v956;
        vint32m2_t v957 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v958 = __riscv_vwmacc_vv_i32m2(v957, v948, v937, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v959 = __riscv_vwmacc_vv_i32m2(v958, v950, v938, 16);
        v51 = v959;
        vint32m2_t v960 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v961 = __riscv_vwmacc_vv_i32m2(v960, v948, v945, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v962 = __riscv_vwmacc_vv_i32m2(v961, v950, v946, 16);
        v53 = v962;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v963 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v964 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v965 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v966 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v967 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v968 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v969 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v970 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v971 = v30 + 512;
        const uint8_t* v972 = (const uint8_t*) v971;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v973 = __riscv_vle8_v_u8mf2(v972, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v974 = __riscv_vand_vx_u8mf2(v973, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v975 = __riscv_vreinterpret_v_u8mf2_i8mf2(v974);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v976 = __riscv_vsrl_vx_u8mf2(v973, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v977 = __riscv_vreinterpret_v_u8mf2_i8mf2(v976);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v978 = v32 + 80;
        const int8_t* v979 = (const int8_t*) v978;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v980 = *(const int8_t *)(v979);
        const uint8_t* v981 = v32 + 208;
        const int8_t* v982 = (const int8_t*) v981;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v983 = *(const int8_t *)(v982);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v984 = __riscv_vwmacc_vx_i16m1(v963, v980, v975, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v985 = __riscv_vwmacc_vx_i16m1(v964, v983, v977, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v986 = v32 + 81;
        const int8_t* v987 = (const int8_t*) v986;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v988 = *(const int8_t *)(v987);
        const uint8_t* v989 = v32 + 209;
        const int8_t* v990 = (const int8_t*) v989;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v991 = *(const int8_t *)(v990);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v992 = __riscv_vwmacc_vx_i16m1(v965, v988, v975, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v993 = __riscv_vwmacc_vx_i16m1(v966, v991, v977, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v994 = v32 + 82;
        const int8_t* v995 = (const int8_t*) v994;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v996 = *(const int8_t *)(v995);
        const uint8_t* v997 = v32 + 210;
        const int8_t* v998 = (const int8_t*) v997;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v999 = *(const int8_t *)(v998);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1000 = __riscv_vwmacc_vx_i16m1(v967, v996, v975, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1001 = __riscv_vwmacc_vx_i16m1(v968, v999, v977, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1002 = v32 + 83;
        const int8_t* v1003 = (const int8_t*) v1002;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1004 = *(const int8_t *)(v1003);
        const uint8_t* v1005 = v32 + 211;
        const int8_t* v1006 = (const int8_t*) v1005;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1007 = *(const int8_t *)(v1006);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1008 = __riscv_vwmacc_vx_i16m1(v969, v1004, v975, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1009 = __riscv_vwmacc_vx_i16m1(v970, v1007, v977, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1010 = v30 + 528;
        const uint8_t* v1011 = (const uint8_t*) v1010;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1012 = __riscv_vle8_v_u8mf2(v1011, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1013 = __riscv_vand_vx_u8mf2(v1012, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1014 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1013);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1015 = __riscv_vsrl_vx_u8mf2(v1012, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1016 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1015);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1017 = v32 + 84;
        const int8_t* v1018 = (const int8_t*) v1017;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1019 = *(const int8_t *)(v1018);
        const uint8_t* v1020 = v32 + 212;
        const int8_t* v1021 = (const int8_t*) v1020;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1022 = *(const int8_t *)(v1021);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1023 = __riscv_vwmacc_vx_i16m1(v984, v1019, v1014, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1024 = __riscv_vwmacc_vx_i16m1(v985, v1022, v1016, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1025 = v32 + 85;
        const int8_t* v1026 = (const int8_t*) v1025;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1027 = *(const int8_t *)(v1026);
        const uint8_t* v1028 = v32 + 213;
        const int8_t* v1029 = (const int8_t*) v1028;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1030 = *(const int8_t *)(v1029);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1031 = __riscv_vwmacc_vx_i16m1(v992, v1027, v1014, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1032 = __riscv_vwmacc_vx_i16m1(v993, v1030, v1016, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1033 = v32 + 86;
        const int8_t* v1034 = (const int8_t*) v1033;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1035 = *(const int8_t *)(v1034);
        const uint8_t* v1036 = v32 + 214;
        const int8_t* v1037 = (const int8_t*) v1036;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1038 = *(const int8_t *)(v1037);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1039 = __riscv_vwmacc_vx_i16m1(v1000, v1035, v1014, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1040 = __riscv_vwmacc_vx_i16m1(v1001, v1038, v1016, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1041 = v32 + 87;
        const int8_t* v1042 = (const int8_t*) v1041;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1043 = *(const int8_t *)(v1042);
        const uint8_t* v1044 = v32 + 215;
        const int8_t* v1045 = (const int8_t*) v1044;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1046 = *(const int8_t *)(v1045);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1047 = __riscv_vwmacc_vx_i16m1(v1008, v1043, v1014, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1048 = __riscv_vwmacc_vx_i16m1(v1009, v1046, v1016, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1049 = v30 + 544;
        const uint8_t* v1050 = (const uint8_t*) v1049;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1051 = __riscv_vle8_v_u8mf2(v1050, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1052 = __riscv_vand_vx_u8mf2(v1051, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1053 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1052);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1054 = __riscv_vsrl_vx_u8mf2(v1051, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1055 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1054);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1056 = v32 + 88;
        const int8_t* v1057 = (const int8_t*) v1056;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1058 = *(const int8_t *)(v1057);
        const uint8_t* v1059 = v32 + 216;
        const int8_t* v1060 = (const int8_t*) v1059;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1061 = *(const int8_t *)(v1060);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1062 = __riscv_vwmacc_vx_i16m1(v1023, v1058, v1053, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1063 = __riscv_vwmacc_vx_i16m1(v1024, v1061, v1055, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1064 = v32 + 89;
        const int8_t* v1065 = (const int8_t*) v1064;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1066 = *(const int8_t *)(v1065);
        const uint8_t* v1067 = v32 + 217;
        const int8_t* v1068 = (const int8_t*) v1067;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1069 = *(const int8_t *)(v1068);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1070 = __riscv_vwmacc_vx_i16m1(v1031, v1066, v1053, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1071 = __riscv_vwmacc_vx_i16m1(v1032, v1069, v1055, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1072 = v32 + 90;
        const int8_t* v1073 = (const int8_t*) v1072;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1074 = *(const int8_t *)(v1073);
        const uint8_t* v1075 = v32 + 218;
        const int8_t* v1076 = (const int8_t*) v1075;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1077 = *(const int8_t *)(v1076);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1078 = __riscv_vwmacc_vx_i16m1(v1039, v1074, v1053, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1079 = __riscv_vwmacc_vx_i16m1(v1040, v1077, v1055, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1080 = v32 + 91;
        const int8_t* v1081 = (const int8_t*) v1080;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1082 = *(const int8_t *)(v1081);
        const uint8_t* v1083 = v32 + 219;
        const int8_t* v1084 = (const int8_t*) v1083;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1085 = *(const int8_t *)(v1084);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1086 = __riscv_vwmacc_vx_i16m1(v1047, v1082, v1053, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1087 = __riscv_vwmacc_vx_i16m1(v1048, v1085, v1055, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1088 = v30 + 560;
        const uint8_t* v1089 = (const uint8_t*) v1088;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1090 = __riscv_vle8_v_u8mf2(v1089, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1091 = __riscv_vand_vx_u8mf2(v1090, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1092 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1091);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1093 = __riscv_vsrl_vx_u8mf2(v1090, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1094 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1093);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1095 = v32 + 92;
        const int8_t* v1096 = (const int8_t*) v1095;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1097 = *(const int8_t *)(v1096);
        const uint8_t* v1098 = v32 + 220;
        const int8_t* v1099 = (const int8_t*) v1098;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1100 = *(const int8_t *)(v1099);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1101 = __riscv_vwmacc_vx_i16m1(v1062, v1097, v1092, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1102 = __riscv_vwmacc_vx_i16m1(v1063, v1100, v1094, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1103 = v32 + 93;
        const int8_t* v1104 = (const int8_t*) v1103;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1105 = *(const int8_t *)(v1104);
        const uint8_t* v1106 = v32 + 221;
        const int8_t* v1107 = (const int8_t*) v1106;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1108 = *(const int8_t *)(v1107);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1109 = __riscv_vwmacc_vx_i16m1(v1070, v1105, v1092, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1110 = __riscv_vwmacc_vx_i16m1(v1071, v1108, v1094, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1111 = v32 + 94;
        const int8_t* v1112 = (const int8_t*) v1111;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1113 = *(const int8_t *)(v1112);
        const uint8_t* v1114 = v32 + 222;
        const int8_t* v1115 = (const int8_t*) v1114;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1116 = *(const int8_t *)(v1115);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1117 = __riscv_vwmacc_vx_i16m1(v1078, v1113, v1092, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1118 = __riscv_vwmacc_vx_i16m1(v1079, v1116, v1094, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1119 = v32 + 95;
        const int8_t* v1120 = (const int8_t*) v1119;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1121 = *(const int8_t *)(v1120);
        const uint8_t* v1122 = v32 + 223;
        const int8_t* v1123 = (const int8_t*) v1122;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1124 = *(const int8_t *)(v1123);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1125 = __riscv_vwmacc_vx_i16m1(v1086, v1121, v1092, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1126 = __riscv_vwmacc_vx_i16m1(v1087, v1124, v1094, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1127 = v30 + 576;
        const uint8_t* v1128 = (const uint8_t*) v1127;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1129 = __riscv_vle8_v_u8mf2(v1128, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1130 = __riscv_vand_vx_u8mf2(v1129, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1131 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1130);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1132 = __riscv_vsrl_vx_u8mf2(v1129, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1133 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1132);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1134 = v32 + 96;
        const int8_t* v1135 = (const int8_t*) v1134;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1136 = *(const int8_t *)(v1135);
        const uint8_t* v1137 = v32 + 224;
        const int8_t* v1138 = (const int8_t*) v1137;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1139 = *(const int8_t *)(v1138);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1140 = __riscv_vwmacc_vx_i16m1(v1101, v1136, v1131, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1141 = __riscv_vwmacc_vx_i16m1(v1102, v1139, v1133, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1142 = v32 + 97;
        const int8_t* v1143 = (const int8_t*) v1142;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1144 = *(const int8_t *)(v1143);
        const uint8_t* v1145 = v32 + 225;
        const int8_t* v1146 = (const int8_t*) v1145;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1147 = *(const int8_t *)(v1146);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1148 = __riscv_vwmacc_vx_i16m1(v1109, v1144, v1131, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1149 = __riscv_vwmacc_vx_i16m1(v1110, v1147, v1133, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1150 = v32 + 98;
        const int8_t* v1151 = (const int8_t*) v1150;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1152 = *(const int8_t *)(v1151);
        const uint8_t* v1153 = v32 + 226;
        const int8_t* v1154 = (const int8_t*) v1153;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1155 = *(const int8_t *)(v1154);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1156 = __riscv_vwmacc_vx_i16m1(v1117, v1152, v1131, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1157 = __riscv_vwmacc_vx_i16m1(v1118, v1155, v1133, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1158 = v32 + 99;
        const int8_t* v1159 = (const int8_t*) v1158;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1160 = *(const int8_t *)(v1159);
        const uint8_t* v1161 = v32 + 227;
        const int8_t* v1162 = (const int8_t*) v1161;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1163 = *(const int8_t *)(v1162);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1164 = __riscv_vwmacc_vx_i16m1(v1125, v1160, v1131, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1165 = __riscv_vwmacc_vx_i16m1(v1126, v1163, v1133, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1166 = v30 + 592;
        const uint8_t* v1167 = (const uint8_t*) v1166;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1168 = __riscv_vle8_v_u8mf2(v1167, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1169 = __riscv_vand_vx_u8mf2(v1168, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1170 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1169);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1171 = __riscv_vsrl_vx_u8mf2(v1168, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1172 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1171);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1173 = v32 + 100;
        const int8_t* v1174 = (const int8_t*) v1173;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1175 = *(const int8_t *)(v1174);
        const uint8_t* v1176 = v32 + 228;
        const int8_t* v1177 = (const int8_t*) v1176;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1178 = *(const int8_t *)(v1177);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1179 = __riscv_vwmacc_vx_i16m1(v1140, v1175, v1170, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1180 = __riscv_vwmacc_vx_i16m1(v1141, v1178, v1172, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1181 = v32 + 101;
        const int8_t* v1182 = (const int8_t*) v1181;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1183 = *(const int8_t *)(v1182);
        const uint8_t* v1184 = v32 + 229;
        const int8_t* v1185 = (const int8_t*) v1184;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1186 = *(const int8_t *)(v1185);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1187 = __riscv_vwmacc_vx_i16m1(v1148, v1183, v1170, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1188 = __riscv_vwmacc_vx_i16m1(v1149, v1186, v1172, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1189 = v32 + 102;
        const int8_t* v1190 = (const int8_t*) v1189;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1191 = *(const int8_t *)(v1190);
        const uint8_t* v1192 = v32 + 230;
        const int8_t* v1193 = (const int8_t*) v1192;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1194 = *(const int8_t *)(v1193);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1195 = __riscv_vwmacc_vx_i16m1(v1156, v1191, v1170, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1196 = __riscv_vwmacc_vx_i16m1(v1157, v1194, v1172, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1197 = v32 + 103;
        const int8_t* v1198 = (const int8_t*) v1197;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1199 = *(const int8_t *)(v1198);
        const uint8_t* v1200 = v32 + 231;
        const int8_t* v1201 = (const int8_t*) v1200;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1202 = *(const int8_t *)(v1201);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1203 = __riscv_vwmacc_vx_i16m1(v1164, v1199, v1170, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1204 = __riscv_vwmacc_vx_i16m1(v1165, v1202, v1172, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1205 = v30 + 608;
        const uint8_t* v1206 = (const uint8_t*) v1205;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1207 = __riscv_vle8_v_u8mf2(v1206, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1208 = __riscv_vand_vx_u8mf2(v1207, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1209 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1208);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1210 = __riscv_vsrl_vx_u8mf2(v1207, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1211 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1210);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1212 = v32 + 104;
        const int8_t* v1213 = (const int8_t*) v1212;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1214 = *(const int8_t *)(v1213);
        const uint8_t* v1215 = v32 + 232;
        const int8_t* v1216 = (const int8_t*) v1215;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1217 = *(const int8_t *)(v1216);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1218 = __riscv_vwmacc_vx_i16m1(v1179, v1214, v1209, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1219 = __riscv_vwmacc_vx_i16m1(v1180, v1217, v1211, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1220 = v32 + 105;
        const int8_t* v1221 = (const int8_t*) v1220;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1222 = *(const int8_t *)(v1221);
        const uint8_t* v1223 = v32 + 233;
        const int8_t* v1224 = (const int8_t*) v1223;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1225 = *(const int8_t *)(v1224);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1226 = __riscv_vwmacc_vx_i16m1(v1187, v1222, v1209, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1227 = __riscv_vwmacc_vx_i16m1(v1188, v1225, v1211, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1228 = v32 + 106;
        const int8_t* v1229 = (const int8_t*) v1228;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1230 = *(const int8_t *)(v1229);
        const uint8_t* v1231 = v32 + 234;
        const int8_t* v1232 = (const int8_t*) v1231;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1233 = *(const int8_t *)(v1232);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1234 = __riscv_vwmacc_vx_i16m1(v1195, v1230, v1209, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1235 = __riscv_vwmacc_vx_i16m1(v1196, v1233, v1211, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1236 = v32 + 107;
        const int8_t* v1237 = (const int8_t*) v1236;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1238 = *(const int8_t *)(v1237);
        const uint8_t* v1239 = v32 + 235;
        const int8_t* v1240 = (const int8_t*) v1239;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1241 = *(const int8_t *)(v1240);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1242 = __riscv_vwmacc_vx_i16m1(v1203, v1238, v1209, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1243 = __riscv_vwmacc_vx_i16m1(v1204, v1241, v1211, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1244 = v30 + 624;
        const uint8_t* v1245 = (const uint8_t*) v1244;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1246 = __riscv_vle8_v_u8mf2(v1245, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1247 = __riscv_vand_vx_u8mf2(v1246, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1248 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1247);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1249 = __riscv_vsrl_vx_u8mf2(v1246, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1250 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1249);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1251 = v32 + 108;
        const int8_t* v1252 = (const int8_t*) v1251;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1253 = *(const int8_t *)(v1252);
        const uint8_t* v1254 = v32 + 236;
        const int8_t* v1255 = (const int8_t*) v1254;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1256 = *(const int8_t *)(v1255);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1257 = __riscv_vwmacc_vx_i16m1(v1218, v1253, v1248, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1258 = __riscv_vwmacc_vx_i16m1(v1219, v1256, v1250, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1259 = v32 + 109;
        const int8_t* v1260 = (const int8_t*) v1259;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1261 = *(const int8_t *)(v1260);
        const uint8_t* v1262 = v32 + 237;
        const int8_t* v1263 = (const int8_t*) v1262;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1264 = *(const int8_t *)(v1263);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1265 = __riscv_vwmacc_vx_i16m1(v1226, v1261, v1248, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1266 = __riscv_vwmacc_vx_i16m1(v1227, v1264, v1250, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1267 = v32 + 110;
        const int8_t* v1268 = (const int8_t*) v1267;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1269 = *(const int8_t *)(v1268);
        const uint8_t* v1270 = v32 + 238;
        const int8_t* v1271 = (const int8_t*) v1270;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1272 = *(const int8_t *)(v1271);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1273 = __riscv_vwmacc_vx_i16m1(v1234, v1269, v1248, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1274 = __riscv_vwmacc_vx_i16m1(v1235, v1272, v1250, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1275 = v32 + 111;
        const int8_t* v1276 = (const int8_t*) v1275;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1277 = *(const int8_t *)(v1276);
        const uint8_t* v1278 = v32 + 239;
        const int8_t* v1279 = (const int8_t*) v1278;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1280 = *(const int8_t *)(v1279);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1281 = __riscv_vwmacc_vx_i16m1(v1242, v1277, v1248, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1282 = __riscv_vwmacc_vx_i16m1(v1243, v1280, v1250, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1283 = v30 + 640;
        const uint8_t* v1284 = (const uint8_t*) v1283;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1285 = __riscv_vle8_v_u8mf2(v1284, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1286 = __riscv_vand_vx_u8mf2(v1285, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1287 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1286);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1288 = __riscv_vsrl_vx_u8mf2(v1285, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1289 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1288);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1290 = v32 + 112;
        const int8_t* v1291 = (const int8_t*) v1290;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1292 = *(const int8_t *)(v1291);
        const uint8_t* v1293 = v32 + 240;
        const int8_t* v1294 = (const int8_t*) v1293;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1295 = *(const int8_t *)(v1294);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1296 = __riscv_vwmacc_vx_i16m1(v1257, v1292, v1287, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1297 = __riscv_vwmacc_vx_i16m1(v1258, v1295, v1289, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1298 = v32 + 113;
        const int8_t* v1299 = (const int8_t*) v1298;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1300 = *(const int8_t *)(v1299);
        const uint8_t* v1301 = v32 + 241;
        const int8_t* v1302 = (const int8_t*) v1301;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1303 = *(const int8_t *)(v1302);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1304 = __riscv_vwmacc_vx_i16m1(v1265, v1300, v1287, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1305 = __riscv_vwmacc_vx_i16m1(v1266, v1303, v1289, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1306 = v32 + 114;
        const int8_t* v1307 = (const int8_t*) v1306;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1308 = *(const int8_t *)(v1307);
        const uint8_t* v1309 = v32 + 242;
        const int8_t* v1310 = (const int8_t*) v1309;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1311 = *(const int8_t *)(v1310);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1312 = __riscv_vwmacc_vx_i16m1(v1273, v1308, v1287, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1313 = __riscv_vwmacc_vx_i16m1(v1274, v1311, v1289, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1314 = v32 + 115;
        const int8_t* v1315 = (const int8_t*) v1314;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1316 = *(const int8_t *)(v1315);
        const uint8_t* v1317 = v32 + 243;
        const int8_t* v1318 = (const int8_t*) v1317;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1319 = *(const int8_t *)(v1318);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1320 = __riscv_vwmacc_vx_i16m1(v1281, v1316, v1287, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1321 = __riscv_vwmacc_vx_i16m1(v1282, v1319, v1289, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1322 = v30 + 656;
        const uint8_t* v1323 = (const uint8_t*) v1322;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1324 = __riscv_vle8_v_u8mf2(v1323, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1325 = __riscv_vand_vx_u8mf2(v1324, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1326 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1325);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1327 = __riscv_vsrl_vx_u8mf2(v1324, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1328 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1327);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1329 = v32 + 116;
        const int8_t* v1330 = (const int8_t*) v1329;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1331 = *(const int8_t *)(v1330);
        const uint8_t* v1332 = v32 + 244;
        const int8_t* v1333 = (const int8_t*) v1332;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1334 = *(const int8_t *)(v1333);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1335 = __riscv_vwmacc_vx_i16m1(v1296, v1331, v1326, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1336 = __riscv_vwmacc_vx_i16m1(v1297, v1334, v1328, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1337 = v32 + 117;
        const int8_t* v1338 = (const int8_t*) v1337;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1339 = *(const int8_t *)(v1338);
        const uint8_t* v1340 = v32 + 245;
        const int8_t* v1341 = (const int8_t*) v1340;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1342 = *(const int8_t *)(v1341);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1343 = __riscv_vwmacc_vx_i16m1(v1304, v1339, v1326, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1344 = __riscv_vwmacc_vx_i16m1(v1305, v1342, v1328, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1345 = v32 + 118;
        const int8_t* v1346 = (const int8_t*) v1345;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1347 = *(const int8_t *)(v1346);
        const uint8_t* v1348 = v32 + 246;
        const int8_t* v1349 = (const int8_t*) v1348;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1350 = *(const int8_t *)(v1349);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1351 = __riscv_vwmacc_vx_i16m1(v1312, v1347, v1326, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1352 = __riscv_vwmacc_vx_i16m1(v1313, v1350, v1328, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1353 = v32 + 119;
        const int8_t* v1354 = (const int8_t*) v1353;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1355 = *(const int8_t *)(v1354);
        const uint8_t* v1356 = v32 + 247;
        const int8_t* v1357 = (const int8_t*) v1356;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1358 = *(const int8_t *)(v1357);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1359 = __riscv_vwmacc_vx_i16m1(v1320, v1355, v1326, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1360 = __riscv_vwmacc_vx_i16m1(v1321, v1358, v1328, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1361 = v30 + 672;
        const uint8_t* v1362 = (const uint8_t*) v1361;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1363 = __riscv_vle8_v_u8mf2(v1362, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1364 = __riscv_vand_vx_u8mf2(v1363, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1365 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1364);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1366 = __riscv_vsrl_vx_u8mf2(v1363, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1367 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1366);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1368 = v32 + 120;
        const int8_t* v1369 = (const int8_t*) v1368;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1370 = *(const int8_t *)(v1369);
        const uint8_t* v1371 = v32 + 248;
        const int8_t* v1372 = (const int8_t*) v1371;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1373 = *(const int8_t *)(v1372);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1374 = __riscv_vwmacc_vx_i16m1(v1335, v1370, v1365, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1375 = __riscv_vwmacc_vx_i16m1(v1336, v1373, v1367, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1376 = v32 + 121;
        const int8_t* v1377 = (const int8_t*) v1376;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1378 = *(const int8_t *)(v1377);
        const uint8_t* v1379 = v32 + 249;
        const int8_t* v1380 = (const int8_t*) v1379;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1381 = *(const int8_t *)(v1380);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1382 = __riscv_vwmacc_vx_i16m1(v1343, v1378, v1365, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1383 = __riscv_vwmacc_vx_i16m1(v1344, v1381, v1367, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1384 = v32 + 122;
        const int8_t* v1385 = (const int8_t*) v1384;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1386 = *(const int8_t *)(v1385);
        const uint8_t* v1387 = v32 + 250;
        const int8_t* v1388 = (const int8_t*) v1387;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1389 = *(const int8_t *)(v1388);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1390 = __riscv_vwmacc_vx_i16m1(v1351, v1386, v1365, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1391 = __riscv_vwmacc_vx_i16m1(v1352, v1389, v1367, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1392 = v32 + 123;
        const int8_t* v1393 = (const int8_t*) v1392;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1394 = *(const int8_t *)(v1393);
        const uint8_t* v1395 = v32 + 251;
        const int8_t* v1396 = (const int8_t*) v1395;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1397 = *(const int8_t *)(v1396);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1398 = __riscv_vwmacc_vx_i16m1(v1359, v1394, v1365, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1399 = __riscv_vwmacc_vx_i16m1(v1360, v1397, v1367, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1400 = v30 + 688;
        const uint8_t* v1401 = (const uint8_t*) v1400;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1402 = __riscv_vle8_v_u8mf2(v1401, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1403 = __riscv_vand_vx_u8mf2(v1402, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1404 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1403);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1405 = __riscv_vsrl_vx_u8mf2(v1402, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1406 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1405);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1407 = v32 + 124;
        const int8_t* v1408 = (const int8_t*) v1407;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1409 = *(const int8_t *)(v1408);
        const uint8_t* v1410 = v32 + 252;
        const int8_t* v1411 = (const int8_t*) v1410;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1412 = *(const int8_t *)(v1411);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1413 = __riscv_vwmacc_vx_i16m1(v1374, v1409, v1404, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1414 = __riscv_vwmacc_vx_i16m1(v1375, v1412, v1406, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1415 = v32 + 125;
        const int8_t* v1416 = (const int8_t*) v1415;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1417 = *(const int8_t *)(v1416);
        const uint8_t* v1418 = v32 + 253;
        const int8_t* v1419 = (const int8_t*) v1418;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1420 = *(const int8_t *)(v1419);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1421 = __riscv_vwmacc_vx_i16m1(v1382, v1417, v1404, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1422 = __riscv_vwmacc_vx_i16m1(v1383, v1420, v1406, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1423 = v32 + 126;
        const int8_t* v1424 = (const int8_t*) v1423;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1425 = *(const int8_t *)(v1424);
        const uint8_t* v1426 = v32 + 254;
        const int8_t* v1427 = (const int8_t*) v1426;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1428 = *(const int8_t *)(v1427);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1429 = __riscv_vwmacc_vx_i16m1(v1390, v1425, v1404, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1430 = __riscv_vwmacc_vx_i16m1(v1391, v1428, v1406, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1431 = v32 + 127;
        const int8_t* v1432 = (const int8_t*) v1431;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1433 = *(const int8_t *)(v1432);
        const uint8_t* v1434 = v32 + 255;
        const int8_t* v1435 = (const int8_t*) v1434;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1436 = *(const int8_t *)(v1435);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1437 = __riscv_vwmacc_vx_i16m1(v1398, v1433, v1404, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1438 = __riscv_vwmacc_vx_i16m1(v1399, v1436, v1406, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1439 = v30 + 704;
        const uint8_t* v1440 = (const uint8_t*) v1439;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1441 = __riscv_vle8_v_u8mf2(v1440, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1442 = __riscv_vand_vx_u8mf2(v1441, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1443 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1442);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1444 = __riscv_vsrl_vx_u8mf2(v1441, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1445 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1444);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1446 = v32 + 128;
        const int8_t* v1447 = (const int8_t*) v1446;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1448 = *(const int8_t *)(v1447);
        const uint8_t* v1449 = v32 + 256;
        const int8_t* v1450 = (const int8_t*) v1449;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1451 = *(const int8_t *)(v1450);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1452 = __riscv_vwmacc_vx_i16m1(v1413, v1448, v1443, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1453 = __riscv_vwmacc_vx_i16m1(v1414, v1451, v1445, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1454 = v32 + 129;
        const int8_t* v1455 = (const int8_t*) v1454;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1456 = *(const int8_t *)(v1455);
        const uint8_t* v1457 = v32 + 257;
        const int8_t* v1458 = (const int8_t*) v1457;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1459 = *(const int8_t *)(v1458);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1460 = __riscv_vwmacc_vx_i16m1(v1421, v1456, v1443, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1461 = __riscv_vwmacc_vx_i16m1(v1422, v1459, v1445, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1462 = v32 + 130;
        const int8_t* v1463 = (const int8_t*) v1462;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1464 = *(const int8_t *)(v1463);
        const uint8_t* v1465 = v32 + 258;
        const int8_t* v1466 = (const int8_t*) v1465;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1467 = *(const int8_t *)(v1466);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1468 = __riscv_vwmacc_vx_i16m1(v1429, v1464, v1443, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1469 = __riscv_vwmacc_vx_i16m1(v1430, v1467, v1445, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1470 = v32 + 131;
        const int8_t* v1471 = (const int8_t*) v1470;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1472 = *(const int8_t *)(v1471);
        const uint8_t* v1473 = v32 + 259;
        const int8_t* v1474 = (const int8_t*) v1473;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1475 = *(const int8_t *)(v1474);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1476 = __riscv_vwmacc_vx_i16m1(v1437, v1472, v1443, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1477 = __riscv_vwmacc_vx_i16m1(v1438, v1475, v1445, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1478 = v30 + 720;
        const uint8_t* v1479 = (const uint8_t*) v1478;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1480 = __riscv_vle8_v_u8mf2(v1479, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1481 = __riscv_vand_vx_u8mf2(v1480, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1482 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1481);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1483 = __riscv_vsrl_vx_u8mf2(v1480, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1484 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1483);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1485 = v32 + 132;
        const int8_t* v1486 = (const int8_t*) v1485;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1487 = *(const int8_t *)(v1486);
        const uint8_t* v1488 = v32 + 260;
        const int8_t* v1489 = (const int8_t*) v1488;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1490 = *(const int8_t *)(v1489);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1491 = __riscv_vwmacc_vx_i16m1(v1452, v1487, v1482, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1492 = __riscv_vwmacc_vx_i16m1(v1453, v1490, v1484, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1493 = v32 + 133;
        const int8_t* v1494 = (const int8_t*) v1493;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1495 = *(const int8_t *)(v1494);
        const uint8_t* v1496 = v32 + 261;
        const int8_t* v1497 = (const int8_t*) v1496;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1498 = *(const int8_t *)(v1497);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1499 = __riscv_vwmacc_vx_i16m1(v1460, v1495, v1482, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1500 = __riscv_vwmacc_vx_i16m1(v1461, v1498, v1484, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1501 = v32 + 134;
        const int8_t* v1502 = (const int8_t*) v1501;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1503 = *(const int8_t *)(v1502);
        const uint8_t* v1504 = v32 + 262;
        const int8_t* v1505 = (const int8_t*) v1504;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1506 = *(const int8_t *)(v1505);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1507 = __riscv_vwmacc_vx_i16m1(v1468, v1503, v1482, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1508 = __riscv_vwmacc_vx_i16m1(v1469, v1506, v1484, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1509 = v32 + 135;
        const int8_t* v1510 = (const int8_t*) v1509;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1511 = *(const int8_t *)(v1510);
        const uint8_t* v1512 = v32 + 263;
        const int8_t* v1513 = (const int8_t*) v1512;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1514 = *(const int8_t *)(v1513);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1515 = __riscv_vwmacc_vx_i16m1(v1476, v1511, v1482, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1516 = __riscv_vwmacc_vx_i16m1(v1477, v1514, v1484, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1517 = v30 + 736;
        const uint8_t* v1518 = (const uint8_t*) v1517;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1519 = __riscv_vle8_v_u8mf2(v1518, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1520 = __riscv_vand_vx_u8mf2(v1519, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1521 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1520);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1522 = __riscv_vsrl_vx_u8mf2(v1519, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1523 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1522);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1524 = v32 + 136;
        const int8_t* v1525 = (const int8_t*) v1524;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1526 = *(const int8_t *)(v1525);
        const uint8_t* v1527 = v32 + 264;
        const int8_t* v1528 = (const int8_t*) v1527;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1529 = *(const int8_t *)(v1528);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1530 = __riscv_vwmacc_vx_i16m1(v1491, v1526, v1521, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1531 = __riscv_vwmacc_vx_i16m1(v1492, v1529, v1523, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1532 = v32 + 137;
        const int8_t* v1533 = (const int8_t*) v1532;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1534 = *(const int8_t *)(v1533);
        const uint8_t* v1535 = v32 + 265;
        const int8_t* v1536 = (const int8_t*) v1535;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1537 = *(const int8_t *)(v1536);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1538 = __riscv_vwmacc_vx_i16m1(v1499, v1534, v1521, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1539 = __riscv_vwmacc_vx_i16m1(v1500, v1537, v1523, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1540 = v32 + 138;
        const int8_t* v1541 = (const int8_t*) v1540;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1542 = *(const int8_t *)(v1541);
        const uint8_t* v1543 = v32 + 266;
        const int8_t* v1544 = (const int8_t*) v1543;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1545 = *(const int8_t *)(v1544);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1546 = __riscv_vwmacc_vx_i16m1(v1507, v1542, v1521, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1547 = __riscv_vwmacc_vx_i16m1(v1508, v1545, v1523, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1548 = v32 + 139;
        const int8_t* v1549 = (const int8_t*) v1548;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1550 = *(const int8_t *)(v1549);
        const uint8_t* v1551 = v32 + 267;
        const int8_t* v1552 = (const int8_t*) v1551;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1553 = *(const int8_t *)(v1552);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1554 = __riscv_vwmacc_vx_i16m1(v1515, v1550, v1521, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1555 = __riscv_vwmacc_vx_i16m1(v1516, v1553, v1523, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1556 = v30 + 752;
        const uint8_t* v1557 = (const uint8_t*) v1556;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1558 = __riscv_vle8_v_u8mf2(v1557, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1559 = __riscv_vand_vx_u8mf2(v1558, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1560 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1559);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1561 = __riscv_vsrl_vx_u8mf2(v1558, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1562 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1561);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1563 = v32 + 140;
        const int8_t* v1564 = (const int8_t*) v1563;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1565 = *(const int8_t *)(v1564);
        const uint8_t* v1566 = v32 + 268;
        const int8_t* v1567 = (const int8_t*) v1566;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1568 = *(const int8_t *)(v1567);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1569 = __riscv_vwmacc_vx_i16m1(v1530, v1565, v1560, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1570 = __riscv_vwmacc_vx_i16m1(v1531, v1568, v1562, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1571 = v32 + 141;
        const int8_t* v1572 = (const int8_t*) v1571;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1573 = *(const int8_t *)(v1572);
        const uint8_t* v1574 = v32 + 269;
        const int8_t* v1575 = (const int8_t*) v1574;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1576 = *(const int8_t *)(v1575);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1577 = __riscv_vwmacc_vx_i16m1(v1538, v1573, v1560, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1578 = __riscv_vwmacc_vx_i16m1(v1539, v1576, v1562, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1579 = v32 + 142;
        const int8_t* v1580 = (const int8_t*) v1579;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1581 = *(const int8_t *)(v1580);
        const uint8_t* v1582 = v32 + 270;
        const int8_t* v1583 = (const int8_t*) v1582;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1584 = *(const int8_t *)(v1583);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1585 = __riscv_vwmacc_vx_i16m1(v1546, v1581, v1560, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1586 = __riscv_vwmacc_vx_i16m1(v1547, v1584, v1562, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1587 = v32 + 143;
        const int8_t* v1588 = (const int8_t*) v1587;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1589 = *(const int8_t *)(v1588);
        const uint8_t* v1590 = v32 + 271;
        const int8_t* v1591 = (const int8_t*) v1590;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1592 = *(const int8_t *)(v1591);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1593 = __riscv_vwmacc_vx_i16m1(v1554, v1589, v1560, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1594 = __riscv_vwmacc_vx_i16m1(v1555, v1592, v1562, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v1595 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1596 = __riscv_vle16_v_i16m1(v1595, 16);
        int16_t* v1597 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1598 = __riscv_vle16_v_i16m1(v1597, 16);
        vint32m2_t v1599 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1600 = __riscv_vwmacc_vv_i32m2(v1599, v1596, v1569, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1601 = __riscv_vwmacc_vv_i32m2(v1600, v1598, v1570, 16);
        v47 = v1601;
        vint32m2_t v1602 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1603 = __riscv_vwmacc_vv_i32m2(v1602, v1596, v1577, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1604 = __riscv_vwmacc_vv_i32m2(v1603, v1598, v1578, 16);
        v49 = v1604;
        vint32m2_t v1605 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1606 = __riscv_vwmacc_vv_i32m2(v1605, v1596, v1585, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1607 = __riscv_vwmacc_vv_i32m2(v1606, v1598, v1586, 16);
        v51 = v1607;
        vint32m2_t v1608 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1609 = __riscv_vwmacc_vv_i32m2(v1608, v1596, v1593, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1610 = __riscv_vwmacc_vv_i32m2(v1609, v1598, v1594, 16);
        v53 = v1610;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1611 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1612 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1613 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1614 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1615 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1616 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1617 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1618 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1619 = v30 + 768;
        const uint8_t* v1620 = (const uint8_t*) v1619;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1621 = __riscv_vle8_v_u8mf2(v1620, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1622 = __riscv_vand_vx_u8mf2(v1621, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1623 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1622);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1624 = __riscv_vsrl_vx_u8mf2(v1621, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1625 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1624);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1626 = v32 + 272;
        const int8_t* v1627 = (const int8_t*) v1626;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1628 = *(const int8_t *)(v1627);
        const uint8_t* v1629 = v32 + 400;
        const int8_t* v1630 = (const int8_t*) v1629;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1631 = *(const int8_t *)(v1630);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1632 = __riscv_vwmacc_vx_i16m1(v1611, v1628, v1623, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1633 = __riscv_vwmacc_vx_i16m1(v1612, v1631, v1625, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1634 = v32 + 273;
        const int8_t* v1635 = (const int8_t*) v1634;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1636 = *(const int8_t *)(v1635);
        const uint8_t* v1637 = v32 + 401;
        const int8_t* v1638 = (const int8_t*) v1637;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1639 = *(const int8_t *)(v1638);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1640 = __riscv_vwmacc_vx_i16m1(v1613, v1636, v1623, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1641 = __riscv_vwmacc_vx_i16m1(v1614, v1639, v1625, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1642 = v32 + 274;
        const int8_t* v1643 = (const int8_t*) v1642;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1644 = *(const int8_t *)(v1643);
        const uint8_t* v1645 = v32 + 402;
        const int8_t* v1646 = (const int8_t*) v1645;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1647 = *(const int8_t *)(v1646);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1648 = __riscv_vwmacc_vx_i16m1(v1615, v1644, v1623, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1649 = __riscv_vwmacc_vx_i16m1(v1616, v1647, v1625, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1650 = v32 + 275;
        const int8_t* v1651 = (const int8_t*) v1650;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1652 = *(const int8_t *)(v1651);
        const uint8_t* v1653 = v32 + 403;
        const int8_t* v1654 = (const int8_t*) v1653;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1655 = *(const int8_t *)(v1654);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1656 = __riscv_vwmacc_vx_i16m1(v1617, v1652, v1623, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1657 = __riscv_vwmacc_vx_i16m1(v1618, v1655, v1625, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1658 = v30 + 784;
        const uint8_t* v1659 = (const uint8_t*) v1658;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1660 = __riscv_vle8_v_u8mf2(v1659, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1661 = __riscv_vand_vx_u8mf2(v1660, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1662 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1661);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1663 = __riscv_vsrl_vx_u8mf2(v1660, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1664 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1663);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1665 = v32 + 276;
        const int8_t* v1666 = (const int8_t*) v1665;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1667 = *(const int8_t *)(v1666);
        const uint8_t* v1668 = v32 + 404;
        const int8_t* v1669 = (const int8_t*) v1668;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1670 = *(const int8_t *)(v1669);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1671 = __riscv_vwmacc_vx_i16m1(v1632, v1667, v1662, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1672 = __riscv_vwmacc_vx_i16m1(v1633, v1670, v1664, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1673 = v32 + 277;
        const int8_t* v1674 = (const int8_t*) v1673;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1675 = *(const int8_t *)(v1674);
        const uint8_t* v1676 = v32 + 405;
        const int8_t* v1677 = (const int8_t*) v1676;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1678 = *(const int8_t *)(v1677);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1679 = __riscv_vwmacc_vx_i16m1(v1640, v1675, v1662, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1680 = __riscv_vwmacc_vx_i16m1(v1641, v1678, v1664, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1681 = v32 + 278;
        const int8_t* v1682 = (const int8_t*) v1681;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1683 = *(const int8_t *)(v1682);
        const uint8_t* v1684 = v32 + 406;
        const int8_t* v1685 = (const int8_t*) v1684;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1686 = *(const int8_t *)(v1685);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1687 = __riscv_vwmacc_vx_i16m1(v1648, v1683, v1662, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1688 = __riscv_vwmacc_vx_i16m1(v1649, v1686, v1664, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1689 = v32 + 279;
        const int8_t* v1690 = (const int8_t*) v1689;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1691 = *(const int8_t *)(v1690);
        const uint8_t* v1692 = v32 + 407;
        const int8_t* v1693 = (const int8_t*) v1692;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1694 = *(const int8_t *)(v1693);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1695 = __riscv_vwmacc_vx_i16m1(v1656, v1691, v1662, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1696 = __riscv_vwmacc_vx_i16m1(v1657, v1694, v1664, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1697 = v30 + 800;
        const uint8_t* v1698 = (const uint8_t*) v1697;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1699 = __riscv_vle8_v_u8mf2(v1698, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1700 = __riscv_vand_vx_u8mf2(v1699, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1701 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1700);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1702 = __riscv_vsrl_vx_u8mf2(v1699, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1703 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1702);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1704 = v32 + 280;
        const int8_t* v1705 = (const int8_t*) v1704;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1706 = *(const int8_t *)(v1705);
        const uint8_t* v1707 = v32 + 408;
        const int8_t* v1708 = (const int8_t*) v1707;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1709 = *(const int8_t *)(v1708);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1710 = __riscv_vwmacc_vx_i16m1(v1671, v1706, v1701, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1711 = __riscv_vwmacc_vx_i16m1(v1672, v1709, v1703, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1712 = v32 + 281;
        const int8_t* v1713 = (const int8_t*) v1712;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1714 = *(const int8_t *)(v1713);
        const uint8_t* v1715 = v32 + 409;
        const int8_t* v1716 = (const int8_t*) v1715;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1717 = *(const int8_t *)(v1716);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1718 = __riscv_vwmacc_vx_i16m1(v1679, v1714, v1701, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1719 = __riscv_vwmacc_vx_i16m1(v1680, v1717, v1703, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1720 = v32 + 282;
        const int8_t* v1721 = (const int8_t*) v1720;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1722 = *(const int8_t *)(v1721);
        const uint8_t* v1723 = v32 + 410;
        const int8_t* v1724 = (const int8_t*) v1723;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1725 = *(const int8_t *)(v1724);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1726 = __riscv_vwmacc_vx_i16m1(v1687, v1722, v1701, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1727 = __riscv_vwmacc_vx_i16m1(v1688, v1725, v1703, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1728 = v32 + 283;
        const int8_t* v1729 = (const int8_t*) v1728;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1730 = *(const int8_t *)(v1729);
        const uint8_t* v1731 = v32 + 411;
        const int8_t* v1732 = (const int8_t*) v1731;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1733 = *(const int8_t *)(v1732);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1734 = __riscv_vwmacc_vx_i16m1(v1695, v1730, v1701, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1735 = __riscv_vwmacc_vx_i16m1(v1696, v1733, v1703, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1736 = v30 + 816;
        const uint8_t* v1737 = (const uint8_t*) v1736;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1738 = __riscv_vle8_v_u8mf2(v1737, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1739 = __riscv_vand_vx_u8mf2(v1738, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1740 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1739);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1741 = __riscv_vsrl_vx_u8mf2(v1738, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1742 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1741);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1743 = v32 + 284;
        const int8_t* v1744 = (const int8_t*) v1743;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1745 = *(const int8_t *)(v1744);
        const uint8_t* v1746 = v32 + 412;
        const int8_t* v1747 = (const int8_t*) v1746;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1748 = *(const int8_t *)(v1747);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1749 = __riscv_vwmacc_vx_i16m1(v1710, v1745, v1740, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1750 = __riscv_vwmacc_vx_i16m1(v1711, v1748, v1742, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1751 = v32 + 285;
        const int8_t* v1752 = (const int8_t*) v1751;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1753 = *(const int8_t *)(v1752);
        const uint8_t* v1754 = v32 + 413;
        const int8_t* v1755 = (const int8_t*) v1754;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1756 = *(const int8_t *)(v1755);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1757 = __riscv_vwmacc_vx_i16m1(v1718, v1753, v1740, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1758 = __riscv_vwmacc_vx_i16m1(v1719, v1756, v1742, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1759 = v32 + 286;
        const int8_t* v1760 = (const int8_t*) v1759;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1761 = *(const int8_t *)(v1760);
        const uint8_t* v1762 = v32 + 414;
        const int8_t* v1763 = (const int8_t*) v1762;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1764 = *(const int8_t *)(v1763);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1765 = __riscv_vwmacc_vx_i16m1(v1726, v1761, v1740, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1766 = __riscv_vwmacc_vx_i16m1(v1727, v1764, v1742, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1767 = v32 + 287;
        const int8_t* v1768 = (const int8_t*) v1767;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1769 = *(const int8_t *)(v1768);
        const uint8_t* v1770 = v32 + 415;
        const int8_t* v1771 = (const int8_t*) v1770;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1772 = *(const int8_t *)(v1771);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1773 = __riscv_vwmacc_vx_i16m1(v1734, v1769, v1740, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1774 = __riscv_vwmacc_vx_i16m1(v1735, v1772, v1742, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1775 = v30 + 832;
        const uint8_t* v1776 = (const uint8_t*) v1775;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1777 = __riscv_vle8_v_u8mf2(v1776, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1778 = __riscv_vand_vx_u8mf2(v1777, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1779 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1778);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1780 = __riscv_vsrl_vx_u8mf2(v1777, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1781 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1780);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1782 = v32 + 288;
        const int8_t* v1783 = (const int8_t*) v1782;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1784 = *(const int8_t *)(v1783);
        const uint8_t* v1785 = v32 + 416;
        const int8_t* v1786 = (const int8_t*) v1785;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1787 = *(const int8_t *)(v1786);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1788 = __riscv_vwmacc_vx_i16m1(v1749, v1784, v1779, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1789 = __riscv_vwmacc_vx_i16m1(v1750, v1787, v1781, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1790 = v32 + 289;
        const int8_t* v1791 = (const int8_t*) v1790;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1792 = *(const int8_t *)(v1791);
        const uint8_t* v1793 = v32 + 417;
        const int8_t* v1794 = (const int8_t*) v1793;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1795 = *(const int8_t *)(v1794);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1796 = __riscv_vwmacc_vx_i16m1(v1757, v1792, v1779, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1797 = __riscv_vwmacc_vx_i16m1(v1758, v1795, v1781, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1798 = v32 + 290;
        const int8_t* v1799 = (const int8_t*) v1798;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1800 = *(const int8_t *)(v1799);
        const uint8_t* v1801 = v32 + 418;
        const int8_t* v1802 = (const int8_t*) v1801;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1803 = *(const int8_t *)(v1802);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1804 = __riscv_vwmacc_vx_i16m1(v1765, v1800, v1779, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1805 = __riscv_vwmacc_vx_i16m1(v1766, v1803, v1781, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1806 = v32 + 291;
        const int8_t* v1807 = (const int8_t*) v1806;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1808 = *(const int8_t *)(v1807);
        const uint8_t* v1809 = v32 + 419;
        const int8_t* v1810 = (const int8_t*) v1809;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1811 = *(const int8_t *)(v1810);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1812 = __riscv_vwmacc_vx_i16m1(v1773, v1808, v1779, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1813 = __riscv_vwmacc_vx_i16m1(v1774, v1811, v1781, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1814 = v30 + 848;
        const uint8_t* v1815 = (const uint8_t*) v1814;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1816 = __riscv_vle8_v_u8mf2(v1815, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1817 = __riscv_vand_vx_u8mf2(v1816, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1818 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1817);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1819 = __riscv_vsrl_vx_u8mf2(v1816, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1820 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1819);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1821 = v32 + 292;
        const int8_t* v1822 = (const int8_t*) v1821;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1823 = *(const int8_t *)(v1822);
        const uint8_t* v1824 = v32 + 420;
        const int8_t* v1825 = (const int8_t*) v1824;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1826 = *(const int8_t *)(v1825);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1827 = __riscv_vwmacc_vx_i16m1(v1788, v1823, v1818, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1828 = __riscv_vwmacc_vx_i16m1(v1789, v1826, v1820, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1829 = v32 + 293;
        const int8_t* v1830 = (const int8_t*) v1829;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1831 = *(const int8_t *)(v1830);
        const uint8_t* v1832 = v32 + 421;
        const int8_t* v1833 = (const int8_t*) v1832;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1834 = *(const int8_t *)(v1833);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1835 = __riscv_vwmacc_vx_i16m1(v1796, v1831, v1818, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1836 = __riscv_vwmacc_vx_i16m1(v1797, v1834, v1820, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1837 = v32 + 294;
        const int8_t* v1838 = (const int8_t*) v1837;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1839 = *(const int8_t *)(v1838);
        const uint8_t* v1840 = v32 + 422;
        const int8_t* v1841 = (const int8_t*) v1840;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1842 = *(const int8_t *)(v1841);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1843 = __riscv_vwmacc_vx_i16m1(v1804, v1839, v1818, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1844 = __riscv_vwmacc_vx_i16m1(v1805, v1842, v1820, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1845 = v32 + 295;
        const int8_t* v1846 = (const int8_t*) v1845;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1847 = *(const int8_t *)(v1846);
        const uint8_t* v1848 = v32 + 423;
        const int8_t* v1849 = (const int8_t*) v1848;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1850 = *(const int8_t *)(v1849);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1851 = __riscv_vwmacc_vx_i16m1(v1812, v1847, v1818, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1852 = __riscv_vwmacc_vx_i16m1(v1813, v1850, v1820, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1853 = v30 + 864;
        const uint8_t* v1854 = (const uint8_t*) v1853;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1855 = __riscv_vle8_v_u8mf2(v1854, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1856 = __riscv_vand_vx_u8mf2(v1855, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1857 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1856);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1858 = __riscv_vsrl_vx_u8mf2(v1855, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1859 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1858);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1860 = v32 + 296;
        const int8_t* v1861 = (const int8_t*) v1860;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1862 = *(const int8_t *)(v1861);
        const uint8_t* v1863 = v32 + 424;
        const int8_t* v1864 = (const int8_t*) v1863;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1865 = *(const int8_t *)(v1864);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1866 = __riscv_vwmacc_vx_i16m1(v1827, v1862, v1857, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1867 = __riscv_vwmacc_vx_i16m1(v1828, v1865, v1859, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1868 = v32 + 297;
        const int8_t* v1869 = (const int8_t*) v1868;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1870 = *(const int8_t *)(v1869);
        const uint8_t* v1871 = v32 + 425;
        const int8_t* v1872 = (const int8_t*) v1871;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1873 = *(const int8_t *)(v1872);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1874 = __riscv_vwmacc_vx_i16m1(v1835, v1870, v1857, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1875 = __riscv_vwmacc_vx_i16m1(v1836, v1873, v1859, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1876 = v32 + 298;
        const int8_t* v1877 = (const int8_t*) v1876;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1878 = *(const int8_t *)(v1877);
        const uint8_t* v1879 = v32 + 426;
        const int8_t* v1880 = (const int8_t*) v1879;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1881 = *(const int8_t *)(v1880);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1882 = __riscv_vwmacc_vx_i16m1(v1843, v1878, v1857, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1883 = __riscv_vwmacc_vx_i16m1(v1844, v1881, v1859, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1884 = v32 + 299;
        const int8_t* v1885 = (const int8_t*) v1884;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1886 = *(const int8_t *)(v1885);
        const uint8_t* v1887 = v32 + 427;
        const int8_t* v1888 = (const int8_t*) v1887;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1889 = *(const int8_t *)(v1888);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1890 = __riscv_vwmacc_vx_i16m1(v1851, v1886, v1857, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1891 = __riscv_vwmacc_vx_i16m1(v1852, v1889, v1859, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1892 = v30 + 880;
        const uint8_t* v1893 = (const uint8_t*) v1892;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1894 = __riscv_vle8_v_u8mf2(v1893, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1895 = __riscv_vand_vx_u8mf2(v1894, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1896 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1895);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1897 = __riscv_vsrl_vx_u8mf2(v1894, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1898 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1897);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1899 = v32 + 300;
        const int8_t* v1900 = (const int8_t*) v1899;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1901 = *(const int8_t *)(v1900);
        const uint8_t* v1902 = v32 + 428;
        const int8_t* v1903 = (const int8_t*) v1902;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1904 = *(const int8_t *)(v1903);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1905 = __riscv_vwmacc_vx_i16m1(v1866, v1901, v1896, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1906 = __riscv_vwmacc_vx_i16m1(v1867, v1904, v1898, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1907 = v32 + 301;
        const int8_t* v1908 = (const int8_t*) v1907;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1909 = *(const int8_t *)(v1908);
        const uint8_t* v1910 = v32 + 429;
        const int8_t* v1911 = (const int8_t*) v1910;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1912 = *(const int8_t *)(v1911);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1913 = __riscv_vwmacc_vx_i16m1(v1874, v1909, v1896, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1914 = __riscv_vwmacc_vx_i16m1(v1875, v1912, v1898, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1915 = v32 + 302;
        const int8_t* v1916 = (const int8_t*) v1915;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1917 = *(const int8_t *)(v1916);
        const uint8_t* v1918 = v32 + 430;
        const int8_t* v1919 = (const int8_t*) v1918;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1920 = *(const int8_t *)(v1919);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1921 = __riscv_vwmacc_vx_i16m1(v1882, v1917, v1896, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1922 = __riscv_vwmacc_vx_i16m1(v1883, v1920, v1898, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1923 = v32 + 303;
        const int8_t* v1924 = (const int8_t*) v1923;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1925 = *(const int8_t *)(v1924);
        const uint8_t* v1926 = v32 + 431;
        const int8_t* v1927 = (const int8_t*) v1926;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1928 = *(const int8_t *)(v1927);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1929 = __riscv_vwmacc_vx_i16m1(v1890, v1925, v1896, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1930 = __riscv_vwmacc_vx_i16m1(v1891, v1928, v1898, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1931 = v30 + 896;
        const uint8_t* v1932 = (const uint8_t*) v1931;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1933 = __riscv_vle8_v_u8mf2(v1932, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1934 = __riscv_vand_vx_u8mf2(v1933, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1935 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1934);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1936 = __riscv_vsrl_vx_u8mf2(v1933, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1937 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1936);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1938 = v32 + 304;
        const int8_t* v1939 = (const int8_t*) v1938;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1940 = *(const int8_t *)(v1939);
        const uint8_t* v1941 = v32 + 432;
        const int8_t* v1942 = (const int8_t*) v1941;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1943 = *(const int8_t *)(v1942);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1944 = __riscv_vwmacc_vx_i16m1(v1905, v1940, v1935, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1945 = __riscv_vwmacc_vx_i16m1(v1906, v1943, v1937, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1946 = v32 + 305;
        const int8_t* v1947 = (const int8_t*) v1946;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1948 = *(const int8_t *)(v1947);
        const uint8_t* v1949 = v32 + 433;
        const int8_t* v1950 = (const int8_t*) v1949;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1951 = *(const int8_t *)(v1950);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1952 = __riscv_vwmacc_vx_i16m1(v1913, v1948, v1935, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1953 = __riscv_vwmacc_vx_i16m1(v1914, v1951, v1937, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1954 = v32 + 306;
        const int8_t* v1955 = (const int8_t*) v1954;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1956 = *(const int8_t *)(v1955);
        const uint8_t* v1957 = v32 + 434;
        const int8_t* v1958 = (const int8_t*) v1957;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1959 = *(const int8_t *)(v1958);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1960 = __riscv_vwmacc_vx_i16m1(v1921, v1956, v1935, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1961 = __riscv_vwmacc_vx_i16m1(v1922, v1959, v1937, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1962 = v32 + 307;
        const int8_t* v1963 = (const int8_t*) v1962;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1964 = *(const int8_t *)(v1963);
        const uint8_t* v1965 = v32 + 435;
        const int8_t* v1966 = (const int8_t*) v1965;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1967 = *(const int8_t *)(v1966);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1968 = __riscv_vwmacc_vx_i16m1(v1929, v1964, v1935, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1969 = __riscv_vwmacc_vx_i16m1(v1930, v1967, v1937, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v1970 = v30 + 912;
        const uint8_t* v1971 = (const uint8_t*) v1970;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1972 = __riscv_vle8_v_u8mf2(v1971, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1973 = __riscv_vand_vx_u8mf2(v1972, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1974 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1973);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1975 = __riscv_vsrl_vx_u8mf2(v1972, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1976 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1975);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1977 = v32 + 308;
        const int8_t* v1978 = (const int8_t*) v1977;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1979 = *(const int8_t *)(v1978);
        const uint8_t* v1980 = v32 + 436;
        const int8_t* v1981 = (const int8_t*) v1980;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1982 = *(const int8_t *)(v1981);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1983 = __riscv_vwmacc_vx_i16m1(v1944, v1979, v1974, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1984 = __riscv_vwmacc_vx_i16m1(v1945, v1982, v1976, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1985 = v32 + 309;
        const int8_t* v1986 = (const int8_t*) v1985;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1987 = *(const int8_t *)(v1986);
        const uint8_t* v1988 = v32 + 437;
        const int8_t* v1989 = (const int8_t*) v1988;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1990 = *(const int8_t *)(v1989);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1991 = __riscv_vwmacc_vx_i16m1(v1952, v1987, v1974, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1992 = __riscv_vwmacc_vx_i16m1(v1953, v1990, v1976, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v1993 = v32 + 310;
        const int8_t* v1994 = (const int8_t*) v1993;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1995 = *(const int8_t *)(v1994);
        const uint8_t* v1996 = v32 + 438;
        const int8_t* v1997 = (const int8_t*) v1996;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1998 = *(const int8_t *)(v1997);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1999 = __riscv_vwmacc_vx_i16m1(v1960, v1995, v1974, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2000 = __riscv_vwmacc_vx_i16m1(v1961, v1998, v1976, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2001 = v32 + 311;
        const int8_t* v2002 = (const int8_t*) v2001;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2003 = *(const int8_t *)(v2002);
        const uint8_t* v2004 = v32 + 439;
        const int8_t* v2005 = (const int8_t*) v2004;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2006 = *(const int8_t *)(v2005);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2007 = __riscv_vwmacc_vx_i16m1(v1968, v2003, v1974, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2008 = __riscv_vwmacc_vx_i16m1(v1969, v2006, v1976, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2009 = v30 + 928;
        const uint8_t* v2010 = (const uint8_t*) v2009;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2011 = __riscv_vle8_v_u8mf2(v2010, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2012 = __riscv_vand_vx_u8mf2(v2011, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2013 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2012);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2014 = __riscv_vsrl_vx_u8mf2(v2011, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2015 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2014);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2016 = v32 + 312;
        const int8_t* v2017 = (const int8_t*) v2016;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2018 = *(const int8_t *)(v2017);
        const uint8_t* v2019 = v32 + 440;
        const int8_t* v2020 = (const int8_t*) v2019;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2021 = *(const int8_t *)(v2020);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2022 = __riscv_vwmacc_vx_i16m1(v1983, v2018, v2013, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2023 = __riscv_vwmacc_vx_i16m1(v1984, v2021, v2015, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2024 = v32 + 313;
        const int8_t* v2025 = (const int8_t*) v2024;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2026 = *(const int8_t *)(v2025);
        const uint8_t* v2027 = v32 + 441;
        const int8_t* v2028 = (const int8_t*) v2027;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2029 = *(const int8_t *)(v2028);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2030 = __riscv_vwmacc_vx_i16m1(v1991, v2026, v2013, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2031 = __riscv_vwmacc_vx_i16m1(v1992, v2029, v2015, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2032 = v32 + 314;
        const int8_t* v2033 = (const int8_t*) v2032;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2034 = *(const int8_t *)(v2033);
        const uint8_t* v2035 = v32 + 442;
        const int8_t* v2036 = (const int8_t*) v2035;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2037 = *(const int8_t *)(v2036);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2038 = __riscv_vwmacc_vx_i16m1(v1999, v2034, v2013, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2039 = __riscv_vwmacc_vx_i16m1(v2000, v2037, v2015, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2040 = v32 + 315;
        const int8_t* v2041 = (const int8_t*) v2040;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2042 = *(const int8_t *)(v2041);
        const uint8_t* v2043 = v32 + 443;
        const int8_t* v2044 = (const int8_t*) v2043;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2045 = *(const int8_t *)(v2044);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2046 = __riscv_vwmacc_vx_i16m1(v2007, v2042, v2013, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2047 = __riscv_vwmacc_vx_i16m1(v2008, v2045, v2015, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2048 = v30 + 944;
        const uint8_t* v2049 = (const uint8_t*) v2048;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2050 = __riscv_vle8_v_u8mf2(v2049, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2051 = __riscv_vand_vx_u8mf2(v2050, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2052 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2051);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2053 = __riscv_vsrl_vx_u8mf2(v2050, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2054 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2053);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2055 = v32 + 316;
        const int8_t* v2056 = (const int8_t*) v2055;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2057 = *(const int8_t *)(v2056);
        const uint8_t* v2058 = v32 + 444;
        const int8_t* v2059 = (const int8_t*) v2058;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2060 = *(const int8_t *)(v2059);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2061 = __riscv_vwmacc_vx_i16m1(v2022, v2057, v2052, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2062 = __riscv_vwmacc_vx_i16m1(v2023, v2060, v2054, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2063 = v32 + 317;
        const int8_t* v2064 = (const int8_t*) v2063;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2065 = *(const int8_t *)(v2064);
        const uint8_t* v2066 = v32 + 445;
        const int8_t* v2067 = (const int8_t*) v2066;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2068 = *(const int8_t *)(v2067);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2069 = __riscv_vwmacc_vx_i16m1(v2030, v2065, v2052, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2070 = __riscv_vwmacc_vx_i16m1(v2031, v2068, v2054, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2071 = v32 + 318;
        const int8_t* v2072 = (const int8_t*) v2071;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2073 = *(const int8_t *)(v2072);
        const uint8_t* v2074 = v32 + 446;
        const int8_t* v2075 = (const int8_t*) v2074;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2076 = *(const int8_t *)(v2075);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2077 = __riscv_vwmacc_vx_i16m1(v2038, v2073, v2052, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2078 = __riscv_vwmacc_vx_i16m1(v2039, v2076, v2054, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2079 = v32 + 319;
        const int8_t* v2080 = (const int8_t*) v2079;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2081 = *(const int8_t *)(v2080);
        const uint8_t* v2082 = v32 + 447;
        const int8_t* v2083 = (const int8_t*) v2082;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2084 = *(const int8_t *)(v2083);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2085 = __riscv_vwmacc_vx_i16m1(v2046, v2081, v2052, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2086 = __riscv_vwmacc_vx_i16m1(v2047, v2084, v2054, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2087 = v30 + 960;
        const uint8_t* v2088 = (const uint8_t*) v2087;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2089 = __riscv_vle8_v_u8mf2(v2088, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2090 = __riscv_vand_vx_u8mf2(v2089, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2091 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2090);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2092 = __riscv_vsrl_vx_u8mf2(v2089, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2093 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2092);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2094 = v32 + 320;
        const int8_t* v2095 = (const int8_t*) v2094;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2096 = *(const int8_t *)(v2095);
        const uint8_t* v2097 = v32 + 448;
        const int8_t* v2098 = (const int8_t*) v2097;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2099 = *(const int8_t *)(v2098);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2100 = __riscv_vwmacc_vx_i16m1(v2061, v2096, v2091, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2101 = __riscv_vwmacc_vx_i16m1(v2062, v2099, v2093, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2102 = v32 + 321;
        const int8_t* v2103 = (const int8_t*) v2102;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2104 = *(const int8_t *)(v2103);
        const uint8_t* v2105 = v32 + 449;
        const int8_t* v2106 = (const int8_t*) v2105;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2107 = *(const int8_t *)(v2106);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2108 = __riscv_vwmacc_vx_i16m1(v2069, v2104, v2091, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2109 = __riscv_vwmacc_vx_i16m1(v2070, v2107, v2093, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2110 = v32 + 322;
        const int8_t* v2111 = (const int8_t*) v2110;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2112 = *(const int8_t *)(v2111);
        const uint8_t* v2113 = v32 + 450;
        const int8_t* v2114 = (const int8_t*) v2113;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2115 = *(const int8_t *)(v2114);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2116 = __riscv_vwmacc_vx_i16m1(v2077, v2112, v2091, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2117 = __riscv_vwmacc_vx_i16m1(v2078, v2115, v2093, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2118 = v32 + 323;
        const int8_t* v2119 = (const int8_t*) v2118;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2120 = *(const int8_t *)(v2119);
        const uint8_t* v2121 = v32 + 451;
        const int8_t* v2122 = (const int8_t*) v2121;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2123 = *(const int8_t *)(v2122);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2124 = __riscv_vwmacc_vx_i16m1(v2085, v2120, v2091, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2125 = __riscv_vwmacc_vx_i16m1(v2086, v2123, v2093, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2126 = v30 + 976;
        const uint8_t* v2127 = (const uint8_t*) v2126;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2128 = __riscv_vle8_v_u8mf2(v2127, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2129 = __riscv_vand_vx_u8mf2(v2128, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2130 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2129);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2131 = __riscv_vsrl_vx_u8mf2(v2128, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2132 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2131);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2133 = v32 + 324;
        const int8_t* v2134 = (const int8_t*) v2133;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2135 = *(const int8_t *)(v2134);
        const uint8_t* v2136 = v32 + 452;
        const int8_t* v2137 = (const int8_t*) v2136;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2138 = *(const int8_t *)(v2137);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2139 = __riscv_vwmacc_vx_i16m1(v2100, v2135, v2130, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2140 = __riscv_vwmacc_vx_i16m1(v2101, v2138, v2132, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2141 = v32 + 325;
        const int8_t* v2142 = (const int8_t*) v2141;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2143 = *(const int8_t *)(v2142);
        const uint8_t* v2144 = v32 + 453;
        const int8_t* v2145 = (const int8_t*) v2144;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2146 = *(const int8_t *)(v2145);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2147 = __riscv_vwmacc_vx_i16m1(v2108, v2143, v2130, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2148 = __riscv_vwmacc_vx_i16m1(v2109, v2146, v2132, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2149 = v32 + 326;
        const int8_t* v2150 = (const int8_t*) v2149;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2151 = *(const int8_t *)(v2150);
        const uint8_t* v2152 = v32 + 454;
        const int8_t* v2153 = (const int8_t*) v2152;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2154 = *(const int8_t *)(v2153);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2155 = __riscv_vwmacc_vx_i16m1(v2116, v2151, v2130, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2156 = __riscv_vwmacc_vx_i16m1(v2117, v2154, v2132, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2157 = v32 + 327;
        const int8_t* v2158 = (const int8_t*) v2157;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2159 = *(const int8_t *)(v2158);
        const uint8_t* v2160 = v32 + 455;
        const int8_t* v2161 = (const int8_t*) v2160;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2162 = *(const int8_t *)(v2161);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2163 = __riscv_vwmacc_vx_i16m1(v2124, v2159, v2130, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2164 = __riscv_vwmacc_vx_i16m1(v2125, v2162, v2132, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2165 = v30 + 992;
        const uint8_t* v2166 = (const uint8_t*) v2165;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2167 = __riscv_vle8_v_u8mf2(v2166, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2168 = __riscv_vand_vx_u8mf2(v2167, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2169 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2168);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2170 = __riscv_vsrl_vx_u8mf2(v2167, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2171 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2170);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2172 = v32 + 328;
        const int8_t* v2173 = (const int8_t*) v2172;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2174 = *(const int8_t *)(v2173);
        const uint8_t* v2175 = v32 + 456;
        const int8_t* v2176 = (const int8_t*) v2175;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2177 = *(const int8_t *)(v2176);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2178 = __riscv_vwmacc_vx_i16m1(v2139, v2174, v2169, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2179 = __riscv_vwmacc_vx_i16m1(v2140, v2177, v2171, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2180 = v32 + 329;
        const int8_t* v2181 = (const int8_t*) v2180;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2182 = *(const int8_t *)(v2181);
        const uint8_t* v2183 = v32 + 457;
        const int8_t* v2184 = (const int8_t*) v2183;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2185 = *(const int8_t *)(v2184);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2186 = __riscv_vwmacc_vx_i16m1(v2147, v2182, v2169, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2187 = __riscv_vwmacc_vx_i16m1(v2148, v2185, v2171, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2188 = v32 + 330;
        const int8_t* v2189 = (const int8_t*) v2188;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2190 = *(const int8_t *)(v2189);
        const uint8_t* v2191 = v32 + 458;
        const int8_t* v2192 = (const int8_t*) v2191;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2193 = *(const int8_t *)(v2192);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2194 = __riscv_vwmacc_vx_i16m1(v2155, v2190, v2169, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2195 = __riscv_vwmacc_vx_i16m1(v2156, v2193, v2171, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2196 = v32 + 331;
        const int8_t* v2197 = (const int8_t*) v2196;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2198 = *(const int8_t *)(v2197);
        const uint8_t* v2199 = v32 + 459;
        const int8_t* v2200 = (const int8_t*) v2199;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2201 = *(const int8_t *)(v2200);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2202 = __riscv_vwmacc_vx_i16m1(v2163, v2198, v2169, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2203 = __riscv_vwmacc_vx_i16m1(v2164, v2201, v2171, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2204 = v30 + 1008;
        const uint8_t* v2205 = (const uint8_t*) v2204;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2206 = __riscv_vle8_v_u8mf2(v2205, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2207 = __riscv_vand_vx_u8mf2(v2206, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2208 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2207);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2209 = __riscv_vsrl_vx_u8mf2(v2206, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2210 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2209);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2211 = v32 + 332;
        const int8_t* v2212 = (const int8_t*) v2211;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2213 = *(const int8_t *)(v2212);
        const uint8_t* v2214 = v32 + 460;
        const int8_t* v2215 = (const int8_t*) v2214;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2216 = *(const int8_t *)(v2215);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2217 = __riscv_vwmacc_vx_i16m1(v2178, v2213, v2208, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2218 = __riscv_vwmacc_vx_i16m1(v2179, v2216, v2210, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2219 = v32 + 333;
        const int8_t* v2220 = (const int8_t*) v2219;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2221 = *(const int8_t *)(v2220);
        const uint8_t* v2222 = v32 + 461;
        const int8_t* v2223 = (const int8_t*) v2222;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2224 = *(const int8_t *)(v2223);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2225 = __riscv_vwmacc_vx_i16m1(v2186, v2221, v2208, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2226 = __riscv_vwmacc_vx_i16m1(v2187, v2224, v2210, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2227 = v32 + 334;
        const int8_t* v2228 = (const int8_t*) v2227;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2229 = *(const int8_t *)(v2228);
        const uint8_t* v2230 = v32 + 462;
        const int8_t* v2231 = (const int8_t*) v2230;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2232 = *(const int8_t *)(v2231);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2233 = __riscv_vwmacc_vx_i16m1(v2194, v2229, v2208, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2234 = __riscv_vwmacc_vx_i16m1(v2195, v2232, v2210, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2235 = v32 + 335;
        const int8_t* v2236 = (const int8_t*) v2235;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2237 = *(const int8_t *)(v2236);
        const uint8_t* v2238 = v32 + 463;
        const int8_t* v2239 = (const int8_t*) v2238;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2240 = *(const int8_t *)(v2239);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2241 = __riscv_vwmacc_vx_i16m1(v2202, v2237, v2208, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2242 = __riscv_vwmacc_vx_i16m1(v2203, v2240, v2210, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v2243 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v2244 = __riscv_vle16_v_i16m1(v2243, 16);
        int16_t* v2245 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v2246 = __riscv_vle16_v_i16m1(v2245, 16);
        vint32m2_t v2247 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2248 = __riscv_vwmacc_vv_i32m2(v2247, v2244, v2217, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2249 = __riscv_vwmacc_vv_i32m2(v2248, v2246, v2218, 16);
        v47 = v2249;
        vint32m2_t v2250 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2251 = __riscv_vwmacc_vv_i32m2(v2250, v2244, v2225, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2252 = __riscv_vwmacc_vv_i32m2(v2251, v2246, v2226, 16);
        v49 = v2252;
        vint32m2_t v2253 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2254 = __riscv_vwmacc_vv_i32m2(v2253, v2244, v2233, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2255 = __riscv_vwmacc_vv_i32m2(v2254, v2246, v2234, 16);
        v51 = v2255;
        vint32m2_t v2256 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2257 = __riscv_vwmacc_vv_i32m2(v2256, v2244, v2241, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2258 = __riscv_vwmacc_vv_i32m2(v2257, v2246, v2242, 16);
        v53 = v2258;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2259 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2260 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2261 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2262 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2263 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2264 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2265 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v2266 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2267 = v30 + 1024;
        const uint8_t* v2268 = (const uint8_t*) v2267;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2269 = __riscv_vle8_v_u8mf2(v2268, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2270 = __riscv_vand_vx_u8mf2(v2269, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2271 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2270);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2272 = __riscv_vsrl_vx_u8mf2(v2269, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2273 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2272);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2274 = v32 + 336;
        const int8_t* v2275 = (const int8_t*) v2274;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2276 = *(const int8_t *)(v2275);
        const uint8_t* v2277 = v32 + 464;
        const int8_t* v2278 = (const int8_t*) v2277;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2279 = *(const int8_t *)(v2278);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2280 = __riscv_vwmacc_vx_i16m1(v2259, v2276, v2271, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2281 = __riscv_vwmacc_vx_i16m1(v2260, v2279, v2273, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2282 = v32 + 337;
        const int8_t* v2283 = (const int8_t*) v2282;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2284 = *(const int8_t *)(v2283);
        const uint8_t* v2285 = v32 + 465;
        const int8_t* v2286 = (const int8_t*) v2285;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2287 = *(const int8_t *)(v2286);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2288 = __riscv_vwmacc_vx_i16m1(v2261, v2284, v2271, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2289 = __riscv_vwmacc_vx_i16m1(v2262, v2287, v2273, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2290 = v32 + 338;
        const int8_t* v2291 = (const int8_t*) v2290;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2292 = *(const int8_t *)(v2291);
        const uint8_t* v2293 = v32 + 466;
        const int8_t* v2294 = (const int8_t*) v2293;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2295 = *(const int8_t *)(v2294);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2296 = __riscv_vwmacc_vx_i16m1(v2263, v2292, v2271, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2297 = __riscv_vwmacc_vx_i16m1(v2264, v2295, v2273, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2298 = v32 + 339;
        const int8_t* v2299 = (const int8_t*) v2298;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2300 = *(const int8_t *)(v2299);
        const uint8_t* v2301 = v32 + 467;
        const int8_t* v2302 = (const int8_t*) v2301;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2303 = *(const int8_t *)(v2302);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2304 = __riscv_vwmacc_vx_i16m1(v2265, v2300, v2271, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2305 = __riscv_vwmacc_vx_i16m1(v2266, v2303, v2273, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2306 = v30 + 1040;
        const uint8_t* v2307 = (const uint8_t*) v2306;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2308 = __riscv_vle8_v_u8mf2(v2307, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2309 = __riscv_vand_vx_u8mf2(v2308, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2310 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2309);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2311 = __riscv_vsrl_vx_u8mf2(v2308, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2312 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2311);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2313 = v32 + 340;
        const int8_t* v2314 = (const int8_t*) v2313;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2315 = *(const int8_t *)(v2314);
        const uint8_t* v2316 = v32 + 468;
        const int8_t* v2317 = (const int8_t*) v2316;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2318 = *(const int8_t *)(v2317);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2319 = __riscv_vwmacc_vx_i16m1(v2280, v2315, v2310, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2320 = __riscv_vwmacc_vx_i16m1(v2281, v2318, v2312, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2321 = v32 + 341;
        const int8_t* v2322 = (const int8_t*) v2321;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2323 = *(const int8_t *)(v2322);
        const uint8_t* v2324 = v32 + 469;
        const int8_t* v2325 = (const int8_t*) v2324;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2326 = *(const int8_t *)(v2325);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2327 = __riscv_vwmacc_vx_i16m1(v2288, v2323, v2310, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2328 = __riscv_vwmacc_vx_i16m1(v2289, v2326, v2312, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2329 = v32 + 342;
        const int8_t* v2330 = (const int8_t*) v2329;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2331 = *(const int8_t *)(v2330);
        const uint8_t* v2332 = v32 + 470;
        const int8_t* v2333 = (const int8_t*) v2332;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2334 = *(const int8_t *)(v2333);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2335 = __riscv_vwmacc_vx_i16m1(v2296, v2331, v2310, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2336 = __riscv_vwmacc_vx_i16m1(v2297, v2334, v2312, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2337 = v32 + 343;
        const int8_t* v2338 = (const int8_t*) v2337;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2339 = *(const int8_t *)(v2338);
        const uint8_t* v2340 = v32 + 471;
        const int8_t* v2341 = (const int8_t*) v2340;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2342 = *(const int8_t *)(v2341);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2343 = __riscv_vwmacc_vx_i16m1(v2304, v2339, v2310, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2344 = __riscv_vwmacc_vx_i16m1(v2305, v2342, v2312, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2345 = v30 + 1056;
        const uint8_t* v2346 = (const uint8_t*) v2345;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2347 = __riscv_vle8_v_u8mf2(v2346, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2348 = __riscv_vand_vx_u8mf2(v2347, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2349 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2348);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2350 = __riscv_vsrl_vx_u8mf2(v2347, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2351 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2350);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2352 = v32 + 344;
        const int8_t* v2353 = (const int8_t*) v2352;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2354 = *(const int8_t *)(v2353);
        const uint8_t* v2355 = v32 + 472;
        const int8_t* v2356 = (const int8_t*) v2355;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2357 = *(const int8_t *)(v2356);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2358 = __riscv_vwmacc_vx_i16m1(v2319, v2354, v2349, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2359 = __riscv_vwmacc_vx_i16m1(v2320, v2357, v2351, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2360 = v32 + 345;
        const int8_t* v2361 = (const int8_t*) v2360;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2362 = *(const int8_t *)(v2361);
        const uint8_t* v2363 = v32 + 473;
        const int8_t* v2364 = (const int8_t*) v2363;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2365 = *(const int8_t *)(v2364);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2366 = __riscv_vwmacc_vx_i16m1(v2327, v2362, v2349, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2367 = __riscv_vwmacc_vx_i16m1(v2328, v2365, v2351, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2368 = v32 + 346;
        const int8_t* v2369 = (const int8_t*) v2368;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2370 = *(const int8_t *)(v2369);
        const uint8_t* v2371 = v32 + 474;
        const int8_t* v2372 = (const int8_t*) v2371;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2373 = *(const int8_t *)(v2372);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2374 = __riscv_vwmacc_vx_i16m1(v2335, v2370, v2349, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2375 = __riscv_vwmacc_vx_i16m1(v2336, v2373, v2351, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2376 = v32 + 347;
        const int8_t* v2377 = (const int8_t*) v2376;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2378 = *(const int8_t *)(v2377);
        const uint8_t* v2379 = v32 + 475;
        const int8_t* v2380 = (const int8_t*) v2379;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2381 = *(const int8_t *)(v2380);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2382 = __riscv_vwmacc_vx_i16m1(v2343, v2378, v2349, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2383 = __riscv_vwmacc_vx_i16m1(v2344, v2381, v2351, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2384 = v30 + 1072;
        const uint8_t* v2385 = (const uint8_t*) v2384;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2386 = __riscv_vle8_v_u8mf2(v2385, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2387 = __riscv_vand_vx_u8mf2(v2386, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2388 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2387);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2389 = __riscv_vsrl_vx_u8mf2(v2386, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2390 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2389);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2391 = v32 + 348;
        const int8_t* v2392 = (const int8_t*) v2391;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2393 = *(const int8_t *)(v2392);
        const uint8_t* v2394 = v32 + 476;
        const int8_t* v2395 = (const int8_t*) v2394;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2396 = *(const int8_t *)(v2395);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2397 = __riscv_vwmacc_vx_i16m1(v2358, v2393, v2388, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2398 = __riscv_vwmacc_vx_i16m1(v2359, v2396, v2390, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2399 = v32 + 349;
        const int8_t* v2400 = (const int8_t*) v2399;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2401 = *(const int8_t *)(v2400);
        const uint8_t* v2402 = v32 + 477;
        const int8_t* v2403 = (const int8_t*) v2402;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2404 = *(const int8_t *)(v2403);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2405 = __riscv_vwmacc_vx_i16m1(v2366, v2401, v2388, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2406 = __riscv_vwmacc_vx_i16m1(v2367, v2404, v2390, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2407 = v32 + 350;
        const int8_t* v2408 = (const int8_t*) v2407;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2409 = *(const int8_t *)(v2408);
        const uint8_t* v2410 = v32 + 478;
        const int8_t* v2411 = (const int8_t*) v2410;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2412 = *(const int8_t *)(v2411);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2413 = __riscv_vwmacc_vx_i16m1(v2374, v2409, v2388, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2414 = __riscv_vwmacc_vx_i16m1(v2375, v2412, v2390, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2415 = v32 + 351;
        const int8_t* v2416 = (const int8_t*) v2415;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2417 = *(const int8_t *)(v2416);
        const uint8_t* v2418 = v32 + 479;
        const int8_t* v2419 = (const int8_t*) v2418;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2420 = *(const int8_t *)(v2419);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2421 = __riscv_vwmacc_vx_i16m1(v2382, v2417, v2388, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2422 = __riscv_vwmacc_vx_i16m1(v2383, v2420, v2390, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2423 = v30 + 1088;
        const uint8_t* v2424 = (const uint8_t*) v2423;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2425 = __riscv_vle8_v_u8mf2(v2424, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2426 = __riscv_vand_vx_u8mf2(v2425, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2427 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2426);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2428 = __riscv_vsrl_vx_u8mf2(v2425, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2429 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2428);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2430 = v32 + 352;
        const int8_t* v2431 = (const int8_t*) v2430;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2432 = *(const int8_t *)(v2431);
        const uint8_t* v2433 = v32 + 480;
        const int8_t* v2434 = (const int8_t*) v2433;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2435 = *(const int8_t *)(v2434);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2436 = __riscv_vwmacc_vx_i16m1(v2397, v2432, v2427, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2437 = __riscv_vwmacc_vx_i16m1(v2398, v2435, v2429, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2438 = v32 + 353;
        const int8_t* v2439 = (const int8_t*) v2438;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2440 = *(const int8_t *)(v2439);
        const uint8_t* v2441 = v32 + 481;
        const int8_t* v2442 = (const int8_t*) v2441;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2443 = *(const int8_t *)(v2442);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2444 = __riscv_vwmacc_vx_i16m1(v2405, v2440, v2427, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2445 = __riscv_vwmacc_vx_i16m1(v2406, v2443, v2429, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2446 = v32 + 354;
        const int8_t* v2447 = (const int8_t*) v2446;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2448 = *(const int8_t *)(v2447);
        const uint8_t* v2449 = v32 + 482;
        const int8_t* v2450 = (const int8_t*) v2449;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2451 = *(const int8_t *)(v2450);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2452 = __riscv_vwmacc_vx_i16m1(v2413, v2448, v2427, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2453 = __riscv_vwmacc_vx_i16m1(v2414, v2451, v2429, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2454 = v32 + 355;
        const int8_t* v2455 = (const int8_t*) v2454;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2456 = *(const int8_t *)(v2455);
        const uint8_t* v2457 = v32 + 483;
        const int8_t* v2458 = (const int8_t*) v2457;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2459 = *(const int8_t *)(v2458);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2460 = __riscv_vwmacc_vx_i16m1(v2421, v2456, v2427, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2461 = __riscv_vwmacc_vx_i16m1(v2422, v2459, v2429, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2462 = v30 + 1104;
        const uint8_t* v2463 = (const uint8_t*) v2462;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2464 = __riscv_vle8_v_u8mf2(v2463, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2465 = __riscv_vand_vx_u8mf2(v2464, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2466 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2465);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2467 = __riscv_vsrl_vx_u8mf2(v2464, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2468 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2467);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2469 = v32 + 356;
        const int8_t* v2470 = (const int8_t*) v2469;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2471 = *(const int8_t *)(v2470);
        const uint8_t* v2472 = v32 + 484;
        const int8_t* v2473 = (const int8_t*) v2472;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2474 = *(const int8_t *)(v2473);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2475 = __riscv_vwmacc_vx_i16m1(v2436, v2471, v2466, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2476 = __riscv_vwmacc_vx_i16m1(v2437, v2474, v2468, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2477 = v32 + 357;
        const int8_t* v2478 = (const int8_t*) v2477;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2479 = *(const int8_t *)(v2478);
        const uint8_t* v2480 = v32 + 485;
        const int8_t* v2481 = (const int8_t*) v2480;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2482 = *(const int8_t *)(v2481);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2483 = __riscv_vwmacc_vx_i16m1(v2444, v2479, v2466, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2484 = __riscv_vwmacc_vx_i16m1(v2445, v2482, v2468, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2485 = v32 + 358;
        const int8_t* v2486 = (const int8_t*) v2485;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2487 = *(const int8_t *)(v2486);
        const uint8_t* v2488 = v32 + 486;
        const int8_t* v2489 = (const int8_t*) v2488;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2490 = *(const int8_t *)(v2489);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2491 = __riscv_vwmacc_vx_i16m1(v2452, v2487, v2466, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2492 = __riscv_vwmacc_vx_i16m1(v2453, v2490, v2468, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2493 = v32 + 359;
        const int8_t* v2494 = (const int8_t*) v2493;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2495 = *(const int8_t *)(v2494);
        const uint8_t* v2496 = v32 + 487;
        const int8_t* v2497 = (const int8_t*) v2496;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2498 = *(const int8_t *)(v2497);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2499 = __riscv_vwmacc_vx_i16m1(v2460, v2495, v2466, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2500 = __riscv_vwmacc_vx_i16m1(v2461, v2498, v2468, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2501 = v30 + 1120;
        const uint8_t* v2502 = (const uint8_t*) v2501;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2503 = __riscv_vle8_v_u8mf2(v2502, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2504 = __riscv_vand_vx_u8mf2(v2503, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2505 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2504);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2506 = __riscv_vsrl_vx_u8mf2(v2503, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2507 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2506);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2508 = v32 + 360;
        const int8_t* v2509 = (const int8_t*) v2508;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2510 = *(const int8_t *)(v2509);
        const uint8_t* v2511 = v32 + 488;
        const int8_t* v2512 = (const int8_t*) v2511;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2513 = *(const int8_t *)(v2512);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2514 = __riscv_vwmacc_vx_i16m1(v2475, v2510, v2505, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2515 = __riscv_vwmacc_vx_i16m1(v2476, v2513, v2507, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2516 = v32 + 361;
        const int8_t* v2517 = (const int8_t*) v2516;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2518 = *(const int8_t *)(v2517);
        const uint8_t* v2519 = v32 + 489;
        const int8_t* v2520 = (const int8_t*) v2519;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2521 = *(const int8_t *)(v2520);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2522 = __riscv_vwmacc_vx_i16m1(v2483, v2518, v2505, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2523 = __riscv_vwmacc_vx_i16m1(v2484, v2521, v2507, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2524 = v32 + 362;
        const int8_t* v2525 = (const int8_t*) v2524;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2526 = *(const int8_t *)(v2525);
        const uint8_t* v2527 = v32 + 490;
        const int8_t* v2528 = (const int8_t*) v2527;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2529 = *(const int8_t *)(v2528);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2530 = __riscv_vwmacc_vx_i16m1(v2491, v2526, v2505, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2531 = __riscv_vwmacc_vx_i16m1(v2492, v2529, v2507, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2532 = v32 + 363;
        const int8_t* v2533 = (const int8_t*) v2532;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2534 = *(const int8_t *)(v2533);
        const uint8_t* v2535 = v32 + 491;
        const int8_t* v2536 = (const int8_t*) v2535;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2537 = *(const int8_t *)(v2536);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2538 = __riscv_vwmacc_vx_i16m1(v2499, v2534, v2505, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2539 = __riscv_vwmacc_vx_i16m1(v2500, v2537, v2507, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2540 = v30 + 1136;
        const uint8_t* v2541 = (const uint8_t*) v2540;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2542 = __riscv_vle8_v_u8mf2(v2541, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2543 = __riscv_vand_vx_u8mf2(v2542, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2544 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2543);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2545 = __riscv_vsrl_vx_u8mf2(v2542, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2546 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2545);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2547 = v32 + 364;
        const int8_t* v2548 = (const int8_t*) v2547;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2549 = *(const int8_t *)(v2548);
        const uint8_t* v2550 = v32 + 492;
        const int8_t* v2551 = (const int8_t*) v2550;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2552 = *(const int8_t *)(v2551);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2553 = __riscv_vwmacc_vx_i16m1(v2514, v2549, v2544, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2554 = __riscv_vwmacc_vx_i16m1(v2515, v2552, v2546, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2555 = v32 + 365;
        const int8_t* v2556 = (const int8_t*) v2555;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2557 = *(const int8_t *)(v2556);
        const uint8_t* v2558 = v32 + 493;
        const int8_t* v2559 = (const int8_t*) v2558;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2560 = *(const int8_t *)(v2559);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2561 = __riscv_vwmacc_vx_i16m1(v2522, v2557, v2544, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2562 = __riscv_vwmacc_vx_i16m1(v2523, v2560, v2546, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2563 = v32 + 366;
        const int8_t* v2564 = (const int8_t*) v2563;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2565 = *(const int8_t *)(v2564);
        const uint8_t* v2566 = v32 + 494;
        const int8_t* v2567 = (const int8_t*) v2566;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2568 = *(const int8_t *)(v2567);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2569 = __riscv_vwmacc_vx_i16m1(v2530, v2565, v2544, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2570 = __riscv_vwmacc_vx_i16m1(v2531, v2568, v2546, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2571 = v32 + 367;
        const int8_t* v2572 = (const int8_t*) v2571;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2573 = *(const int8_t *)(v2572);
        const uint8_t* v2574 = v32 + 495;
        const int8_t* v2575 = (const int8_t*) v2574;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2576 = *(const int8_t *)(v2575);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2577 = __riscv_vwmacc_vx_i16m1(v2538, v2573, v2544, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2578 = __riscv_vwmacc_vx_i16m1(v2539, v2576, v2546, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2579 = v30 + 1152;
        const uint8_t* v2580 = (const uint8_t*) v2579;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2581 = __riscv_vle8_v_u8mf2(v2580, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2582 = __riscv_vand_vx_u8mf2(v2581, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2583 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2582);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2584 = __riscv_vsrl_vx_u8mf2(v2581, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2585 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2584);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2586 = v32 + 368;
        const int8_t* v2587 = (const int8_t*) v2586;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2588 = *(const int8_t *)(v2587);
        const uint8_t* v2589 = v32 + 496;
        const int8_t* v2590 = (const int8_t*) v2589;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2591 = *(const int8_t *)(v2590);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2592 = __riscv_vwmacc_vx_i16m1(v2553, v2588, v2583, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2593 = __riscv_vwmacc_vx_i16m1(v2554, v2591, v2585, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2594 = v32 + 369;
        const int8_t* v2595 = (const int8_t*) v2594;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2596 = *(const int8_t *)(v2595);
        const uint8_t* v2597 = v32 + 497;
        const int8_t* v2598 = (const int8_t*) v2597;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2599 = *(const int8_t *)(v2598);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2600 = __riscv_vwmacc_vx_i16m1(v2561, v2596, v2583, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2601 = __riscv_vwmacc_vx_i16m1(v2562, v2599, v2585, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2602 = v32 + 370;
        const int8_t* v2603 = (const int8_t*) v2602;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2604 = *(const int8_t *)(v2603);
        const uint8_t* v2605 = v32 + 498;
        const int8_t* v2606 = (const int8_t*) v2605;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2607 = *(const int8_t *)(v2606);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2608 = __riscv_vwmacc_vx_i16m1(v2569, v2604, v2583, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2609 = __riscv_vwmacc_vx_i16m1(v2570, v2607, v2585, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2610 = v32 + 371;
        const int8_t* v2611 = (const int8_t*) v2610;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2612 = *(const int8_t *)(v2611);
        const uint8_t* v2613 = v32 + 499;
        const int8_t* v2614 = (const int8_t*) v2613;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2615 = *(const int8_t *)(v2614);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2616 = __riscv_vwmacc_vx_i16m1(v2577, v2612, v2583, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2617 = __riscv_vwmacc_vx_i16m1(v2578, v2615, v2585, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2618 = v30 + 1168;
        const uint8_t* v2619 = (const uint8_t*) v2618;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2620 = __riscv_vle8_v_u8mf2(v2619, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2621 = __riscv_vand_vx_u8mf2(v2620, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2622 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2621);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2623 = __riscv_vsrl_vx_u8mf2(v2620, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2624 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2623);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2625 = v32 + 372;
        const int8_t* v2626 = (const int8_t*) v2625;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2627 = *(const int8_t *)(v2626);
        const uint8_t* v2628 = v32 + 500;
        const int8_t* v2629 = (const int8_t*) v2628;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2630 = *(const int8_t *)(v2629);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2631 = __riscv_vwmacc_vx_i16m1(v2592, v2627, v2622, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2632 = __riscv_vwmacc_vx_i16m1(v2593, v2630, v2624, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2633 = v32 + 373;
        const int8_t* v2634 = (const int8_t*) v2633;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2635 = *(const int8_t *)(v2634);
        const uint8_t* v2636 = v32 + 501;
        const int8_t* v2637 = (const int8_t*) v2636;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2638 = *(const int8_t *)(v2637);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2639 = __riscv_vwmacc_vx_i16m1(v2600, v2635, v2622, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2640 = __riscv_vwmacc_vx_i16m1(v2601, v2638, v2624, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2641 = v32 + 374;
        const int8_t* v2642 = (const int8_t*) v2641;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2643 = *(const int8_t *)(v2642);
        const uint8_t* v2644 = v32 + 502;
        const int8_t* v2645 = (const int8_t*) v2644;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2646 = *(const int8_t *)(v2645);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2647 = __riscv_vwmacc_vx_i16m1(v2608, v2643, v2622, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2648 = __riscv_vwmacc_vx_i16m1(v2609, v2646, v2624, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2649 = v32 + 375;
        const int8_t* v2650 = (const int8_t*) v2649;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2651 = *(const int8_t *)(v2650);
        const uint8_t* v2652 = v32 + 503;
        const int8_t* v2653 = (const int8_t*) v2652;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2654 = *(const int8_t *)(v2653);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2655 = __riscv_vwmacc_vx_i16m1(v2616, v2651, v2622, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2656 = __riscv_vwmacc_vx_i16m1(v2617, v2654, v2624, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2657 = v30 + 1184;
        const uint8_t* v2658 = (const uint8_t*) v2657;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2659 = __riscv_vle8_v_u8mf2(v2658, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2660 = __riscv_vand_vx_u8mf2(v2659, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2661 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2660);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2662 = __riscv_vsrl_vx_u8mf2(v2659, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2663 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2662);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2664 = v32 + 376;
        const int8_t* v2665 = (const int8_t*) v2664;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2666 = *(const int8_t *)(v2665);
        const uint8_t* v2667 = v32 + 504;
        const int8_t* v2668 = (const int8_t*) v2667;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2669 = *(const int8_t *)(v2668);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2670 = __riscv_vwmacc_vx_i16m1(v2631, v2666, v2661, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2671 = __riscv_vwmacc_vx_i16m1(v2632, v2669, v2663, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2672 = v32 + 377;
        const int8_t* v2673 = (const int8_t*) v2672;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2674 = *(const int8_t *)(v2673);
        const uint8_t* v2675 = v32 + 505;
        const int8_t* v2676 = (const int8_t*) v2675;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2677 = *(const int8_t *)(v2676);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2678 = __riscv_vwmacc_vx_i16m1(v2639, v2674, v2661, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2679 = __riscv_vwmacc_vx_i16m1(v2640, v2677, v2663, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2680 = v32 + 378;
        const int8_t* v2681 = (const int8_t*) v2680;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2682 = *(const int8_t *)(v2681);
        const uint8_t* v2683 = v32 + 506;
        const int8_t* v2684 = (const int8_t*) v2683;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2685 = *(const int8_t *)(v2684);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2686 = __riscv_vwmacc_vx_i16m1(v2647, v2682, v2661, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2687 = __riscv_vwmacc_vx_i16m1(v2648, v2685, v2663, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2688 = v32 + 379;
        const int8_t* v2689 = (const int8_t*) v2688;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2690 = *(const int8_t *)(v2689);
        const uint8_t* v2691 = v32 + 507;
        const int8_t* v2692 = (const int8_t*) v2691;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2693 = *(const int8_t *)(v2692);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2694 = __riscv_vwmacc_vx_i16m1(v2655, v2690, v2661, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2695 = __riscv_vwmacc_vx_i16m1(v2656, v2693, v2663, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2696 = v30 + 1200;
        const uint8_t* v2697 = (const uint8_t*) v2696;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2698 = __riscv_vle8_v_u8mf2(v2697, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2699 = __riscv_vand_vx_u8mf2(v2698, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2700 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2699);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2701 = __riscv_vsrl_vx_u8mf2(v2698, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2702 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2701);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2703 = v32 + 380;
        const int8_t* v2704 = (const int8_t*) v2703;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2705 = *(const int8_t *)(v2704);
        const uint8_t* v2706 = v32 + 508;
        const int8_t* v2707 = (const int8_t*) v2706;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2708 = *(const int8_t *)(v2707);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2709 = __riscv_vwmacc_vx_i16m1(v2670, v2705, v2700, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2710 = __riscv_vwmacc_vx_i16m1(v2671, v2708, v2702, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2711 = v32 + 381;
        const int8_t* v2712 = (const int8_t*) v2711;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2713 = *(const int8_t *)(v2712);
        const uint8_t* v2714 = v32 + 509;
        const int8_t* v2715 = (const int8_t*) v2714;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2716 = *(const int8_t *)(v2715);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2717 = __riscv_vwmacc_vx_i16m1(v2678, v2713, v2700, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2718 = __riscv_vwmacc_vx_i16m1(v2679, v2716, v2702, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2719 = v32 + 382;
        const int8_t* v2720 = (const int8_t*) v2719;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2721 = *(const int8_t *)(v2720);
        const uint8_t* v2722 = v32 + 510;
        const int8_t* v2723 = (const int8_t*) v2722;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2724 = *(const int8_t *)(v2723);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2725 = __riscv_vwmacc_vx_i16m1(v2686, v2721, v2700, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2726 = __riscv_vwmacc_vx_i16m1(v2687, v2724, v2702, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2727 = v32 + 383;
        const int8_t* v2728 = (const int8_t*) v2727;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2729 = *(const int8_t *)(v2728);
        const uint8_t* v2730 = v32 + 511;
        const int8_t* v2731 = (const int8_t*) v2730;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2732 = *(const int8_t *)(v2731);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2733 = __riscv_vwmacc_vx_i16m1(v2694, v2729, v2700, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2734 = __riscv_vwmacc_vx_i16m1(v2695, v2732, v2702, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2735 = v30 + 1216;
        const uint8_t* v2736 = (const uint8_t*) v2735;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2737 = __riscv_vle8_v_u8mf2(v2736, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2738 = __riscv_vand_vx_u8mf2(v2737, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2739 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2738);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2740 = __riscv_vsrl_vx_u8mf2(v2737, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2741 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2740);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2742 = v32 + 384;
        const int8_t* v2743 = (const int8_t*) v2742;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2744 = *(const int8_t *)(v2743);
        const uint8_t* v2745 = v32 + 512;
        const int8_t* v2746 = (const int8_t*) v2745;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2747 = *(const int8_t *)(v2746);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2748 = __riscv_vwmacc_vx_i16m1(v2709, v2744, v2739, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2749 = __riscv_vwmacc_vx_i16m1(v2710, v2747, v2741, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2750 = v32 + 385;
        const int8_t* v2751 = (const int8_t*) v2750;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2752 = *(const int8_t *)(v2751);
        const uint8_t* v2753 = v32 + 513;
        const int8_t* v2754 = (const int8_t*) v2753;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2755 = *(const int8_t *)(v2754);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2756 = __riscv_vwmacc_vx_i16m1(v2717, v2752, v2739, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2757 = __riscv_vwmacc_vx_i16m1(v2718, v2755, v2741, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2758 = v32 + 386;
        const int8_t* v2759 = (const int8_t*) v2758;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2760 = *(const int8_t *)(v2759);
        const uint8_t* v2761 = v32 + 514;
        const int8_t* v2762 = (const int8_t*) v2761;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2763 = *(const int8_t *)(v2762);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2764 = __riscv_vwmacc_vx_i16m1(v2725, v2760, v2739, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2765 = __riscv_vwmacc_vx_i16m1(v2726, v2763, v2741, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2766 = v32 + 387;
        const int8_t* v2767 = (const int8_t*) v2766;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2768 = *(const int8_t *)(v2767);
        const uint8_t* v2769 = v32 + 515;
        const int8_t* v2770 = (const int8_t*) v2769;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2771 = *(const int8_t *)(v2770);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2772 = __riscv_vwmacc_vx_i16m1(v2733, v2768, v2739, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2773 = __riscv_vwmacc_vx_i16m1(v2734, v2771, v2741, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2774 = v30 + 1232;
        const uint8_t* v2775 = (const uint8_t*) v2774;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2776 = __riscv_vle8_v_u8mf2(v2775, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2777 = __riscv_vand_vx_u8mf2(v2776, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2778 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2777);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2779 = __riscv_vsrl_vx_u8mf2(v2776, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2780 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2779);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2781 = v32 + 388;
        const int8_t* v2782 = (const int8_t*) v2781;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2783 = *(const int8_t *)(v2782);
        const uint8_t* v2784 = v32 + 516;
        const int8_t* v2785 = (const int8_t*) v2784;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2786 = *(const int8_t *)(v2785);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2787 = __riscv_vwmacc_vx_i16m1(v2748, v2783, v2778, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2788 = __riscv_vwmacc_vx_i16m1(v2749, v2786, v2780, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2789 = v32 + 389;
        const int8_t* v2790 = (const int8_t*) v2789;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2791 = *(const int8_t *)(v2790);
        const uint8_t* v2792 = v32 + 517;
        const int8_t* v2793 = (const int8_t*) v2792;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2794 = *(const int8_t *)(v2793);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2795 = __riscv_vwmacc_vx_i16m1(v2756, v2791, v2778, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2796 = __riscv_vwmacc_vx_i16m1(v2757, v2794, v2780, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2797 = v32 + 390;
        const int8_t* v2798 = (const int8_t*) v2797;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2799 = *(const int8_t *)(v2798);
        const uint8_t* v2800 = v32 + 518;
        const int8_t* v2801 = (const int8_t*) v2800;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2802 = *(const int8_t *)(v2801);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2803 = __riscv_vwmacc_vx_i16m1(v2764, v2799, v2778, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2804 = __riscv_vwmacc_vx_i16m1(v2765, v2802, v2780, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2805 = v32 + 391;
        const int8_t* v2806 = (const int8_t*) v2805;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2807 = *(const int8_t *)(v2806);
        const uint8_t* v2808 = v32 + 519;
        const int8_t* v2809 = (const int8_t*) v2808;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2810 = *(const int8_t *)(v2809);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2811 = __riscv_vwmacc_vx_i16m1(v2772, v2807, v2778, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2812 = __riscv_vwmacc_vx_i16m1(v2773, v2810, v2780, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2813 = v30 + 1248;
        const uint8_t* v2814 = (const uint8_t*) v2813;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2815 = __riscv_vle8_v_u8mf2(v2814, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2816 = __riscv_vand_vx_u8mf2(v2815, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2817 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2816);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2818 = __riscv_vsrl_vx_u8mf2(v2815, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2819 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2818);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2820 = v32 + 392;
        const int8_t* v2821 = (const int8_t*) v2820;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2822 = *(const int8_t *)(v2821);
        const uint8_t* v2823 = v32 + 520;
        const int8_t* v2824 = (const int8_t*) v2823;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2825 = *(const int8_t *)(v2824);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2826 = __riscv_vwmacc_vx_i16m1(v2787, v2822, v2817, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2827 = __riscv_vwmacc_vx_i16m1(v2788, v2825, v2819, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2828 = v32 + 393;
        const int8_t* v2829 = (const int8_t*) v2828;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2830 = *(const int8_t *)(v2829);
        const uint8_t* v2831 = v32 + 521;
        const int8_t* v2832 = (const int8_t*) v2831;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2833 = *(const int8_t *)(v2832);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2834 = __riscv_vwmacc_vx_i16m1(v2795, v2830, v2817, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2835 = __riscv_vwmacc_vx_i16m1(v2796, v2833, v2819, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2836 = v32 + 394;
        const int8_t* v2837 = (const int8_t*) v2836;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2838 = *(const int8_t *)(v2837);
        const uint8_t* v2839 = v32 + 522;
        const int8_t* v2840 = (const int8_t*) v2839;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2841 = *(const int8_t *)(v2840);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2842 = __riscv_vwmacc_vx_i16m1(v2803, v2838, v2817, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2843 = __riscv_vwmacc_vx_i16m1(v2804, v2841, v2819, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2844 = v32 + 395;
        const int8_t* v2845 = (const int8_t*) v2844;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2846 = *(const int8_t *)(v2845);
        const uint8_t* v2847 = v32 + 523;
        const int8_t* v2848 = (const int8_t*) v2847;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2849 = *(const int8_t *)(v2848);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2850 = __riscv_vwmacc_vx_i16m1(v2811, v2846, v2817, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2851 = __riscv_vwmacc_vx_i16m1(v2812, v2849, v2819, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v2852 = v30 + 1264;
        const uint8_t* v2853 = (const uint8_t*) v2852;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2854 = __riscv_vle8_v_u8mf2(v2853, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2855 = __riscv_vand_vx_u8mf2(v2854, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2856 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2855);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2857 = __riscv_vsrl_vx_u8mf2(v2854, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v2858 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2857);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2859 = v32 + 396;
        const int8_t* v2860 = (const int8_t*) v2859;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2861 = *(const int8_t *)(v2860);
        const uint8_t* v2862 = v32 + 524;
        const int8_t* v2863 = (const int8_t*) v2862;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2864 = *(const int8_t *)(v2863);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2865 = __riscv_vwmacc_vx_i16m1(v2826, v2861, v2856, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2866 = __riscv_vwmacc_vx_i16m1(v2827, v2864, v2858, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2867 = v32 + 397;
        const int8_t* v2868 = (const int8_t*) v2867;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2869 = *(const int8_t *)(v2868);
        const uint8_t* v2870 = v32 + 525;
        const int8_t* v2871 = (const int8_t*) v2870;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2872 = *(const int8_t *)(v2871);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2873 = __riscv_vwmacc_vx_i16m1(v2834, v2869, v2856, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2874 = __riscv_vwmacc_vx_i16m1(v2835, v2872, v2858, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2875 = v32 + 398;
        const int8_t* v2876 = (const int8_t*) v2875;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2877 = *(const int8_t *)(v2876);
        const uint8_t* v2878 = v32 + 526;
        const int8_t* v2879 = (const int8_t*) v2878;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2880 = *(const int8_t *)(v2879);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2881 = __riscv_vwmacc_vx_i16m1(v2842, v2877, v2856, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2882 = __riscv_vwmacc_vx_i16m1(v2843, v2880, v2858, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v2883 = v32 + 399;
        const int8_t* v2884 = (const int8_t*) v2883;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2885 = *(const int8_t *)(v2884);
        const uint8_t* v2886 = v32 + 527;
        const int8_t* v2887 = (const int8_t*) v2886;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v2888 = *(const int8_t *)(v2887);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2889 = __riscv_vwmacc_vx_i16m1(v2850, v2885, v2856, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v2890 = __riscv_vwmacc_vx_i16m1(v2851, v2888, v2858, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v2891 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v2892 = __riscv_vle16_v_i16m1(v2891, 16);
        int16_t* v2893 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v2894 = __riscv_vle16_v_i16m1(v2893, 16);
        vint32m2_t v2895 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2896 = __riscv_vwmacc_vv_i32m2(v2895, v2892, v2865, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2897 = __riscv_vwmacc_vv_i32m2(v2896, v2894, v2866, 16);
        v47 = v2897;
        vint32m2_t v2898 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2899 = __riscv_vwmacc_vv_i32m2(v2898, v2892, v2873, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2900 = __riscv_vwmacc_vv_i32m2(v2899, v2894, v2874, 16);
        v49 = v2900;
        vint32m2_t v2901 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2902 = __riscv_vwmacc_vv_i32m2(v2901, v2892, v2881, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2903 = __riscv_vwmacc_vv_i32m2(v2902, v2894, v2882, 16);
        v51 = v2903;
        vint32m2_t v2904 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2905 = __riscv_vwmacc_vv_i32m2(v2904, v2892, v2889, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v2906 = __riscv_vwmacc_vv_i32m2(v2905, v2894, v2890, 16);
        v53 = v2906;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
        const uint8_t* v2907 = v30 + 128;
        const uint8_t* v2908 = (const uint8_t*) v2907;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2909 = __riscv_vle8_v_u8mf2(v2908, 16);
        const uint8_t* v2910 = v30 + 192;
        const uint8_t* v2911 = (const uint8_t*) v2910;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2912 = __riscv_vle8_v_u8mf2(v2911, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2913 = __riscv_vand_vx_u8mf2(v2909, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2914 = __riscv_vsrl_vx_u8mf2(v2909, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2915 = __riscv_vand_vx_u8mf2(v2912, 0x30, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2916 = __riscv_vand_vx_u8mf2(v2912, 0xC0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2917 = __riscv_vsrl_vx_u8mf2(v2916, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v2918 = __riscv_vor_vv_u8mf2(v2915, v2913, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v2919 = __riscv_vor_vv_u8mf2(v2917, v2914, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v2920 = __riscv_vzext_vf2_u16m1(v2918, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v2921 = __riscv_vreinterpret_v_u16m1_i16m1(v2920);
        int16_t* v2922 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v2922, v2921, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v2923 = __riscv_vzext_vf2_u16m1(v2919, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v2924 = __riscv_vreinterpret_v_u16m1_i16m1(v2923);
        int16_t* v2925 = &v45[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v2925, v2924, 16);
        const uint8_t* v2926 = v30 + 144;
        const uint8_t* v2927 = (const uint8_t*) v2926;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2928 = __riscv_vle8_v_u8mf2(v2927, 16);
        const uint8_t* v2929 = v30 + 208;
        const uint8_t* v2930 = (const uint8_t*) v2929;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2931 = __riscv_vle8_v_u8mf2(v2930, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2932 = __riscv_vand_vx_u8mf2(v2928, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2933 = __riscv_vsrl_vx_u8mf2(v2928, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2934 = __riscv_vand_vx_u8mf2(v2931, 0x30, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2935 = __riscv_vand_vx_u8mf2(v2931, 0xC0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2936 = __riscv_vsrl_vx_u8mf2(v2935, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v2937 = __riscv_vor_vv_u8mf2(v2934, v2932, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v2938 = __riscv_vor_vv_u8mf2(v2936, v2933, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v2939 = __riscv_vzext_vf2_u16m1(v2937, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v2940 = __riscv_vreinterpret_v_u16m1_i16m1(v2939);
        int16_t* v2941 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v2941, v2940, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v2942 = __riscv_vzext_vf2_u16m1(v2938, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v2943 = __riscv_vreinterpret_v_u16m1_i16m1(v2942);
        int16_t* v2944 = &v45[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v2944, v2943, 16);
        const uint8_t* v2945 = v30 + 160;
        const uint8_t* v2946 = (const uint8_t*) v2945;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2947 = __riscv_vle8_v_u8mf2(v2946, 16);
        const uint8_t* v2948 = v30 + 224;
        const uint8_t* v2949 = (const uint8_t*) v2948;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2950 = __riscv_vle8_v_u8mf2(v2949, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2951 = __riscv_vand_vx_u8mf2(v2947, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2952 = __riscv_vsrl_vx_u8mf2(v2947, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2953 = __riscv_vand_vx_u8mf2(v2950, 0x30, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2954 = __riscv_vand_vx_u8mf2(v2950, 0xC0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2955 = __riscv_vsrl_vx_u8mf2(v2954, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v2956 = __riscv_vor_vv_u8mf2(v2953, v2951, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v2957 = __riscv_vor_vv_u8mf2(v2955, v2952, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v2958 = __riscv_vzext_vf2_u16m1(v2956, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v2959 = __riscv_vreinterpret_v_u16m1_i16m1(v2958);
        int16_t* v2960 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v2960, v2959, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v2961 = __riscv_vzext_vf2_u16m1(v2957, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v2962 = __riscv_vreinterpret_v_u16m1_i16m1(v2961);
        int16_t* v2963 = &v45[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v2963, v2962, 16);
        const uint8_t* v2964 = v30 + 176;
        const uint8_t* v2965 = (const uint8_t*) v2964;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2966 = __riscv_vle8_v_u8mf2(v2965, 16);
        const uint8_t* v2967 = v30 + 240;
        const uint8_t* v2968 = (const uint8_t*) v2967;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v2969 = __riscv_vle8_v_u8mf2(v2968, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2970 = __riscv_vand_vx_u8mf2(v2966, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2971 = __riscv_vsrl_vx_u8mf2(v2966, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2972 = __riscv_vand_vx_u8mf2(v2969, 0x30, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v2973 = __riscv_vand_vx_u8mf2(v2969, 0xC0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v2974 = __riscv_vsrl_vx_u8mf2(v2973, 2, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v2975 = __riscv_vor_vv_u8mf2(v2972, v2970, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v2976 = __riscv_vor_vv_u8mf2(v2974, v2971, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v2977 = __riscv_vzext_vf2_u16m1(v2975, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v2978 = __riscv_vreinterpret_v_u16m1_i16m1(v2977);
        int16_t* v2979 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v2979, v2978, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v2980 = __riscv_vzext_vf2_u16m1(v2976, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v2981 = __riscv_vreinterpret_v_u16m1_i16m1(v2980);
        int16_t* v2982 = &v45[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v2982, v2981, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold
        int16_t* v2983 = &v45[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v2984 = __riscv_vle16_v_i16m1(v2983, 16);
        const uint8_t* v2985 = v32 + 1104;
        const int16_t* v2986 = (const int16_t*) v2985;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v2987 = *(const int16_t *)(v2986);
        const uint8_t* v2988 = v32 + 1112;
        const int16_t* v2989 = (const int16_t*) v2988;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v2990 = *(const int16_t *)(v2989);
        int32_t v2991 = v2987 + v2990;
        int32_t* v2992 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v2993 = __riscv_vle32_v_i32m2(v2992, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v2994 = __riscv_vwmacc_vx_i32m2(v2993, v2991, v2984, 16);
        int32_t* v2995 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v2995, v2994, 16);
        const uint8_t* v2996 = v32 + 1106;
        const int16_t* v2997 = (const int16_t*) v2996;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v2998 = *(const int16_t *)(v2997);
        const uint8_t* v2999 = v32 + 1114;
        const int16_t* v3000 = (const int16_t*) v2999;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3001 = *(const int16_t *)(v3000);
        int32_t v3002 = v2998 + v3001;
        int32_t* v3003 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3004 = __riscv_vle32_v_i32m2(v3003, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3005 = __riscv_vwmacc_vx_i32m2(v3004, v3002, v2984, 16);
        int32_t* v3006 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3006, v3005, 16);
        const uint8_t* v3007 = v32 + 1108;
        const int16_t* v3008 = (const int16_t*) v3007;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3009 = *(const int16_t *)(v3008);
        const uint8_t* v3010 = v32 + 1116;
        const int16_t* v3011 = (const int16_t*) v3010;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3012 = *(const int16_t *)(v3011);
        int32_t v3013 = v3009 + v3012;
        int32_t* v3014 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3015 = __riscv_vle32_v_i32m2(v3014, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3016 = __riscv_vwmacc_vx_i32m2(v3015, v3013, v2984, 16);
        int32_t* v3017 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3017, v3016, 16);
        const uint8_t* v3018 = v32 + 1110;
        const int16_t* v3019 = (const int16_t*) v3018;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3020 = *(const int16_t *)(v3019);
        const uint8_t* v3021 = v32 + 1118;
        const int16_t* v3022 = (const int16_t*) v3021;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3023 = *(const int16_t *)(v3022);
        int32_t v3024 = v3020 + v3023;
        int32_t* v3025 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3026 = __riscv_vle32_v_i32m2(v3025, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3027 = __riscv_vwmacc_vx_i32m2(v3026, v3024, v2984, 16);
        int32_t* v3028 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3028, v3027, 16);
        int16_t* v3029 = &v45[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v3030 = __riscv_vle16_v_i16m1(v3029, 16);
        const uint8_t* v3031 = v32 + 1120;
        const int16_t* v3032 = (const int16_t*) v3031;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3033 = *(const int16_t *)(v3032);
        const uint8_t* v3034 = v32 + 1128;
        const int16_t* v3035 = (const int16_t*) v3034;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3036 = *(const int16_t *)(v3035);
        int32_t v3037 = v3033 + v3036;
        int32_t* v3038 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3039 = __riscv_vle32_v_i32m2(v3038, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3040 = __riscv_vwmacc_vx_i32m2(v3039, v3037, v3030, 16);
        int32_t* v3041 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3041, v3040, 16);
        const uint8_t* v3042 = v32 + 1122;
        const int16_t* v3043 = (const int16_t*) v3042;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3044 = *(const int16_t *)(v3043);
        const uint8_t* v3045 = v32 + 1130;
        const int16_t* v3046 = (const int16_t*) v3045;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3047 = *(const int16_t *)(v3046);
        int32_t v3048 = v3044 + v3047;
        int32_t* v3049 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3050 = __riscv_vle32_v_i32m2(v3049, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3051 = __riscv_vwmacc_vx_i32m2(v3050, v3048, v3030, 16);
        int32_t* v3052 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3052, v3051, 16);
        const uint8_t* v3053 = v32 + 1124;
        const int16_t* v3054 = (const int16_t*) v3053;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3055 = *(const int16_t *)(v3054);
        const uint8_t* v3056 = v32 + 1132;
        const int16_t* v3057 = (const int16_t*) v3056;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3058 = *(const int16_t *)(v3057);
        int32_t v3059 = v3055 + v3058;
        int32_t* v3060 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3061 = __riscv_vle32_v_i32m2(v3060, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3062 = __riscv_vwmacc_vx_i32m2(v3061, v3059, v3030, 16);
        int32_t* v3063 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3063, v3062, 16);
        const uint8_t* v3064 = v32 + 1126;
        const int16_t* v3065 = (const int16_t*) v3064;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3066 = *(const int16_t *)(v3065);
        const uint8_t* v3067 = v32 + 1134;
        const int16_t* v3068 = (const int16_t*) v3067;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3069 = *(const int16_t *)(v3068);
        int32_t v3070 = v3066 + v3069;
        int32_t* v3071 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3072 = __riscv_vle32_v_i32m2(v3071, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3073 = __riscv_vwmacc_vx_i32m2(v3072, v3070, v3030, 16);
        int32_t* v3074 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3074, v3073, 16);
        int16_t* v3075 = &v45[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v3076 = __riscv_vle16_v_i16m1(v3075, 16);
        const uint8_t* v3077 = v32 + 1136;
        const int16_t* v3078 = (const int16_t*) v3077;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3079 = *(const int16_t *)(v3078);
        const uint8_t* v3080 = v32 + 1144;
        const int16_t* v3081 = (const int16_t*) v3080;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3082 = *(const int16_t *)(v3081);
        int32_t v3083 = v3079 + v3082;
        int32_t* v3084 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3085 = __riscv_vle32_v_i32m2(v3084, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3086 = __riscv_vwmacc_vx_i32m2(v3085, v3083, v3076, 16);
        int32_t* v3087 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3087, v3086, 16);
        const uint8_t* v3088 = v32 + 1138;
        const int16_t* v3089 = (const int16_t*) v3088;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3090 = *(const int16_t *)(v3089);
        const uint8_t* v3091 = v32 + 1146;
        const int16_t* v3092 = (const int16_t*) v3091;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3093 = *(const int16_t *)(v3092);
        int32_t v3094 = v3090 + v3093;
        int32_t* v3095 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3096 = __riscv_vle32_v_i32m2(v3095, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3097 = __riscv_vwmacc_vx_i32m2(v3096, v3094, v3076, 16);
        int32_t* v3098 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3098, v3097, 16);
        const uint8_t* v3099 = v32 + 1140;
        const int16_t* v3100 = (const int16_t*) v3099;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3101 = *(const int16_t *)(v3100);
        const uint8_t* v3102 = v32 + 1148;
        const int16_t* v3103 = (const int16_t*) v3102;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3104 = *(const int16_t *)(v3103);
        int32_t v3105 = v3101 + v3104;
        int32_t* v3106 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3107 = __riscv_vle32_v_i32m2(v3106, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3108 = __riscv_vwmacc_vx_i32m2(v3107, v3105, v3076, 16);
        int32_t* v3109 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3109, v3108, 16);
        const uint8_t* v3110 = v32 + 1142;
        const int16_t* v3111 = (const int16_t*) v3110;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3112 = *(const int16_t *)(v3111);
        const uint8_t* v3113 = v32 + 1150;
        const int16_t* v3114 = (const int16_t*) v3113;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3115 = *(const int16_t *)(v3114);
        int32_t v3116 = v3112 + v3115;
        int32_t* v3117 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3118 = __riscv_vle32_v_i32m2(v3117, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3119 = __riscv_vwmacc_vx_i32m2(v3118, v3116, v3076, 16);
        int32_t* v3120 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3120, v3119, 16);
        int16_t* v3121 = &v45[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v3122 = __riscv_vle16_v_i16m1(v3121, 16);
        const uint8_t* v3123 = v32 + 1152;
        const int16_t* v3124 = (const int16_t*) v3123;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3125 = *(const int16_t *)(v3124);
        const uint8_t* v3126 = v32 + 1160;
        const int16_t* v3127 = (const int16_t*) v3126;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3128 = *(const int16_t *)(v3127);
        int32_t v3129 = v3125 + v3128;
        int32_t* v3130 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3131 = __riscv_vle32_v_i32m2(v3130, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3132 = __riscv_vwmacc_vx_i32m2(v3131, v3129, v3122, 16);
        int32_t* v3133 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3133, v3132, 16);
        const uint8_t* v3134 = v32 + 1154;
        const int16_t* v3135 = (const int16_t*) v3134;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3136 = *(const int16_t *)(v3135);
        const uint8_t* v3137 = v32 + 1162;
        const int16_t* v3138 = (const int16_t*) v3137;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3139 = *(const int16_t *)(v3138);
        int32_t v3140 = v3136 + v3139;
        int32_t* v3141 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3142 = __riscv_vle32_v_i32m2(v3141, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3143 = __riscv_vwmacc_vx_i32m2(v3142, v3140, v3122, 16);
        int32_t* v3144 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3144, v3143, 16);
        const uint8_t* v3145 = v32 + 1156;
        const int16_t* v3146 = (const int16_t*) v3145;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3147 = *(const int16_t *)(v3146);
        const uint8_t* v3148 = v32 + 1164;
        const int16_t* v3149 = (const int16_t*) v3148;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3150 = *(const int16_t *)(v3149);
        int32_t v3151 = v3147 + v3150;
        int32_t* v3152 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3153 = __riscv_vle32_v_i32m2(v3152, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3154 = __riscv_vwmacc_vx_i32m2(v3153, v3151, v3122, 16);
        int32_t* v3155 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3155, v3154, 16);
        const uint8_t* v3156 = v32 + 1158;
        const int16_t* v3157 = (const int16_t*) v3156;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3158 = *(const int16_t *)(v3157);
        const uint8_t* v3159 = v32 + 1166;
        const int16_t* v3160 = (const int16_t*) v3159;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v3161 = *(const int16_t *)(v3160);
        int32_t v3162 = v3158 + v3161;
        int32_t* v3163 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v3164 = __riscv_vle32_v_i32m2(v3163, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v3165 = __riscv_vwmacc_vx_i32m2(v3164, v3162, v3122, 16);
        int32_t* v3166 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v3166, v3165, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3167 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3168 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3169 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3170 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3171 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3172 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3173 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3174 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3175 = v30 + 1280;
        const uint8_t* v3176 = (const uint8_t*) v3175;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3177 = __riscv_vle8_v_u8mf2(v3176, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3178 = __riscv_vand_vx_u8mf2(v3177, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3179 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3178);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3180 = __riscv_vsrl_vx_u8mf2(v3177, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3181 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3180);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3182 = v32 + 528;
        const int8_t* v3183 = (const int8_t*) v3182;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3184 = *(const int8_t *)(v3183);
        const uint8_t* v3185 = v32 + 656;
        const int8_t* v3186 = (const int8_t*) v3185;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3187 = *(const int8_t *)(v3186);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3188 = __riscv_vwmacc_vx_i16m1(v3167, v3184, v3179, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3189 = __riscv_vwmacc_vx_i16m1(v3168, v3187, v3181, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3190 = v32 + 529;
        const int8_t* v3191 = (const int8_t*) v3190;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3192 = *(const int8_t *)(v3191);
        const uint8_t* v3193 = v32 + 657;
        const int8_t* v3194 = (const int8_t*) v3193;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3195 = *(const int8_t *)(v3194);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3196 = __riscv_vwmacc_vx_i16m1(v3169, v3192, v3179, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3197 = __riscv_vwmacc_vx_i16m1(v3170, v3195, v3181, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3198 = v32 + 530;
        const int8_t* v3199 = (const int8_t*) v3198;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3200 = *(const int8_t *)(v3199);
        const uint8_t* v3201 = v32 + 658;
        const int8_t* v3202 = (const int8_t*) v3201;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3203 = *(const int8_t *)(v3202);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3204 = __riscv_vwmacc_vx_i16m1(v3171, v3200, v3179, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3205 = __riscv_vwmacc_vx_i16m1(v3172, v3203, v3181, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3206 = v32 + 531;
        const int8_t* v3207 = (const int8_t*) v3206;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3208 = *(const int8_t *)(v3207);
        const uint8_t* v3209 = v32 + 659;
        const int8_t* v3210 = (const int8_t*) v3209;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3211 = *(const int8_t *)(v3210);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3212 = __riscv_vwmacc_vx_i16m1(v3173, v3208, v3179, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3213 = __riscv_vwmacc_vx_i16m1(v3174, v3211, v3181, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3214 = v30 + 1296;
        const uint8_t* v3215 = (const uint8_t*) v3214;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3216 = __riscv_vle8_v_u8mf2(v3215, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3217 = __riscv_vand_vx_u8mf2(v3216, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3218 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3217);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3219 = __riscv_vsrl_vx_u8mf2(v3216, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3220 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3219);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3221 = v32 + 532;
        const int8_t* v3222 = (const int8_t*) v3221;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3223 = *(const int8_t *)(v3222);
        const uint8_t* v3224 = v32 + 660;
        const int8_t* v3225 = (const int8_t*) v3224;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3226 = *(const int8_t *)(v3225);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3227 = __riscv_vwmacc_vx_i16m1(v3188, v3223, v3218, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3228 = __riscv_vwmacc_vx_i16m1(v3189, v3226, v3220, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3229 = v32 + 533;
        const int8_t* v3230 = (const int8_t*) v3229;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3231 = *(const int8_t *)(v3230);
        const uint8_t* v3232 = v32 + 661;
        const int8_t* v3233 = (const int8_t*) v3232;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3234 = *(const int8_t *)(v3233);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3235 = __riscv_vwmacc_vx_i16m1(v3196, v3231, v3218, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3236 = __riscv_vwmacc_vx_i16m1(v3197, v3234, v3220, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3237 = v32 + 534;
        const int8_t* v3238 = (const int8_t*) v3237;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3239 = *(const int8_t *)(v3238);
        const uint8_t* v3240 = v32 + 662;
        const int8_t* v3241 = (const int8_t*) v3240;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3242 = *(const int8_t *)(v3241);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3243 = __riscv_vwmacc_vx_i16m1(v3204, v3239, v3218, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3244 = __riscv_vwmacc_vx_i16m1(v3205, v3242, v3220, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3245 = v32 + 535;
        const int8_t* v3246 = (const int8_t*) v3245;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3247 = *(const int8_t *)(v3246);
        const uint8_t* v3248 = v32 + 663;
        const int8_t* v3249 = (const int8_t*) v3248;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3250 = *(const int8_t *)(v3249);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3251 = __riscv_vwmacc_vx_i16m1(v3212, v3247, v3218, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3252 = __riscv_vwmacc_vx_i16m1(v3213, v3250, v3220, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3253 = v30 + 1312;
        const uint8_t* v3254 = (const uint8_t*) v3253;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3255 = __riscv_vle8_v_u8mf2(v3254, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3256 = __riscv_vand_vx_u8mf2(v3255, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3257 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3256);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3258 = __riscv_vsrl_vx_u8mf2(v3255, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3259 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3258);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3260 = v32 + 536;
        const int8_t* v3261 = (const int8_t*) v3260;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3262 = *(const int8_t *)(v3261);
        const uint8_t* v3263 = v32 + 664;
        const int8_t* v3264 = (const int8_t*) v3263;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3265 = *(const int8_t *)(v3264);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3266 = __riscv_vwmacc_vx_i16m1(v3227, v3262, v3257, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3267 = __riscv_vwmacc_vx_i16m1(v3228, v3265, v3259, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3268 = v32 + 537;
        const int8_t* v3269 = (const int8_t*) v3268;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3270 = *(const int8_t *)(v3269);
        const uint8_t* v3271 = v32 + 665;
        const int8_t* v3272 = (const int8_t*) v3271;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3273 = *(const int8_t *)(v3272);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3274 = __riscv_vwmacc_vx_i16m1(v3235, v3270, v3257, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3275 = __riscv_vwmacc_vx_i16m1(v3236, v3273, v3259, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3276 = v32 + 538;
        const int8_t* v3277 = (const int8_t*) v3276;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3278 = *(const int8_t *)(v3277);
        const uint8_t* v3279 = v32 + 666;
        const int8_t* v3280 = (const int8_t*) v3279;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3281 = *(const int8_t *)(v3280);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3282 = __riscv_vwmacc_vx_i16m1(v3243, v3278, v3257, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3283 = __riscv_vwmacc_vx_i16m1(v3244, v3281, v3259, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3284 = v32 + 539;
        const int8_t* v3285 = (const int8_t*) v3284;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3286 = *(const int8_t *)(v3285);
        const uint8_t* v3287 = v32 + 667;
        const int8_t* v3288 = (const int8_t*) v3287;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3289 = *(const int8_t *)(v3288);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3290 = __riscv_vwmacc_vx_i16m1(v3251, v3286, v3257, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3291 = __riscv_vwmacc_vx_i16m1(v3252, v3289, v3259, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3292 = v30 + 1328;
        const uint8_t* v3293 = (const uint8_t*) v3292;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3294 = __riscv_vle8_v_u8mf2(v3293, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3295 = __riscv_vand_vx_u8mf2(v3294, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3296 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3295);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3297 = __riscv_vsrl_vx_u8mf2(v3294, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3298 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3297);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3299 = v32 + 540;
        const int8_t* v3300 = (const int8_t*) v3299;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3301 = *(const int8_t *)(v3300);
        const uint8_t* v3302 = v32 + 668;
        const int8_t* v3303 = (const int8_t*) v3302;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3304 = *(const int8_t *)(v3303);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3305 = __riscv_vwmacc_vx_i16m1(v3266, v3301, v3296, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3306 = __riscv_vwmacc_vx_i16m1(v3267, v3304, v3298, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3307 = v32 + 541;
        const int8_t* v3308 = (const int8_t*) v3307;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3309 = *(const int8_t *)(v3308);
        const uint8_t* v3310 = v32 + 669;
        const int8_t* v3311 = (const int8_t*) v3310;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3312 = *(const int8_t *)(v3311);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3313 = __riscv_vwmacc_vx_i16m1(v3274, v3309, v3296, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3314 = __riscv_vwmacc_vx_i16m1(v3275, v3312, v3298, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3315 = v32 + 542;
        const int8_t* v3316 = (const int8_t*) v3315;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3317 = *(const int8_t *)(v3316);
        const uint8_t* v3318 = v32 + 670;
        const int8_t* v3319 = (const int8_t*) v3318;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3320 = *(const int8_t *)(v3319);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3321 = __riscv_vwmacc_vx_i16m1(v3282, v3317, v3296, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3322 = __riscv_vwmacc_vx_i16m1(v3283, v3320, v3298, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3323 = v32 + 543;
        const int8_t* v3324 = (const int8_t*) v3323;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3325 = *(const int8_t *)(v3324);
        const uint8_t* v3326 = v32 + 671;
        const int8_t* v3327 = (const int8_t*) v3326;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3328 = *(const int8_t *)(v3327);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3329 = __riscv_vwmacc_vx_i16m1(v3290, v3325, v3296, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3330 = __riscv_vwmacc_vx_i16m1(v3291, v3328, v3298, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3331 = v30 + 1344;
        const uint8_t* v3332 = (const uint8_t*) v3331;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3333 = __riscv_vle8_v_u8mf2(v3332, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3334 = __riscv_vand_vx_u8mf2(v3333, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3335 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3334);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3336 = __riscv_vsrl_vx_u8mf2(v3333, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3337 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3336);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3338 = v32 + 544;
        const int8_t* v3339 = (const int8_t*) v3338;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3340 = *(const int8_t *)(v3339);
        const uint8_t* v3341 = v32 + 672;
        const int8_t* v3342 = (const int8_t*) v3341;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3343 = *(const int8_t *)(v3342);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3344 = __riscv_vwmacc_vx_i16m1(v3305, v3340, v3335, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3345 = __riscv_vwmacc_vx_i16m1(v3306, v3343, v3337, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3346 = v32 + 545;
        const int8_t* v3347 = (const int8_t*) v3346;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3348 = *(const int8_t *)(v3347);
        const uint8_t* v3349 = v32 + 673;
        const int8_t* v3350 = (const int8_t*) v3349;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3351 = *(const int8_t *)(v3350);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3352 = __riscv_vwmacc_vx_i16m1(v3313, v3348, v3335, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3353 = __riscv_vwmacc_vx_i16m1(v3314, v3351, v3337, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3354 = v32 + 546;
        const int8_t* v3355 = (const int8_t*) v3354;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3356 = *(const int8_t *)(v3355);
        const uint8_t* v3357 = v32 + 674;
        const int8_t* v3358 = (const int8_t*) v3357;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3359 = *(const int8_t *)(v3358);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3360 = __riscv_vwmacc_vx_i16m1(v3321, v3356, v3335, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3361 = __riscv_vwmacc_vx_i16m1(v3322, v3359, v3337, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3362 = v32 + 547;
        const int8_t* v3363 = (const int8_t*) v3362;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3364 = *(const int8_t *)(v3363);
        const uint8_t* v3365 = v32 + 675;
        const int8_t* v3366 = (const int8_t*) v3365;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3367 = *(const int8_t *)(v3366);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3368 = __riscv_vwmacc_vx_i16m1(v3329, v3364, v3335, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3369 = __riscv_vwmacc_vx_i16m1(v3330, v3367, v3337, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3370 = v30 + 1360;
        const uint8_t* v3371 = (const uint8_t*) v3370;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3372 = __riscv_vle8_v_u8mf2(v3371, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3373 = __riscv_vand_vx_u8mf2(v3372, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3374 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3373);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3375 = __riscv_vsrl_vx_u8mf2(v3372, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3376 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3375);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3377 = v32 + 548;
        const int8_t* v3378 = (const int8_t*) v3377;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3379 = *(const int8_t *)(v3378);
        const uint8_t* v3380 = v32 + 676;
        const int8_t* v3381 = (const int8_t*) v3380;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3382 = *(const int8_t *)(v3381);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3383 = __riscv_vwmacc_vx_i16m1(v3344, v3379, v3374, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3384 = __riscv_vwmacc_vx_i16m1(v3345, v3382, v3376, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3385 = v32 + 549;
        const int8_t* v3386 = (const int8_t*) v3385;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3387 = *(const int8_t *)(v3386);
        const uint8_t* v3388 = v32 + 677;
        const int8_t* v3389 = (const int8_t*) v3388;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3390 = *(const int8_t *)(v3389);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3391 = __riscv_vwmacc_vx_i16m1(v3352, v3387, v3374, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3392 = __riscv_vwmacc_vx_i16m1(v3353, v3390, v3376, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3393 = v32 + 550;
        const int8_t* v3394 = (const int8_t*) v3393;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3395 = *(const int8_t *)(v3394);
        const uint8_t* v3396 = v32 + 678;
        const int8_t* v3397 = (const int8_t*) v3396;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3398 = *(const int8_t *)(v3397);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3399 = __riscv_vwmacc_vx_i16m1(v3360, v3395, v3374, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3400 = __riscv_vwmacc_vx_i16m1(v3361, v3398, v3376, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3401 = v32 + 551;
        const int8_t* v3402 = (const int8_t*) v3401;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3403 = *(const int8_t *)(v3402);
        const uint8_t* v3404 = v32 + 679;
        const int8_t* v3405 = (const int8_t*) v3404;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3406 = *(const int8_t *)(v3405);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3407 = __riscv_vwmacc_vx_i16m1(v3368, v3403, v3374, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3408 = __riscv_vwmacc_vx_i16m1(v3369, v3406, v3376, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3409 = v30 + 1376;
        const uint8_t* v3410 = (const uint8_t*) v3409;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3411 = __riscv_vle8_v_u8mf2(v3410, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3412 = __riscv_vand_vx_u8mf2(v3411, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3413 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3412);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3414 = __riscv_vsrl_vx_u8mf2(v3411, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3415 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3414);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3416 = v32 + 552;
        const int8_t* v3417 = (const int8_t*) v3416;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3418 = *(const int8_t *)(v3417);
        const uint8_t* v3419 = v32 + 680;
        const int8_t* v3420 = (const int8_t*) v3419;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3421 = *(const int8_t *)(v3420);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3422 = __riscv_vwmacc_vx_i16m1(v3383, v3418, v3413, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3423 = __riscv_vwmacc_vx_i16m1(v3384, v3421, v3415, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3424 = v32 + 553;
        const int8_t* v3425 = (const int8_t*) v3424;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3426 = *(const int8_t *)(v3425);
        const uint8_t* v3427 = v32 + 681;
        const int8_t* v3428 = (const int8_t*) v3427;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3429 = *(const int8_t *)(v3428);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3430 = __riscv_vwmacc_vx_i16m1(v3391, v3426, v3413, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3431 = __riscv_vwmacc_vx_i16m1(v3392, v3429, v3415, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3432 = v32 + 554;
        const int8_t* v3433 = (const int8_t*) v3432;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3434 = *(const int8_t *)(v3433);
        const uint8_t* v3435 = v32 + 682;
        const int8_t* v3436 = (const int8_t*) v3435;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3437 = *(const int8_t *)(v3436);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3438 = __riscv_vwmacc_vx_i16m1(v3399, v3434, v3413, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3439 = __riscv_vwmacc_vx_i16m1(v3400, v3437, v3415, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3440 = v32 + 555;
        const int8_t* v3441 = (const int8_t*) v3440;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3442 = *(const int8_t *)(v3441);
        const uint8_t* v3443 = v32 + 683;
        const int8_t* v3444 = (const int8_t*) v3443;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3445 = *(const int8_t *)(v3444);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3446 = __riscv_vwmacc_vx_i16m1(v3407, v3442, v3413, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3447 = __riscv_vwmacc_vx_i16m1(v3408, v3445, v3415, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3448 = v30 + 1392;
        const uint8_t* v3449 = (const uint8_t*) v3448;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3450 = __riscv_vle8_v_u8mf2(v3449, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3451 = __riscv_vand_vx_u8mf2(v3450, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3452 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3451);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3453 = __riscv_vsrl_vx_u8mf2(v3450, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3454 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3453);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3455 = v32 + 556;
        const int8_t* v3456 = (const int8_t*) v3455;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3457 = *(const int8_t *)(v3456);
        const uint8_t* v3458 = v32 + 684;
        const int8_t* v3459 = (const int8_t*) v3458;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3460 = *(const int8_t *)(v3459);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3461 = __riscv_vwmacc_vx_i16m1(v3422, v3457, v3452, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3462 = __riscv_vwmacc_vx_i16m1(v3423, v3460, v3454, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3463 = v32 + 557;
        const int8_t* v3464 = (const int8_t*) v3463;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3465 = *(const int8_t *)(v3464);
        const uint8_t* v3466 = v32 + 685;
        const int8_t* v3467 = (const int8_t*) v3466;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3468 = *(const int8_t *)(v3467);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3469 = __riscv_vwmacc_vx_i16m1(v3430, v3465, v3452, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3470 = __riscv_vwmacc_vx_i16m1(v3431, v3468, v3454, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3471 = v32 + 558;
        const int8_t* v3472 = (const int8_t*) v3471;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3473 = *(const int8_t *)(v3472);
        const uint8_t* v3474 = v32 + 686;
        const int8_t* v3475 = (const int8_t*) v3474;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3476 = *(const int8_t *)(v3475);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3477 = __riscv_vwmacc_vx_i16m1(v3438, v3473, v3452, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3478 = __riscv_vwmacc_vx_i16m1(v3439, v3476, v3454, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3479 = v32 + 559;
        const int8_t* v3480 = (const int8_t*) v3479;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3481 = *(const int8_t *)(v3480);
        const uint8_t* v3482 = v32 + 687;
        const int8_t* v3483 = (const int8_t*) v3482;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3484 = *(const int8_t *)(v3483);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3485 = __riscv_vwmacc_vx_i16m1(v3446, v3481, v3452, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3486 = __riscv_vwmacc_vx_i16m1(v3447, v3484, v3454, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3487 = v30 + 1408;
        const uint8_t* v3488 = (const uint8_t*) v3487;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3489 = __riscv_vle8_v_u8mf2(v3488, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3490 = __riscv_vand_vx_u8mf2(v3489, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3491 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3490);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3492 = __riscv_vsrl_vx_u8mf2(v3489, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3493 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3492);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3494 = v32 + 560;
        const int8_t* v3495 = (const int8_t*) v3494;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3496 = *(const int8_t *)(v3495);
        const uint8_t* v3497 = v32 + 688;
        const int8_t* v3498 = (const int8_t*) v3497;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3499 = *(const int8_t *)(v3498);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3500 = __riscv_vwmacc_vx_i16m1(v3461, v3496, v3491, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3501 = __riscv_vwmacc_vx_i16m1(v3462, v3499, v3493, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3502 = v32 + 561;
        const int8_t* v3503 = (const int8_t*) v3502;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3504 = *(const int8_t *)(v3503);
        const uint8_t* v3505 = v32 + 689;
        const int8_t* v3506 = (const int8_t*) v3505;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3507 = *(const int8_t *)(v3506);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3508 = __riscv_vwmacc_vx_i16m1(v3469, v3504, v3491, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3509 = __riscv_vwmacc_vx_i16m1(v3470, v3507, v3493, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3510 = v32 + 562;
        const int8_t* v3511 = (const int8_t*) v3510;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3512 = *(const int8_t *)(v3511);
        const uint8_t* v3513 = v32 + 690;
        const int8_t* v3514 = (const int8_t*) v3513;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3515 = *(const int8_t *)(v3514);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3516 = __riscv_vwmacc_vx_i16m1(v3477, v3512, v3491, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3517 = __riscv_vwmacc_vx_i16m1(v3478, v3515, v3493, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3518 = v32 + 563;
        const int8_t* v3519 = (const int8_t*) v3518;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3520 = *(const int8_t *)(v3519);
        const uint8_t* v3521 = v32 + 691;
        const int8_t* v3522 = (const int8_t*) v3521;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3523 = *(const int8_t *)(v3522);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3524 = __riscv_vwmacc_vx_i16m1(v3485, v3520, v3491, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3525 = __riscv_vwmacc_vx_i16m1(v3486, v3523, v3493, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3526 = v30 + 1424;
        const uint8_t* v3527 = (const uint8_t*) v3526;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3528 = __riscv_vle8_v_u8mf2(v3527, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3529 = __riscv_vand_vx_u8mf2(v3528, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3530 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3529);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3531 = __riscv_vsrl_vx_u8mf2(v3528, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3532 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3531);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3533 = v32 + 564;
        const int8_t* v3534 = (const int8_t*) v3533;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3535 = *(const int8_t *)(v3534);
        const uint8_t* v3536 = v32 + 692;
        const int8_t* v3537 = (const int8_t*) v3536;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3538 = *(const int8_t *)(v3537);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3539 = __riscv_vwmacc_vx_i16m1(v3500, v3535, v3530, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3540 = __riscv_vwmacc_vx_i16m1(v3501, v3538, v3532, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3541 = v32 + 565;
        const int8_t* v3542 = (const int8_t*) v3541;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3543 = *(const int8_t *)(v3542);
        const uint8_t* v3544 = v32 + 693;
        const int8_t* v3545 = (const int8_t*) v3544;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3546 = *(const int8_t *)(v3545);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3547 = __riscv_vwmacc_vx_i16m1(v3508, v3543, v3530, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3548 = __riscv_vwmacc_vx_i16m1(v3509, v3546, v3532, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3549 = v32 + 566;
        const int8_t* v3550 = (const int8_t*) v3549;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3551 = *(const int8_t *)(v3550);
        const uint8_t* v3552 = v32 + 694;
        const int8_t* v3553 = (const int8_t*) v3552;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3554 = *(const int8_t *)(v3553);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3555 = __riscv_vwmacc_vx_i16m1(v3516, v3551, v3530, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3556 = __riscv_vwmacc_vx_i16m1(v3517, v3554, v3532, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3557 = v32 + 567;
        const int8_t* v3558 = (const int8_t*) v3557;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3559 = *(const int8_t *)(v3558);
        const uint8_t* v3560 = v32 + 695;
        const int8_t* v3561 = (const int8_t*) v3560;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3562 = *(const int8_t *)(v3561);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3563 = __riscv_vwmacc_vx_i16m1(v3524, v3559, v3530, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3564 = __riscv_vwmacc_vx_i16m1(v3525, v3562, v3532, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3565 = v30 + 1440;
        const uint8_t* v3566 = (const uint8_t*) v3565;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3567 = __riscv_vle8_v_u8mf2(v3566, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3568 = __riscv_vand_vx_u8mf2(v3567, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3569 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3568);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3570 = __riscv_vsrl_vx_u8mf2(v3567, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3571 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3570);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3572 = v32 + 568;
        const int8_t* v3573 = (const int8_t*) v3572;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3574 = *(const int8_t *)(v3573);
        const uint8_t* v3575 = v32 + 696;
        const int8_t* v3576 = (const int8_t*) v3575;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3577 = *(const int8_t *)(v3576);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3578 = __riscv_vwmacc_vx_i16m1(v3539, v3574, v3569, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3579 = __riscv_vwmacc_vx_i16m1(v3540, v3577, v3571, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3580 = v32 + 569;
        const int8_t* v3581 = (const int8_t*) v3580;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3582 = *(const int8_t *)(v3581);
        const uint8_t* v3583 = v32 + 697;
        const int8_t* v3584 = (const int8_t*) v3583;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3585 = *(const int8_t *)(v3584);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3586 = __riscv_vwmacc_vx_i16m1(v3547, v3582, v3569, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3587 = __riscv_vwmacc_vx_i16m1(v3548, v3585, v3571, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3588 = v32 + 570;
        const int8_t* v3589 = (const int8_t*) v3588;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3590 = *(const int8_t *)(v3589);
        const uint8_t* v3591 = v32 + 698;
        const int8_t* v3592 = (const int8_t*) v3591;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3593 = *(const int8_t *)(v3592);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3594 = __riscv_vwmacc_vx_i16m1(v3555, v3590, v3569, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3595 = __riscv_vwmacc_vx_i16m1(v3556, v3593, v3571, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3596 = v32 + 571;
        const int8_t* v3597 = (const int8_t*) v3596;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3598 = *(const int8_t *)(v3597);
        const uint8_t* v3599 = v32 + 699;
        const int8_t* v3600 = (const int8_t*) v3599;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3601 = *(const int8_t *)(v3600);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3602 = __riscv_vwmacc_vx_i16m1(v3563, v3598, v3569, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3603 = __riscv_vwmacc_vx_i16m1(v3564, v3601, v3571, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3604 = v30 + 1456;
        const uint8_t* v3605 = (const uint8_t*) v3604;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3606 = __riscv_vle8_v_u8mf2(v3605, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3607 = __riscv_vand_vx_u8mf2(v3606, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3608 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3607);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3609 = __riscv_vsrl_vx_u8mf2(v3606, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3610 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3609);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3611 = v32 + 572;
        const int8_t* v3612 = (const int8_t*) v3611;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3613 = *(const int8_t *)(v3612);
        const uint8_t* v3614 = v32 + 700;
        const int8_t* v3615 = (const int8_t*) v3614;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3616 = *(const int8_t *)(v3615);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3617 = __riscv_vwmacc_vx_i16m1(v3578, v3613, v3608, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3618 = __riscv_vwmacc_vx_i16m1(v3579, v3616, v3610, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3619 = v32 + 573;
        const int8_t* v3620 = (const int8_t*) v3619;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3621 = *(const int8_t *)(v3620);
        const uint8_t* v3622 = v32 + 701;
        const int8_t* v3623 = (const int8_t*) v3622;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3624 = *(const int8_t *)(v3623);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3625 = __riscv_vwmacc_vx_i16m1(v3586, v3621, v3608, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3626 = __riscv_vwmacc_vx_i16m1(v3587, v3624, v3610, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3627 = v32 + 574;
        const int8_t* v3628 = (const int8_t*) v3627;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3629 = *(const int8_t *)(v3628);
        const uint8_t* v3630 = v32 + 702;
        const int8_t* v3631 = (const int8_t*) v3630;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3632 = *(const int8_t *)(v3631);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3633 = __riscv_vwmacc_vx_i16m1(v3594, v3629, v3608, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3634 = __riscv_vwmacc_vx_i16m1(v3595, v3632, v3610, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3635 = v32 + 575;
        const int8_t* v3636 = (const int8_t*) v3635;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3637 = *(const int8_t *)(v3636);
        const uint8_t* v3638 = v32 + 703;
        const int8_t* v3639 = (const int8_t*) v3638;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3640 = *(const int8_t *)(v3639);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3641 = __riscv_vwmacc_vx_i16m1(v3602, v3637, v3608, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3642 = __riscv_vwmacc_vx_i16m1(v3603, v3640, v3610, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3643 = v30 + 1472;
        const uint8_t* v3644 = (const uint8_t*) v3643;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3645 = __riscv_vle8_v_u8mf2(v3644, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3646 = __riscv_vand_vx_u8mf2(v3645, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3647 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3646);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3648 = __riscv_vsrl_vx_u8mf2(v3645, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3649 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3648);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3650 = v32 + 576;
        const int8_t* v3651 = (const int8_t*) v3650;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3652 = *(const int8_t *)(v3651);
        const uint8_t* v3653 = v32 + 704;
        const int8_t* v3654 = (const int8_t*) v3653;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3655 = *(const int8_t *)(v3654);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3656 = __riscv_vwmacc_vx_i16m1(v3617, v3652, v3647, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3657 = __riscv_vwmacc_vx_i16m1(v3618, v3655, v3649, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3658 = v32 + 577;
        const int8_t* v3659 = (const int8_t*) v3658;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3660 = *(const int8_t *)(v3659);
        const uint8_t* v3661 = v32 + 705;
        const int8_t* v3662 = (const int8_t*) v3661;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3663 = *(const int8_t *)(v3662);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3664 = __riscv_vwmacc_vx_i16m1(v3625, v3660, v3647, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3665 = __riscv_vwmacc_vx_i16m1(v3626, v3663, v3649, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3666 = v32 + 578;
        const int8_t* v3667 = (const int8_t*) v3666;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3668 = *(const int8_t *)(v3667);
        const uint8_t* v3669 = v32 + 706;
        const int8_t* v3670 = (const int8_t*) v3669;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3671 = *(const int8_t *)(v3670);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3672 = __riscv_vwmacc_vx_i16m1(v3633, v3668, v3647, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3673 = __riscv_vwmacc_vx_i16m1(v3634, v3671, v3649, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3674 = v32 + 579;
        const int8_t* v3675 = (const int8_t*) v3674;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3676 = *(const int8_t *)(v3675);
        const uint8_t* v3677 = v32 + 707;
        const int8_t* v3678 = (const int8_t*) v3677;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3679 = *(const int8_t *)(v3678);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3680 = __riscv_vwmacc_vx_i16m1(v3641, v3676, v3647, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3681 = __riscv_vwmacc_vx_i16m1(v3642, v3679, v3649, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3682 = v30 + 1488;
        const uint8_t* v3683 = (const uint8_t*) v3682;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3684 = __riscv_vle8_v_u8mf2(v3683, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3685 = __riscv_vand_vx_u8mf2(v3684, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3686 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3685);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3687 = __riscv_vsrl_vx_u8mf2(v3684, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3688 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3687);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3689 = v32 + 580;
        const int8_t* v3690 = (const int8_t*) v3689;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3691 = *(const int8_t *)(v3690);
        const uint8_t* v3692 = v32 + 708;
        const int8_t* v3693 = (const int8_t*) v3692;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3694 = *(const int8_t *)(v3693);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3695 = __riscv_vwmacc_vx_i16m1(v3656, v3691, v3686, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3696 = __riscv_vwmacc_vx_i16m1(v3657, v3694, v3688, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3697 = v32 + 581;
        const int8_t* v3698 = (const int8_t*) v3697;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3699 = *(const int8_t *)(v3698);
        const uint8_t* v3700 = v32 + 709;
        const int8_t* v3701 = (const int8_t*) v3700;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3702 = *(const int8_t *)(v3701);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3703 = __riscv_vwmacc_vx_i16m1(v3664, v3699, v3686, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3704 = __riscv_vwmacc_vx_i16m1(v3665, v3702, v3688, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3705 = v32 + 582;
        const int8_t* v3706 = (const int8_t*) v3705;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3707 = *(const int8_t *)(v3706);
        const uint8_t* v3708 = v32 + 710;
        const int8_t* v3709 = (const int8_t*) v3708;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3710 = *(const int8_t *)(v3709);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3711 = __riscv_vwmacc_vx_i16m1(v3672, v3707, v3686, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3712 = __riscv_vwmacc_vx_i16m1(v3673, v3710, v3688, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3713 = v32 + 583;
        const int8_t* v3714 = (const int8_t*) v3713;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3715 = *(const int8_t *)(v3714);
        const uint8_t* v3716 = v32 + 711;
        const int8_t* v3717 = (const int8_t*) v3716;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3718 = *(const int8_t *)(v3717);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3719 = __riscv_vwmacc_vx_i16m1(v3680, v3715, v3686, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3720 = __riscv_vwmacc_vx_i16m1(v3681, v3718, v3688, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3721 = v30 + 1504;
        const uint8_t* v3722 = (const uint8_t*) v3721;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3723 = __riscv_vle8_v_u8mf2(v3722, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3724 = __riscv_vand_vx_u8mf2(v3723, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3725 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3724);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3726 = __riscv_vsrl_vx_u8mf2(v3723, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3727 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3726);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3728 = v32 + 584;
        const int8_t* v3729 = (const int8_t*) v3728;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3730 = *(const int8_t *)(v3729);
        const uint8_t* v3731 = v32 + 712;
        const int8_t* v3732 = (const int8_t*) v3731;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3733 = *(const int8_t *)(v3732);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3734 = __riscv_vwmacc_vx_i16m1(v3695, v3730, v3725, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3735 = __riscv_vwmacc_vx_i16m1(v3696, v3733, v3727, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3736 = v32 + 585;
        const int8_t* v3737 = (const int8_t*) v3736;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3738 = *(const int8_t *)(v3737);
        const uint8_t* v3739 = v32 + 713;
        const int8_t* v3740 = (const int8_t*) v3739;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3741 = *(const int8_t *)(v3740);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3742 = __riscv_vwmacc_vx_i16m1(v3703, v3738, v3725, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3743 = __riscv_vwmacc_vx_i16m1(v3704, v3741, v3727, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3744 = v32 + 586;
        const int8_t* v3745 = (const int8_t*) v3744;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3746 = *(const int8_t *)(v3745);
        const uint8_t* v3747 = v32 + 714;
        const int8_t* v3748 = (const int8_t*) v3747;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3749 = *(const int8_t *)(v3748);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3750 = __riscv_vwmacc_vx_i16m1(v3711, v3746, v3725, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3751 = __riscv_vwmacc_vx_i16m1(v3712, v3749, v3727, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3752 = v32 + 587;
        const int8_t* v3753 = (const int8_t*) v3752;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3754 = *(const int8_t *)(v3753);
        const uint8_t* v3755 = v32 + 715;
        const int8_t* v3756 = (const int8_t*) v3755;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3757 = *(const int8_t *)(v3756);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3758 = __riscv_vwmacc_vx_i16m1(v3719, v3754, v3725, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3759 = __riscv_vwmacc_vx_i16m1(v3720, v3757, v3727, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3760 = v30 + 1520;
        const uint8_t* v3761 = (const uint8_t*) v3760;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3762 = __riscv_vle8_v_u8mf2(v3761, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3763 = __riscv_vand_vx_u8mf2(v3762, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3764 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3763);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3765 = __riscv_vsrl_vx_u8mf2(v3762, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3766 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3765);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3767 = v32 + 588;
        const int8_t* v3768 = (const int8_t*) v3767;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3769 = *(const int8_t *)(v3768);
        const uint8_t* v3770 = v32 + 716;
        const int8_t* v3771 = (const int8_t*) v3770;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3772 = *(const int8_t *)(v3771);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3773 = __riscv_vwmacc_vx_i16m1(v3734, v3769, v3764, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3774 = __riscv_vwmacc_vx_i16m1(v3735, v3772, v3766, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3775 = v32 + 589;
        const int8_t* v3776 = (const int8_t*) v3775;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3777 = *(const int8_t *)(v3776);
        const uint8_t* v3778 = v32 + 717;
        const int8_t* v3779 = (const int8_t*) v3778;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3780 = *(const int8_t *)(v3779);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3781 = __riscv_vwmacc_vx_i16m1(v3742, v3777, v3764, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3782 = __riscv_vwmacc_vx_i16m1(v3743, v3780, v3766, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3783 = v32 + 590;
        const int8_t* v3784 = (const int8_t*) v3783;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3785 = *(const int8_t *)(v3784);
        const uint8_t* v3786 = v32 + 718;
        const int8_t* v3787 = (const int8_t*) v3786;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3788 = *(const int8_t *)(v3787);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3789 = __riscv_vwmacc_vx_i16m1(v3750, v3785, v3764, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3790 = __riscv_vwmacc_vx_i16m1(v3751, v3788, v3766, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3791 = v32 + 591;
        const int8_t* v3792 = (const int8_t*) v3791;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3793 = *(const int8_t *)(v3792);
        const uint8_t* v3794 = v32 + 719;
        const int8_t* v3795 = (const int8_t*) v3794;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3796 = *(const int8_t *)(v3795);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3797 = __riscv_vwmacc_vx_i16m1(v3758, v3793, v3764, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3798 = __riscv_vwmacc_vx_i16m1(v3759, v3796, v3766, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v3799 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v3800 = __riscv_vle16_v_i16m1(v3799, 16);
        int16_t* v3801 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v3802 = __riscv_vle16_v_i16m1(v3801, 16);
        vint32m2_t v3803 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v3804 = __riscv_vwmacc_vv_i32m2(v3803, v3800, v3773, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v3805 = __riscv_vwmacc_vv_i32m2(v3804, v3802, v3774, 16);
        v47 = v3805;
        vint32m2_t v3806 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v3807 = __riscv_vwmacc_vv_i32m2(v3806, v3800, v3781, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v3808 = __riscv_vwmacc_vv_i32m2(v3807, v3802, v3782, 16);
        v49 = v3808;
        vint32m2_t v3809 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v3810 = __riscv_vwmacc_vv_i32m2(v3809, v3800, v3789, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v3811 = __riscv_vwmacc_vv_i32m2(v3810, v3802, v3790, 16);
        v51 = v3811;
        vint32m2_t v3812 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v3813 = __riscv_vwmacc_vv_i32m2(v3812, v3800, v3797, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v3814 = __riscv_vwmacc_vv_i32m2(v3813, v3802, v3798, 16);
        v53 = v3814;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3815 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3816 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3817 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3818 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3819 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3820 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3821 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v3822 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3823 = v30 + 1536;
        const uint8_t* v3824 = (const uint8_t*) v3823;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3825 = __riscv_vle8_v_u8mf2(v3824, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3826 = __riscv_vand_vx_u8mf2(v3825, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3827 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3826);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3828 = __riscv_vsrl_vx_u8mf2(v3825, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3829 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3828);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3830 = v32 + 592;
        const int8_t* v3831 = (const int8_t*) v3830;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3832 = *(const int8_t *)(v3831);
        const uint8_t* v3833 = v32 + 720;
        const int8_t* v3834 = (const int8_t*) v3833;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3835 = *(const int8_t *)(v3834);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3836 = __riscv_vwmacc_vx_i16m1(v3815, v3832, v3827, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3837 = __riscv_vwmacc_vx_i16m1(v3816, v3835, v3829, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3838 = v32 + 593;
        const int8_t* v3839 = (const int8_t*) v3838;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3840 = *(const int8_t *)(v3839);
        const uint8_t* v3841 = v32 + 721;
        const int8_t* v3842 = (const int8_t*) v3841;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3843 = *(const int8_t *)(v3842);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3844 = __riscv_vwmacc_vx_i16m1(v3817, v3840, v3827, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3845 = __riscv_vwmacc_vx_i16m1(v3818, v3843, v3829, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3846 = v32 + 594;
        const int8_t* v3847 = (const int8_t*) v3846;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3848 = *(const int8_t *)(v3847);
        const uint8_t* v3849 = v32 + 722;
        const int8_t* v3850 = (const int8_t*) v3849;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3851 = *(const int8_t *)(v3850);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3852 = __riscv_vwmacc_vx_i16m1(v3819, v3848, v3827, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3853 = __riscv_vwmacc_vx_i16m1(v3820, v3851, v3829, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3854 = v32 + 595;
        const int8_t* v3855 = (const int8_t*) v3854;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3856 = *(const int8_t *)(v3855);
        const uint8_t* v3857 = v32 + 723;
        const int8_t* v3858 = (const int8_t*) v3857;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3859 = *(const int8_t *)(v3858);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3860 = __riscv_vwmacc_vx_i16m1(v3821, v3856, v3827, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3861 = __riscv_vwmacc_vx_i16m1(v3822, v3859, v3829, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3862 = v30 + 1552;
        const uint8_t* v3863 = (const uint8_t*) v3862;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3864 = __riscv_vle8_v_u8mf2(v3863, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3865 = __riscv_vand_vx_u8mf2(v3864, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3866 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3865);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3867 = __riscv_vsrl_vx_u8mf2(v3864, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3868 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3867);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3869 = v32 + 596;
        const int8_t* v3870 = (const int8_t*) v3869;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3871 = *(const int8_t *)(v3870);
        const uint8_t* v3872 = v32 + 724;
        const int8_t* v3873 = (const int8_t*) v3872;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3874 = *(const int8_t *)(v3873);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3875 = __riscv_vwmacc_vx_i16m1(v3836, v3871, v3866, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3876 = __riscv_vwmacc_vx_i16m1(v3837, v3874, v3868, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3877 = v32 + 597;
        const int8_t* v3878 = (const int8_t*) v3877;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3879 = *(const int8_t *)(v3878);
        const uint8_t* v3880 = v32 + 725;
        const int8_t* v3881 = (const int8_t*) v3880;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3882 = *(const int8_t *)(v3881);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3883 = __riscv_vwmacc_vx_i16m1(v3844, v3879, v3866, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3884 = __riscv_vwmacc_vx_i16m1(v3845, v3882, v3868, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3885 = v32 + 598;
        const int8_t* v3886 = (const int8_t*) v3885;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3887 = *(const int8_t *)(v3886);
        const uint8_t* v3888 = v32 + 726;
        const int8_t* v3889 = (const int8_t*) v3888;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3890 = *(const int8_t *)(v3889);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3891 = __riscv_vwmacc_vx_i16m1(v3852, v3887, v3866, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3892 = __riscv_vwmacc_vx_i16m1(v3853, v3890, v3868, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3893 = v32 + 599;
        const int8_t* v3894 = (const int8_t*) v3893;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3895 = *(const int8_t *)(v3894);
        const uint8_t* v3896 = v32 + 727;
        const int8_t* v3897 = (const int8_t*) v3896;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3898 = *(const int8_t *)(v3897);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3899 = __riscv_vwmacc_vx_i16m1(v3860, v3895, v3866, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3900 = __riscv_vwmacc_vx_i16m1(v3861, v3898, v3868, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3901 = v30 + 1568;
        const uint8_t* v3902 = (const uint8_t*) v3901;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3903 = __riscv_vle8_v_u8mf2(v3902, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3904 = __riscv_vand_vx_u8mf2(v3903, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3905 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3904);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3906 = __riscv_vsrl_vx_u8mf2(v3903, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3907 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3906);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3908 = v32 + 600;
        const int8_t* v3909 = (const int8_t*) v3908;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3910 = *(const int8_t *)(v3909);
        const uint8_t* v3911 = v32 + 728;
        const int8_t* v3912 = (const int8_t*) v3911;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3913 = *(const int8_t *)(v3912);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3914 = __riscv_vwmacc_vx_i16m1(v3875, v3910, v3905, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3915 = __riscv_vwmacc_vx_i16m1(v3876, v3913, v3907, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3916 = v32 + 601;
        const int8_t* v3917 = (const int8_t*) v3916;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3918 = *(const int8_t *)(v3917);
        const uint8_t* v3919 = v32 + 729;
        const int8_t* v3920 = (const int8_t*) v3919;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3921 = *(const int8_t *)(v3920);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3922 = __riscv_vwmacc_vx_i16m1(v3883, v3918, v3905, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3923 = __riscv_vwmacc_vx_i16m1(v3884, v3921, v3907, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3924 = v32 + 602;
        const int8_t* v3925 = (const int8_t*) v3924;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3926 = *(const int8_t *)(v3925);
        const uint8_t* v3927 = v32 + 730;
        const int8_t* v3928 = (const int8_t*) v3927;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3929 = *(const int8_t *)(v3928);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3930 = __riscv_vwmacc_vx_i16m1(v3891, v3926, v3905, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3931 = __riscv_vwmacc_vx_i16m1(v3892, v3929, v3907, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3932 = v32 + 603;
        const int8_t* v3933 = (const int8_t*) v3932;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3934 = *(const int8_t *)(v3933);
        const uint8_t* v3935 = v32 + 731;
        const int8_t* v3936 = (const int8_t*) v3935;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3937 = *(const int8_t *)(v3936);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3938 = __riscv_vwmacc_vx_i16m1(v3899, v3934, v3905, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3939 = __riscv_vwmacc_vx_i16m1(v3900, v3937, v3907, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3940 = v30 + 1584;
        const uint8_t* v3941 = (const uint8_t*) v3940;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3942 = __riscv_vle8_v_u8mf2(v3941, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3943 = __riscv_vand_vx_u8mf2(v3942, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3944 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3943);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3945 = __riscv_vsrl_vx_u8mf2(v3942, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3946 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3945);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3947 = v32 + 604;
        const int8_t* v3948 = (const int8_t*) v3947;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3949 = *(const int8_t *)(v3948);
        const uint8_t* v3950 = v32 + 732;
        const int8_t* v3951 = (const int8_t*) v3950;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3952 = *(const int8_t *)(v3951);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3953 = __riscv_vwmacc_vx_i16m1(v3914, v3949, v3944, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3954 = __riscv_vwmacc_vx_i16m1(v3915, v3952, v3946, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3955 = v32 + 605;
        const int8_t* v3956 = (const int8_t*) v3955;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3957 = *(const int8_t *)(v3956);
        const uint8_t* v3958 = v32 + 733;
        const int8_t* v3959 = (const int8_t*) v3958;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3960 = *(const int8_t *)(v3959);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3961 = __riscv_vwmacc_vx_i16m1(v3922, v3957, v3944, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3962 = __riscv_vwmacc_vx_i16m1(v3923, v3960, v3946, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3963 = v32 + 606;
        const int8_t* v3964 = (const int8_t*) v3963;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3965 = *(const int8_t *)(v3964);
        const uint8_t* v3966 = v32 + 734;
        const int8_t* v3967 = (const int8_t*) v3966;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3968 = *(const int8_t *)(v3967);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3969 = __riscv_vwmacc_vx_i16m1(v3930, v3965, v3944, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3970 = __riscv_vwmacc_vx_i16m1(v3931, v3968, v3946, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3971 = v32 + 607;
        const int8_t* v3972 = (const int8_t*) v3971;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3973 = *(const int8_t *)(v3972);
        const uint8_t* v3974 = v32 + 735;
        const int8_t* v3975 = (const int8_t*) v3974;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3976 = *(const int8_t *)(v3975);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3977 = __riscv_vwmacc_vx_i16m1(v3938, v3973, v3944, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3978 = __riscv_vwmacc_vx_i16m1(v3939, v3976, v3946, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v3979 = v30 + 1600;
        const uint8_t* v3980 = (const uint8_t*) v3979;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v3981 = __riscv_vle8_v_u8mf2(v3980, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v3982 = __riscv_vand_vx_u8mf2(v3981, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3983 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3982);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v3984 = __riscv_vsrl_vx_u8mf2(v3981, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v3985 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3984);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3986 = v32 + 608;
        const int8_t* v3987 = (const int8_t*) v3986;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3988 = *(const int8_t *)(v3987);
        const uint8_t* v3989 = v32 + 736;
        const int8_t* v3990 = (const int8_t*) v3989;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3991 = *(const int8_t *)(v3990);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3992 = __riscv_vwmacc_vx_i16m1(v3953, v3988, v3983, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v3993 = __riscv_vwmacc_vx_i16m1(v3954, v3991, v3985, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v3994 = v32 + 609;
        const int8_t* v3995 = (const int8_t*) v3994;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3996 = *(const int8_t *)(v3995);
        const uint8_t* v3997 = v32 + 737;
        const int8_t* v3998 = (const int8_t*) v3997;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v3999 = *(const int8_t *)(v3998);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4000 = __riscv_vwmacc_vx_i16m1(v3961, v3996, v3983, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4001 = __riscv_vwmacc_vx_i16m1(v3962, v3999, v3985, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4002 = v32 + 610;
        const int8_t* v4003 = (const int8_t*) v4002;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4004 = *(const int8_t *)(v4003);
        const uint8_t* v4005 = v32 + 738;
        const int8_t* v4006 = (const int8_t*) v4005;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4007 = *(const int8_t *)(v4006);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4008 = __riscv_vwmacc_vx_i16m1(v3969, v4004, v3983, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4009 = __riscv_vwmacc_vx_i16m1(v3970, v4007, v3985, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4010 = v32 + 611;
        const int8_t* v4011 = (const int8_t*) v4010;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4012 = *(const int8_t *)(v4011);
        const uint8_t* v4013 = v32 + 739;
        const int8_t* v4014 = (const int8_t*) v4013;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4015 = *(const int8_t *)(v4014);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4016 = __riscv_vwmacc_vx_i16m1(v3977, v4012, v3983, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4017 = __riscv_vwmacc_vx_i16m1(v3978, v4015, v3985, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4018 = v30 + 1616;
        const uint8_t* v4019 = (const uint8_t*) v4018;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4020 = __riscv_vle8_v_u8mf2(v4019, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4021 = __riscv_vand_vx_u8mf2(v4020, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4022 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4021);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4023 = __riscv_vsrl_vx_u8mf2(v4020, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4024 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4023);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4025 = v32 + 612;
        const int8_t* v4026 = (const int8_t*) v4025;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4027 = *(const int8_t *)(v4026);
        const uint8_t* v4028 = v32 + 740;
        const int8_t* v4029 = (const int8_t*) v4028;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4030 = *(const int8_t *)(v4029);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4031 = __riscv_vwmacc_vx_i16m1(v3992, v4027, v4022, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4032 = __riscv_vwmacc_vx_i16m1(v3993, v4030, v4024, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4033 = v32 + 613;
        const int8_t* v4034 = (const int8_t*) v4033;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4035 = *(const int8_t *)(v4034);
        const uint8_t* v4036 = v32 + 741;
        const int8_t* v4037 = (const int8_t*) v4036;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4038 = *(const int8_t *)(v4037);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4039 = __riscv_vwmacc_vx_i16m1(v4000, v4035, v4022, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4040 = __riscv_vwmacc_vx_i16m1(v4001, v4038, v4024, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4041 = v32 + 614;
        const int8_t* v4042 = (const int8_t*) v4041;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4043 = *(const int8_t *)(v4042);
        const uint8_t* v4044 = v32 + 742;
        const int8_t* v4045 = (const int8_t*) v4044;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4046 = *(const int8_t *)(v4045);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4047 = __riscv_vwmacc_vx_i16m1(v4008, v4043, v4022, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4048 = __riscv_vwmacc_vx_i16m1(v4009, v4046, v4024, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4049 = v32 + 615;
        const int8_t* v4050 = (const int8_t*) v4049;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4051 = *(const int8_t *)(v4050);
        const uint8_t* v4052 = v32 + 743;
        const int8_t* v4053 = (const int8_t*) v4052;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4054 = *(const int8_t *)(v4053);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4055 = __riscv_vwmacc_vx_i16m1(v4016, v4051, v4022, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4056 = __riscv_vwmacc_vx_i16m1(v4017, v4054, v4024, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4057 = v30 + 1632;
        const uint8_t* v4058 = (const uint8_t*) v4057;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4059 = __riscv_vle8_v_u8mf2(v4058, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4060 = __riscv_vand_vx_u8mf2(v4059, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4061 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4060);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4062 = __riscv_vsrl_vx_u8mf2(v4059, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4063 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4062);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4064 = v32 + 616;
        const int8_t* v4065 = (const int8_t*) v4064;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4066 = *(const int8_t *)(v4065);
        const uint8_t* v4067 = v32 + 744;
        const int8_t* v4068 = (const int8_t*) v4067;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4069 = *(const int8_t *)(v4068);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4070 = __riscv_vwmacc_vx_i16m1(v4031, v4066, v4061, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4071 = __riscv_vwmacc_vx_i16m1(v4032, v4069, v4063, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4072 = v32 + 617;
        const int8_t* v4073 = (const int8_t*) v4072;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4074 = *(const int8_t *)(v4073);
        const uint8_t* v4075 = v32 + 745;
        const int8_t* v4076 = (const int8_t*) v4075;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4077 = *(const int8_t *)(v4076);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4078 = __riscv_vwmacc_vx_i16m1(v4039, v4074, v4061, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4079 = __riscv_vwmacc_vx_i16m1(v4040, v4077, v4063, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4080 = v32 + 618;
        const int8_t* v4081 = (const int8_t*) v4080;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4082 = *(const int8_t *)(v4081);
        const uint8_t* v4083 = v32 + 746;
        const int8_t* v4084 = (const int8_t*) v4083;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4085 = *(const int8_t *)(v4084);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4086 = __riscv_vwmacc_vx_i16m1(v4047, v4082, v4061, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4087 = __riscv_vwmacc_vx_i16m1(v4048, v4085, v4063, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4088 = v32 + 619;
        const int8_t* v4089 = (const int8_t*) v4088;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4090 = *(const int8_t *)(v4089);
        const uint8_t* v4091 = v32 + 747;
        const int8_t* v4092 = (const int8_t*) v4091;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4093 = *(const int8_t *)(v4092);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4094 = __riscv_vwmacc_vx_i16m1(v4055, v4090, v4061, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4095 = __riscv_vwmacc_vx_i16m1(v4056, v4093, v4063, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4096 = v30 + 1648;
        const uint8_t* v4097 = (const uint8_t*) v4096;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4098 = __riscv_vle8_v_u8mf2(v4097, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4099 = __riscv_vand_vx_u8mf2(v4098, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4100 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4099);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4101 = __riscv_vsrl_vx_u8mf2(v4098, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4102 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4101);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4103 = v32 + 620;
        const int8_t* v4104 = (const int8_t*) v4103;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4105 = *(const int8_t *)(v4104);
        const uint8_t* v4106 = v32 + 748;
        const int8_t* v4107 = (const int8_t*) v4106;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4108 = *(const int8_t *)(v4107);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4109 = __riscv_vwmacc_vx_i16m1(v4070, v4105, v4100, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4110 = __riscv_vwmacc_vx_i16m1(v4071, v4108, v4102, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4111 = v32 + 621;
        const int8_t* v4112 = (const int8_t*) v4111;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4113 = *(const int8_t *)(v4112);
        const uint8_t* v4114 = v32 + 749;
        const int8_t* v4115 = (const int8_t*) v4114;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4116 = *(const int8_t *)(v4115);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4117 = __riscv_vwmacc_vx_i16m1(v4078, v4113, v4100, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4118 = __riscv_vwmacc_vx_i16m1(v4079, v4116, v4102, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4119 = v32 + 622;
        const int8_t* v4120 = (const int8_t*) v4119;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4121 = *(const int8_t *)(v4120);
        const uint8_t* v4122 = v32 + 750;
        const int8_t* v4123 = (const int8_t*) v4122;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4124 = *(const int8_t *)(v4123);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4125 = __riscv_vwmacc_vx_i16m1(v4086, v4121, v4100, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4126 = __riscv_vwmacc_vx_i16m1(v4087, v4124, v4102, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4127 = v32 + 623;
        const int8_t* v4128 = (const int8_t*) v4127;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4129 = *(const int8_t *)(v4128);
        const uint8_t* v4130 = v32 + 751;
        const int8_t* v4131 = (const int8_t*) v4130;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4132 = *(const int8_t *)(v4131);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4133 = __riscv_vwmacc_vx_i16m1(v4094, v4129, v4100, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4134 = __riscv_vwmacc_vx_i16m1(v4095, v4132, v4102, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4135 = v30 + 1664;
        const uint8_t* v4136 = (const uint8_t*) v4135;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4137 = __riscv_vle8_v_u8mf2(v4136, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4138 = __riscv_vand_vx_u8mf2(v4137, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4139 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4138);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4140 = __riscv_vsrl_vx_u8mf2(v4137, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4141 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4140);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4142 = v32 + 624;
        const int8_t* v4143 = (const int8_t*) v4142;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4144 = *(const int8_t *)(v4143);
        const uint8_t* v4145 = v32 + 752;
        const int8_t* v4146 = (const int8_t*) v4145;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4147 = *(const int8_t *)(v4146);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4148 = __riscv_vwmacc_vx_i16m1(v4109, v4144, v4139, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4149 = __riscv_vwmacc_vx_i16m1(v4110, v4147, v4141, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4150 = v32 + 625;
        const int8_t* v4151 = (const int8_t*) v4150;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4152 = *(const int8_t *)(v4151);
        const uint8_t* v4153 = v32 + 753;
        const int8_t* v4154 = (const int8_t*) v4153;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4155 = *(const int8_t *)(v4154);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4156 = __riscv_vwmacc_vx_i16m1(v4117, v4152, v4139, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4157 = __riscv_vwmacc_vx_i16m1(v4118, v4155, v4141, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4158 = v32 + 626;
        const int8_t* v4159 = (const int8_t*) v4158;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4160 = *(const int8_t *)(v4159);
        const uint8_t* v4161 = v32 + 754;
        const int8_t* v4162 = (const int8_t*) v4161;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4163 = *(const int8_t *)(v4162);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4164 = __riscv_vwmacc_vx_i16m1(v4125, v4160, v4139, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4165 = __riscv_vwmacc_vx_i16m1(v4126, v4163, v4141, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4166 = v32 + 627;
        const int8_t* v4167 = (const int8_t*) v4166;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4168 = *(const int8_t *)(v4167);
        const uint8_t* v4169 = v32 + 755;
        const int8_t* v4170 = (const int8_t*) v4169;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4171 = *(const int8_t *)(v4170);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4172 = __riscv_vwmacc_vx_i16m1(v4133, v4168, v4139, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4173 = __riscv_vwmacc_vx_i16m1(v4134, v4171, v4141, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4174 = v30 + 1680;
        const uint8_t* v4175 = (const uint8_t*) v4174;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4176 = __riscv_vle8_v_u8mf2(v4175, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4177 = __riscv_vand_vx_u8mf2(v4176, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4178 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4177);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4179 = __riscv_vsrl_vx_u8mf2(v4176, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4180 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4179);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4181 = v32 + 628;
        const int8_t* v4182 = (const int8_t*) v4181;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4183 = *(const int8_t *)(v4182);
        const uint8_t* v4184 = v32 + 756;
        const int8_t* v4185 = (const int8_t*) v4184;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4186 = *(const int8_t *)(v4185);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4187 = __riscv_vwmacc_vx_i16m1(v4148, v4183, v4178, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4188 = __riscv_vwmacc_vx_i16m1(v4149, v4186, v4180, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4189 = v32 + 629;
        const int8_t* v4190 = (const int8_t*) v4189;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4191 = *(const int8_t *)(v4190);
        const uint8_t* v4192 = v32 + 757;
        const int8_t* v4193 = (const int8_t*) v4192;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4194 = *(const int8_t *)(v4193);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4195 = __riscv_vwmacc_vx_i16m1(v4156, v4191, v4178, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4196 = __riscv_vwmacc_vx_i16m1(v4157, v4194, v4180, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4197 = v32 + 630;
        const int8_t* v4198 = (const int8_t*) v4197;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4199 = *(const int8_t *)(v4198);
        const uint8_t* v4200 = v32 + 758;
        const int8_t* v4201 = (const int8_t*) v4200;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4202 = *(const int8_t *)(v4201);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4203 = __riscv_vwmacc_vx_i16m1(v4164, v4199, v4178, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4204 = __riscv_vwmacc_vx_i16m1(v4165, v4202, v4180, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4205 = v32 + 631;
        const int8_t* v4206 = (const int8_t*) v4205;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4207 = *(const int8_t *)(v4206);
        const uint8_t* v4208 = v32 + 759;
        const int8_t* v4209 = (const int8_t*) v4208;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4210 = *(const int8_t *)(v4209);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4211 = __riscv_vwmacc_vx_i16m1(v4172, v4207, v4178, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4212 = __riscv_vwmacc_vx_i16m1(v4173, v4210, v4180, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4213 = v30 + 1696;
        const uint8_t* v4214 = (const uint8_t*) v4213;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4215 = __riscv_vle8_v_u8mf2(v4214, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4216 = __riscv_vand_vx_u8mf2(v4215, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4217 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4216);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4218 = __riscv_vsrl_vx_u8mf2(v4215, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4219 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4218);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4220 = v32 + 632;
        const int8_t* v4221 = (const int8_t*) v4220;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4222 = *(const int8_t *)(v4221);
        const uint8_t* v4223 = v32 + 760;
        const int8_t* v4224 = (const int8_t*) v4223;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4225 = *(const int8_t *)(v4224);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4226 = __riscv_vwmacc_vx_i16m1(v4187, v4222, v4217, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4227 = __riscv_vwmacc_vx_i16m1(v4188, v4225, v4219, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4228 = v32 + 633;
        const int8_t* v4229 = (const int8_t*) v4228;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4230 = *(const int8_t *)(v4229);
        const uint8_t* v4231 = v32 + 761;
        const int8_t* v4232 = (const int8_t*) v4231;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4233 = *(const int8_t *)(v4232);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4234 = __riscv_vwmacc_vx_i16m1(v4195, v4230, v4217, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4235 = __riscv_vwmacc_vx_i16m1(v4196, v4233, v4219, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4236 = v32 + 634;
        const int8_t* v4237 = (const int8_t*) v4236;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4238 = *(const int8_t *)(v4237);
        const uint8_t* v4239 = v32 + 762;
        const int8_t* v4240 = (const int8_t*) v4239;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4241 = *(const int8_t *)(v4240);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4242 = __riscv_vwmacc_vx_i16m1(v4203, v4238, v4217, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4243 = __riscv_vwmacc_vx_i16m1(v4204, v4241, v4219, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4244 = v32 + 635;
        const int8_t* v4245 = (const int8_t*) v4244;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4246 = *(const int8_t *)(v4245);
        const uint8_t* v4247 = v32 + 763;
        const int8_t* v4248 = (const int8_t*) v4247;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4249 = *(const int8_t *)(v4248);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4250 = __riscv_vwmacc_vx_i16m1(v4211, v4246, v4217, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4251 = __riscv_vwmacc_vx_i16m1(v4212, v4249, v4219, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4252 = v30 + 1712;
        const uint8_t* v4253 = (const uint8_t*) v4252;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4254 = __riscv_vle8_v_u8mf2(v4253, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4255 = __riscv_vand_vx_u8mf2(v4254, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4256 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4255);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4257 = __riscv_vsrl_vx_u8mf2(v4254, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4258 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4257);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4259 = v32 + 636;
        const int8_t* v4260 = (const int8_t*) v4259;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4261 = *(const int8_t *)(v4260);
        const uint8_t* v4262 = v32 + 764;
        const int8_t* v4263 = (const int8_t*) v4262;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4264 = *(const int8_t *)(v4263);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4265 = __riscv_vwmacc_vx_i16m1(v4226, v4261, v4256, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4266 = __riscv_vwmacc_vx_i16m1(v4227, v4264, v4258, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4267 = v32 + 637;
        const int8_t* v4268 = (const int8_t*) v4267;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4269 = *(const int8_t *)(v4268);
        const uint8_t* v4270 = v32 + 765;
        const int8_t* v4271 = (const int8_t*) v4270;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4272 = *(const int8_t *)(v4271);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4273 = __riscv_vwmacc_vx_i16m1(v4234, v4269, v4256, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4274 = __riscv_vwmacc_vx_i16m1(v4235, v4272, v4258, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4275 = v32 + 638;
        const int8_t* v4276 = (const int8_t*) v4275;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4277 = *(const int8_t *)(v4276);
        const uint8_t* v4278 = v32 + 766;
        const int8_t* v4279 = (const int8_t*) v4278;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4280 = *(const int8_t *)(v4279);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4281 = __riscv_vwmacc_vx_i16m1(v4242, v4277, v4256, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4282 = __riscv_vwmacc_vx_i16m1(v4243, v4280, v4258, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4283 = v32 + 639;
        const int8_t* v4284 = (const int8_t*) v4283;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4285 = *(const int8_t *)(v4284);
        const uint8_t* v4286 = v32 + 767;
        const int8_t* v4287 = (const int8_t*) v4286;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4288 = *(const int8_t *)(v4287);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4289 = __riscv_vwmacc_vx_i16m1(v4250, v4285, v4256, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4290 = __riscv_vwmacc_vx_i16m1(v4251, v4288, v4258, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4291 = v30 + 1728;
        const uint8_t* v4292 = (const uint8_t*) v4291;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4293 = __riscv_vle8_v_u8mf2(v4292, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4294 = __riscv_vand_vx_u8mf2(v4293, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4295 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4294);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4296 = __riscv_vsrl_vx_u8mf2(v4293, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4297 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4296);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4298 = v32 + 640;
        const int8_t* v4299 = (const int8_t*) v4298;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4300 = *(const int8_t *)(v4299);
        const uint8_t* v4301 = v32 + 768;
        const int8_t* v4302 = (const int8_t*) v4301;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4303 = *(const int8_t *)(v4302);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4304 = __riscv_vwmacc_vx_i16m1(v4265, v4300, v4295, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4305 = __riscv_vwmacc_vx_i16m1(v4266, v4303, v4297, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4306 = v32 + 641;
        const int8_t* v4307 = (const int8_t*) v4306;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4308 = *(const int8_t *)(v4307);
        const uint8_t* v4309 = v32 + 769;
        const int8_t* v4310 = (const int8_t*) v4309;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4311 = *(const int8_t *)(v4310);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4312 = __riscv_vwmacc_vx_i16m1(v4273, v4308, v4295, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4313 = __riscv_vwmacc_vx_i16m1(v4274, v4311, v4297, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4314 = v32 + 642;
        const int8_t* v4315 = (const int8_t*) v4314;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4316 = *(const int8_t *)(v4315);
        const uint8_t* v4317 = v32 + 770;
        const int8_t* v4318 = (const int8_t*) v4317;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4319 = *(const int8_t *)(v4318);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4320 = __riscv_vwmacc_vx_i16m1(v4281, v4316, v4295, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4321 = __riscv_vwmacc_vx_i16m1(v4282, v4319, v4297, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4322 = v32 + 643;
        const int8_t* v4323 = (const int8_t*) v4322;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4324 = *(const int8_t *)(v4323);
        const uint8_t* v4325 = v32 + 771;
        const int8_t* v4326 = (const int8_t*) v4325;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4327 = *(const int8_t *)(v4326);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4328 = __riscv_vwmacc_vx_i16m1(v4289, v4324, v4295, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4329 = __riscv_vwmacc_vx_i16m1(v4290, v4327, v4297, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4330 = v30 + 1744;
        const uint8_t* v4331 = (const uint8_t*) v4330;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4332 = __riscv_vle8_v_u8mf2(v4331, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4333 = __riscv_vand_vx_u8mf2(v4332, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4334 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4333);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4335 = __riscv_vsrl_vx_u8mf2(v4332, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4336 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4335);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4337 = v32 + 644;
        const int8_t* v4338 = (const int8_t*) v4337;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4339 = *(const int8_t *)(v4338);
        const uint8_t* v4340 = v32 + 772;
        const int8_t* v4341 = (const int8_t*) v4340;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4342 = *(const int8_t *)(v4341);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4343 = __riscv_vwmacc_vx_i16m1(v4304, v4339, v4334, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4344 = __riscv_vwmacc_vx_i16m1(v4305, v4342, v4336, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4345 = v32 + 645;
        const int8_t* v4346 = (const int8_t*) v4345;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4347 = *(const int8_t *)(v4346);
        const uint8_t* v4348 = v32 + 773;
        const int8_t* v4349 = (const int8_t*) v4348;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4350 = *(const int8_t *)(v4349);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4351 = __riscv_vwmacc_vx_i16m1(v4312, v4347, v4334, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4352 = __riscv_vwmacc_vx_i16m1(v4313, v4350, v4336, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4353 = v32 + 646;
        const int8_t* v4354 = (const int8_t*) v4353;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4355 = *(const int8_t *)(v4354);
        const uint8_t* v4356 = v32 + 774;
        const int8_t* v4357 = (const int8_t*) v4356;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4358 = *(const int8_t *)(v4357);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4359 = __riscv_vwmacc_vx_i16m1(v4320, v4355, v4334, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4360 = __riscv_vwmacc_vx_i16m1(v4321, v4358, v4336, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4361 = v32 + 647;
        const int8_t* v4362 = (const int8_t*) v4361;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4363 = *(const int8_t *)(v4362);
        const uint8_t* v4364 = v32 + 775;
        const int8_t* v4365 = (const int8_t*) v4364;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4366 = *(const int8_t *)(v4365);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4367 = __riscv_vwmacc_vx_i16m1(v4328, v4363, v4334, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4368 = __riscv_vwmacc_vx_i16m1(v4329, v4366, v4336, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4369 = v30 + 1760;
        const uint8_t* v4370 = (const uint8_t*) v4369;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4371 = __riscv_vle8_v_u8mf2(v4370, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4372 = __riscv_vand_vx_u8mf2(v4371, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4373 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4372);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4374 = __riscv_vsrl_vx_u8mf2(v4371, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4375 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4374);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4376 = v32 + 648;
        const int8_t* v4377 = (const int8_t*) v4376;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4378 = *(const int8_t *)(v4377);
        const uint8_t* v4379 = v32 + 776;
        const int8_t* v4380 = (const int8_t*) v4379;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4381 = *(const int8_t *)(v4380);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4382 = __riscv_vwmacc_vx_i16m1(v4343, v4378, v4373, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4383 = __riscv_vwmacc_vx_i16m1(v4344, v4381, v4375, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4384 = v32 + 649;
        const int8_t* v4385 = (const int8_t*) v4384;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4386 = *(const int8_t *)(v4385);
        const uint8_t* v4387 = v32 + 777;
        const int8_t* v4388 = (const int8_t*) v4387;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4389 = *(const int8_t *)(v4388);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4390 = __riscv_vwmacc_vx_i16m1(v4351, v4386, v4373, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4391 = __riscv_vwmacc_vx_i16m1(v4352, v4389, v4375, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4392 = v32 + 650;
        const int8_t* v4393 = (const int8_t*) v4392;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4394 = *(const int8_t *)(v4393);
        const uint8_t* v4395 = v32 + 778;
        const int8_t* v4396 = (const int8_t*) v4395;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4397 = *(const int8_t *)(v4396);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4398 = __riscv_vwmacc_vx_i16m1(v4359, v4394, v4373, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4399 = __riscv_vwmacc_vx_i16m1(v4360, v4397, v4375, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4400 = v32 + 651;
        const int8_t* v4401 = (const int8_t*) v4400;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4402 = *(const int8_t *)(v4401);
        const uint8_t* v4403 = v32 + 779;
        const int8_t* v4404 = (const int8_t*) v4403;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4405 = *(const int8_t *)(v4404);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4406 = __riscv_vwmacc_vx_i16m1(v4367, v4402, v4373, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4407 = __riscv_vwmacc_vx_i16m1(v4368, v4405, v4375, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4408 = v30 + 1776;
        const uint8_t* v4409 = (const uint8_t*) v4408;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4410 = __riscv_vle8_v_u8mf2(v4409, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4411 = __riscv_vand_vx_u8mf2(v4410, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4412 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4411);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4413 = __riscv_vsrl_vx_u8mf2(v4410, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4414 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4413);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4415 = v32 + 652;
        const int8_t* v4416 = (const int8_t*) v4415;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4417 = *(const int8_t *)(v4416);
        const uint8_t* v4418 = v32 + 780;
        const int8_t* v4419 = (const int8_t*) v4418;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4420 = *(const int8_t *)(v4419);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4421 = __riscv_vwmacc_vx_i16m1(v4382, v4417, v4412, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4422 = __riscv_vwmacc_vx_i16m1(v4383, v4420, v4414, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4423 = v32 + 653;
        const int8_t* v4424 = (const int8_t*) v4423;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4425 = *(const int8_t *)(v4424);
        const uint8_t* v4426 = v32 + 781;
        const int8_t* v4427 = (const int8_t*) v4426;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4428 = *(const int8_t *)(v4427);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4429 = __riscv_vwmacc_vx_i16m1(v4390, v4425, v4412, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4430 = __riscv_vwmacc_vx_i16m1(v4391, v4428, v4414, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4431 = v32 + 654;
        const int8_t* v4432 = (const int8_t*) v4431;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4433 = *(const int8_t *)(v4432);
        const uint8_t* v4434 = v32 + 782;
        const int8_t* v4435 = (const int8_t*) v4434;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4436 = *(const int8_t *)(v4435);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4437 = __riscv_vwmacc_vx_i16m1(v4398, v4433, v4412, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4438 = __riscv_vwmacc_vx_i16m1(v4399, v4436, v4414, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4439 = v32 + 655;
        const int8_t* v4440 = (const int8_t*) v4439;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4441 = *(const int8_t *)(v4440);
        const uint8_t* v4442 = v32 + 783;
        const int8_t* v4443 = (const int8_t*) v4442;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4444 = *(const int8_t *)(v4443);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4445 = __riscv_vwmacc_vx_i16m1(v4406, v4441, v4412, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4446 = __riscv_vwmacc_vx_i16m1(v4407, v4444, v4414, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v4447 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v4448 = __riscv_vle16_v_i16m1(v4447, 16);
        int16_t* v4449 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v4450 = __riscv_vle16_v_i16m1(v4449, 16);
        vint32m2_t v4451 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v4452 = __riscv_vwmacc_vv_i32m2(v4451, v4448, v4421, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v4453 = __riscv_vwmacc_vv_i32m2(v4452, v4450, v4422, 16);
        v47 = v4453;
        vint32m2_t v4454 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v4455 = __riscv_vwmacc_vv_i32m2(v4454, v4448, v4429, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v4456 = __riscv_vwmacc_vv_i32m2(v4455, v4450, v4430, 16);
        v49 = v4456;
        vint32m2_t v4457 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v4458 = __riscv_vwmacc_vv_i32m2(v4457, v4448, v4437, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v4459 = __riscv_vwmacc_vv_i32m2(v4458, v4450, v4438, 16);
        v51 = v4459;
        vint32m2_t v4460 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v4461 = __riscv_vwmacc_vv_i32m2(v4460, v4448, v4445, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v4462 = __riscv_vwmacc_vv_i32m2(v4461, v4450, v4446, 16);
        v53 = v4462;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v4463 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v4464 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v4465 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v4466 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v4467 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v4468 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v4469 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v4470 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4471 = v30 + 1792;
        const uint8_t* v4472 = (const uint8_t*) v4471;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4473 = __riscv_vle8_v_u8mf2(v4472, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4474 = __riscv_vand_vx_u8mf2(v4473, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4475 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4474);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4476 = __riscv_vsrl_vx_u8mf2(v4473, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4477 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4476);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4478 = v32 + 784;
        const int8_t* v4479 = (const int8_t*) v4478;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4480 = *(const int8_t *)(v4479);
        const uint8_t* v4481 = v32 + 912;
        const int8_t* v4482 = (const int8_t*) v4481;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4483 = *(const int8_t *)(v4482);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4484 = __riscv_vwmacc_vx_i16m1(v4463, v4480, v4475, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4485 = __riscv_vwmacc_vx_i16m1(v4464, v4483, v4477, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4486 = v32 + 785;
        const int8_t* v4487 = (const int8_t*) v4486;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4488 = *(const int8_t *)(v4487);
        const uint8_t* v4489 = v32 + 913;
        const int8_t* v4490 = (const int8_t*) v4489;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4491 = *(const int8_t *)(v4490);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4492 = __riscv_vwmacc_vx_i16m1(v4465, v4488, v4475, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4493 = __riscv_vwmacc_vx_i16m1(v4466, v4491, v4477, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4494 = v32 + 786;
        const int8_t* v4495 = (const int8_t*) v4494;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4496 = *(const int8_t *)(v4495);
        const uint8_t* v4497 = v32 + 914;
        const int8_t* v4498 = (const int8_t*) v4497;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4499 = *(const int8_t *)(v4498);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4500 = __riscv_vwmacc_vx_i16m1(v4467, v4496, v4475, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4501 = __riscv_vwmacc_vx_i16m1(v4468, v4499, v4477, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4502 = v32 + 787;
        const int8_t* v4503 = (const int8_t*) v4502;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4504 = *(const int8_t *)(v4503);
        const uint8_t* v4505 = v32 + 915;
        const int8_t* v4506 = (const int8_t*) v4505;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4507 = *(const int8_t *)(v4506);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4508 = __riscv_vwmacc_vx_i16m1(v4469, v4504, v4475, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4509 = __riscv_vwmacc_vx_i16m1(v4470, v4507, v4477, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4510 = v30 + 1808;
        const uint8_t* v4511 = (const uint8_t*) v4510;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4512 = __riscv_vle8_v_u8mf2(v4511, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4513 = __riscv_vand_vx_u8mf2(v4512, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4514 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4513);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4515 = __riscv_vsrl_vx_u8mf2(v4512, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4516 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4515);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4517 = v32 + 788;
        const int8_t* v4518 = (const int8_t*) v4517;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4519 = *(const int8_t *)(v4518);
        const uint8_t* v4520 = v32 + 916;
        const int8_t* v4521 = (const int8_t*) v4520;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4522 = *(const int8_t *)(v4521);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4523 = __riscv_vwmacc_vx_i16m1(v4484, v4519, v4514, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4524 = __riscv_vwmacc_vx_i16m1(v4485, v4522, v4516, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4525 = v32 + 789;
        const int8_t* v4526 = (const int8_t*) v4525;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4527 = *(const int8_t *)(v4526);
        const uint8_t* v4528 = v32 + 917;
        const int8_t* v4529 = (const int8_t*) v4528;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4530 = *(const int8_t *)(v4529);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4531 = __riscv_vwmacc_vx_i16m1(v4492, v4527, v4514, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4532 = __riscv_vwmacc_vx_i16m1(v4493, v4530, v4516, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4533 = v32 + 790;
        const int8_t* v4534 = (const int8_t*) v4533;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4535 = *(const int8_t *)(v4534);
        const uint8_t* v4536 = v32 + 918;
        const int8_t* v4537 = (const int8_t*) v4536;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4538 = *(const int8_t *)(v4537);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4539 = __riscv_vwmacc_vx_i16m1(v4500, v4535, v4514, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4540 = __riscv_vwmacc_vx_i16m1(v4501, v4538, v4516, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4541 = v32 + 791;
        const int8_t* v4542 = (const int8_t*) v4541;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4543 = *(const int8_t *)(v4542);
        const uint8_t* v4544 = v32 + 919;
        const int8_t* v4545 = (const int8_t*) v4544;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4546 = *(const int8_t *)(v4545);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4547 = __riscv_vwmacc_vx_i16m1(v4508, v4543, v4514, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4548 = __riscv_vwmacc_vx_i16m1(v4509, v4546, v4516, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4549 = v30 + 1824;
        const uint8_t* v4550 = (const uint8_t*) v4549;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4551 = __riscv_vle8_v_u8mf2(v4550, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4552 = __riscv_vand_vx_u8mf2(v4551, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4553 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4552);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4554 = __riscv_vsrl_vx_u8mf2(v4551, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4555 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4554);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4556 = v32 + 792;
        const int8_t* v4557 = (const int8_t*) v4556;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4558 = *(const int8_t *)(v4557);
        const uint8_t* v4559 = v32 + 920;
        const int8_t* v4560 = (const int8_t*) v4559;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4561 = *(const int8_t *)(v4560);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4562 = __riscv_vwmacc_vx_i16m1(v4523, v4558, v4553, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4563 = __riscv_vwmacc_vx_i16m1(v4524, v4561, v4555, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4564 = v32 + 793;
        const int8_t* v4565 = (const int8_t*) v4564;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4566 = *(const int8_t *)(v4565);
        const uint8_t* v4567 = v32 + 921;
        const int8_t* v4568 = (const int8_t*) v4567;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4569 = *(const int8_t *)(v4568);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4570 = __riscv_vwmacc_vx_i16m1(v4531, v4566, v4553, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4571 = __riscv_vwmacc_vx_i16m1(v4532, v4569, v4555, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4572 = v32 + 794;
        const int8_t* v4573 = (const int8_t*) v4572;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4574 = *(const int8_t *)(v4573);
        const uint8_t* v4575 = v32 + 922;
        const int8_t* v4576 = (const int8_t*) v4575;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4577 = *(const int8_t *)(v4576);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4578 = __riscv_vwmacc_vx_i16m1(v4539, v4574, v4553, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4579 = __riscv_vwmacc_vx_i16m1(v4540, v4577, v4555, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4580 = v32 + 795;
        const int8_t* v4581 = (const int8_t*) v4580;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4582 = *(const int8_t *)(v4581);
        const uint8_t* v4583 = v32 + 923;
        const int8_t* v4584 = (const int8_t*) v4583;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4585 = *(const int8_t *)(v4584);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4586 = __riscv_vwmacc_vx_i16m1(v4547, v4582, v4553, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4587 = __riscv_vwmacc_vx_i16m1(v4548, v4585, v4555, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4588 = v30 + 1840;
        const uint8_t* v4589 = (const uint8_t*) v4588;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4590 = __riscv_vle8_v_u8mf2(v4589, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4591 = __riscv_vand_vx_u8mf2(v4590, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4592 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4591);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4593 = __riscv_vsrl_vx_u8mf2(v4590, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4594 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4593);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4595 = v32 + 796;
        const int8_t* v4596 = (const int8_t*) v4595;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4597 = *(const int8_t *)(v4596);
        const uint8_t* v4598 = v32 + 924;
        const int8_t* v4599 = (const int8_t*) v4598;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4600 = *(const int8_t *)(v4599);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4601 = __riscv_vwmacc_vx_i16m1(v4562, v4597, v4592, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4602 = __riscv_vwmacc_vx_i16m1(v4563, v4600, v4594, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4603 = v32 + 797;
        const int8_t* v4604 = (const int8_t*) v4603;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4605 = *(const int8_t *)(v4604);
        const uint8_t* v4606 = v32 + 925;
        const int8_t* v4607 = (const int8_t*) v4606;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4608 = *(const int8_t *)(v4607);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4609 = __riscv_vwmacc_vx_i16m1(v4570, v4605, v4592, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4610 = __riscv_vwmacc_vx_i16m1(v4571, v4608, v4594, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4611 = v32 + 798;
        const int8_t* v4612 = (const int8_t*) v4611;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4613 = *(const int8_t *)(v4612);
        const uint8_t* v4614 = v32 + 926;
        const int8_t* v4615 = (const int8_t*) v4614;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4616 = *(const int8_t *)(v4615);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4617 = __riscv_vwmacc_vx_i16m1(v4578, v4613, v4592, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4618 = __riscv_vwmacc_vx_i16m1(v4579, v4616, v4594, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4619 = v32 + 799;
        const int8_t* v4620 = (const int8_t*) v4619;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4621 = *(const int8_t *)(v4620);
        const uint8_t* v4622 = v32 + 927;
        const int8_t* v4623 = (const int8_t*) v4622;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4624 = *(const int8_t *)(v4623);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4625 = __riscv_vwmacc_vx_i16m1(v4586, v4621, v4592, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4626 = __riscv_vwmacc_vx_i16m1(v4587, v4624, v4594, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4627 = v30 + 1856;
        const uint8_t* v4628 = (const uint8_t*) v4627;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4629 = __riscv_vle8_v_u8mf2(v4628, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4630 = __riscv_vand_vx_u8mf2(v4629, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4631 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4630);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4632 = __riscv_vsrl_vx_u8mf2(v4629, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4633 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4632);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4634 = v32 + 800;
        const int8_t* v4635 = (const int8_t*) v4634;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4636 = *(const int8_t *)(v4635);
        const uint8_t* v4637 = v32 + 928;
        const int8_t* v4638 = (const int8_t*) v4637;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4639 = *(const int8_t *)(v4638);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4640 = __riscv_vwmacc_vx_i16m1(v4601, v4636, v4631, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4641 = __riscv_vwmacc_vx_i16m1(v4602, v4639, v4633, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4642 = v32 + 801;
        const int8_t* v4643 = (const int8_t*) v4642;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4644 = *(const int8_t *)(v4643);
        const uint8_t* v4645 = v32 + 929;
        const int8_t* v4646 = (const int8_t*) v4645;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4647 = *(const int8_t *)(v4646);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4648 = __riscv_vwmacc_vx_i16m1(v4609, v4644, v4631, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4649 = __riscv_vwmacc_vx_i16m1(v4610, v4647, v4633, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4650 = v32 + 802;
        const int8_t* v4651 = (const int8_t*) v4650;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4652 = *(const int8_t *)(v4651);
        const uint8_t* v4653 = v32 + 930;
        const int8_t* v4654 = (const int8_t*) v4653;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4655 = *(const int8_t *)(v4654);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4656 = __riscv_vwmacc_vx_i16m1(v4617, v4652, v4631, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4657 = __riscv_vwmacc_vx_i16m1(v4618, v4655, v4633, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4658 = v32 + 803;
        const int8_t* v4659 = (const int8_t*) v4658;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4660 = *(const int8_t *)(v4659);
        const uint8_t* v4661 = v32 + 931;
        const int8_t* v4662 = (const int8_t*) v4661;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4663 = *(const int8_t *)(v4662);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4664 = __riscv_vwmacc_vx_i16m1(v4625, v4660, v4631, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4665 = __riscv_vwmacc_vx_i16m1(v4626, v4663, v4633, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4666 = v30 + 1872;
        const uint8_t* v4667 = (const uint8_t*) v4666;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4668 = __riscv_vle8_v_u8mf2(v4667, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4669 = __riscv_vand_vx_u8mf2(v4668, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4670 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4669);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4671 = __riscv_vsrl_vx_u8mf2(v4668, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4672 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4671);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4673 = v32 + 804;
        const int8_t* v4674 = (const int8_t*) v4673;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4675 = *(const int8_t *)(v4674);
        const uint8_t* v4676 = v32 + 932;
        const int8_t* v4677 = (const int8_t*) v4676;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4678 = *(const int8_t *)(v4677);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4679 = __riscv_vwmacc_vx_i16m1(v4640, v4675, v4670, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4680 = __riscv_vwmacc_vx_i16m1(v4641, v4678, v4672, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4681 = v32 + 805;
        const int8_t* v4682 = (const int8_t*) v4681;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4683 = *(const int8_t *)(v4682);
        const uint8_t* v4684 = v32 + 933;
        const int8_t* v4685 = (const int8_t*) v4684;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4686 = *(const int8_t *)(v4685);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4687 = __riscv_vwmacc_vx_i16m1(v4648, v4683, v4670, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4688 = __riscv_vwmacc_vx_i16m1(v4649, v4686, v4672, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4689 = v32 + 806;
        const int8_t* v4690 = (const int8_t*) v4689;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4691 = *(const int8_t *)(v4690);
        const uint8_t* v4692 = v32 + 934;
        const int8_t* v4693 = (const int8_t*) v4692;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4694 = *(const int8_t *)(v4693);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4695 = __riscv_vwmacc_vx_i16m1(v4656, v4691, v4670, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4696 = __riscv_vwmacc_vx_i16m1(v4657, v4694, v4672, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4697 = v32 + 807;
        const int8_t* v4698 = (const int8_t*) v4697;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4699 = *(const int8_t *)(v4698);
        const uint8_t* v4700 = v32 + 935;
        const int8_t* v4701 = (const int8_t*) v4700;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4702 = *(const int8_t *)(v4701);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4703 = __riscv_vwmacc_vx_i16m1(v4664, v4699, v4670, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4704 = __riscv_vwmacc_vx_i16m1(v4665, v4702, v4672, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4705 = v30 + 1888;
        const uint8_t* v4706 = (const uint8_t*) v4705;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4707 = __riscv_vle8_v_u8mf2(v4706, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4708 = __riscv_vand_vx_u8mf2(v4707, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4709 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4708);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4710 = __riscv_vsrl_vx_u8mf2(v4707, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4711 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4710);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4712 = v32 + 808;
        const int8_t* v4713 = (const int8_t*) v4712;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4714 = *(const int8_t *)(v4713);
        const uint8_t* v4715 = v32 + 936;
        const int8_t* v4716 = (const int8_t*) v4715;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4717 = *(const int8_t *)(v4716);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4718 = __riscv_vwmacc_vx_i16m1(v4679, v4714, v4709, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4719 = __riscv_vwmacc_vx_i16m1(v4680, v4717, v4711, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4720 = v32 + 809;
        const int8_t* v4721 = (const int8_t*) v4720;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4722 = *(const int8_t *)(v4721);
        const uint8_t* v4723 = v32 + 937;
        const int8_t* v4724 = (const int8_t*) v4723;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4725 = *(const int8_t *)(v4724);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4726 = __riscv_vwmacc_vx_i16m1(v4687, v4722, v4709, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4727 = __riscv_vwmacc_vx_i16m1(v4688, v4725, v4711, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4728 = v32 + 810;
        const int8_t* v4729 = (const int8_t*) v4728;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4730 = *(const int8_t *)(v4729);
        const uint8_t* v4731 = v32 + 938;
        const int8_t* v4732 = (const int8_t*) v4731;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4733 = *(const int8_t *)(v4732);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4734 = __riscv_vwmacc_vx_i16m1(v4695, v4730, v4709, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4735 = __riscv_vwmacc_vx_i16m1(v4696, v4733, v4711, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4736 = v32 + 811;
        const int8_t* v4737 = (const int8_t*) v4736;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4738 = *(const int8_t *)(v4737);
        const uint8_t* v4739 = v32 + 939;
        const int8_t* v4740 = (const int8_t*) v4739;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4741 = *(const int8_t *)(v4740);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4742 = __riscv_vwmacc_vx_i16m1(v4703, v4738, v4709, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4743 = __riscv_vwmacc_vx_i16m1(v4704, v4741, v4711, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4744 = v30 + 1904;
        const uint8_t* v4745 = (const uint8_t*) v4744;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4746 = __riscv_vle8_v_u8mf2(v4745, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4747 = __riscv_vand_vx_u8mf2(v4746, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4748 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4747);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4749 = __riscv_vsrl_vx_u8mf2(v4746, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4750 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4749);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4751 = v32 + 812;
        const int8_t* v4752 = (const int8_t*) v4751;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4753 = *(const int8_t *)(v4752);
        const uint8_t* v4754 = v32 + 940;
        const int8_t* v4755 = (const int8_t*) v4754;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4756 = *(const int8_t *)(v4755);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4757 = __riscv_vwmacc_vx_i16m1(v4718, v4753, v4748, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4758 = __riscv_vwmacc_vx_i16m1(v4719, v4756, v4750, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4759 = v32 + 813;
        const int8_t* v4760 = (const int8_t*) v4759;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4761 = *(const int8_t *)(v4760);
        const uint8_t* v4762 = v32 + 941;
        const int8_t* v4763 = (const int8_t*) v4762;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4764 = *(const int8_t *)(v4763);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4765 = __riscv_vwmacc_vx_i16m1(v4726, v4761, v4748, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4766 = __riscv_vwmacc_vx_i16m1(v4727, v4764, v4750, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4767 = v32 + 814;
        const int8_t* v4768 = (const int8_t*) v4767;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4769 = *(const int8_t *)(v4768);
        const uint8_t* v4770 = v32 + 942;
        const int8_t* v4771 = (const int8_t*) v4770;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4772 = *(const int8_t *)(v4771);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4773 = __riscv_vwmacc_vx_i16m1(v4734, v4769, v4748, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4774 = __riscv_vwmacc_vx_i16m1(v4735, v4772, v4750, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4775 = v32 + 815;
        const int8_t* v4776 = (const int8_t*) v4775;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4777 = *(const int8_t *)(v4776);
        const uint8_t* v4778 = v32 + 943;
        const int8_t* v4779 = (const int8_t*) v4778;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4780 = *(const int8_t *)(v4779);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4781 = __riscv_vwmacc_vx_i16m1(v4742, v4777, v4748, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4782 = __riscv_vwmacc_vx_i16m1(v4743, v4780, v4750, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4783 = v30 + 1920;
        const uint8_t* v4784 = (const uint8_t*) v4783;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4785 = __riscv_vle8_v_u8mf2(v4784, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4786 = __riscv_vand_vx_u8mf2(v4785, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4787 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4786);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4788 = __riscv_vsrl_vx_u8mf2(v4785, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4789 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4788);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4790 = v32 + 816;
        const int8_t* v4791 = (const int8_t*) v4790;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4792 = *(const int8_t *)(v4791);
        const uint8_t* v4793 = v32 + 944;
        const int8_t* v4794 = (const int8_t*) v4793;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4795 = *(const int8_t *)(v4794);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4796 = __riscv_vwmacc_vx_i16m1(v4757, v4792, v4787, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4797 = __riscv_vwmacc_vx_i16m1(v4758, v4795, v4789, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4798 = v32 + 817;
        const int8_t* v4799 = (const int8_t*) v4798;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4800 = *(const int8_t *)(v4799);
        const uint8_t* v4801 = v32 + 945;
        const int8_t* v4802 = (const int8_t*) v4801;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4803 = *(const int8_t *)(v4802);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4804 = __riscv_vwmacc_vx_i16m1(v4765, v4800, v4787, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4805 = __riscv_vwmacc_vx_i16m1(v4766, v4803, v4789, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4806 = v32 + 818;
        const int8_t* v4807 = (const int8_t*) v4806;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4808 = *(const int8_t *)(v4807);
        const uint8_t* v4809 = v32 + 946;
        const int8_t* v4810 = (const int8_t*) v4809;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4811 = *(const int8_t *)(v4810);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4812 = __riscv_vwmacc_vx_i16m1(v4773, v4808, v4787, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4813 = __riscv_vwmacc_vx_i16m1(v4774, v4811, v4789, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4814 = v32 + 819;
        const int8_t* v4815 = (const int8_t*) v4814;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4816 = *(const int8_t *)(v4815);
        const uint8_t* v4817 = v32 + 947;
        const int8_t* v4818 = (const int8_t*) v4817;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4819 = *(const int8_t *)(v4818);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4820 = __riscv_vwmacc_vx_i16m1(v4781, v4816, v4787, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4821 = __riscv_vwmacc_vx_i16m1(v4782, v4819, v4789, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4822 = v30 + 1936;
        const uint8_t* v4823 = (const uint8_t*) v4822;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4824 = __riscv_vle8_v_u8mf2(v4823, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4825 = __riscv_vand_vx_u8mf2(v4824, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4826 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4825);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4827 = __riscv_vsrl_vx_u8mf2(v4824, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4828 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4827);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4829 = v32 + 820;
        const int8_t* v4830 = (const int8_t*) v4829;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4831 = *(const int8_t *)(v4830);
        const uint8_t* v4832 = v32 + 948;
        const int8_t* v4833 = (const int8_t*) v4832;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4834 = *(const int8_t *)(v4833);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4835 = __riscv_vwmacc_vx_i16m1(v4796, v4831, v4826, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4836 = __riscv_vwmacc_vx_i16m1(v4797, v4834, v4828, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4837 = v32 + 821;
        const int8_t* v4838 = (const int8_t*) v4837;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4839 = *(const int8_t *)(v4838);
        const uint8_t* v4840 = v32 + 949;
        const int8_t* v4841 = (const int8_t*) v4840;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4842 = *(const int8_t *)(v4841);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4843 = __riscv_vwmacc_vx_i16m1(v4804, v4839, v4826, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4844 = __riscv_vwmacc_vx_i16m1(v4805, v4842, v4828, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4845 = v32 + 822;
        const int8_t* v4846 = (const int8_t*) v4845;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4847 = *(const int8_t *)(v4846);
        const uint8_t* v4848 = v32 + 950;
        const int8_t* v4849 = (const int8_t*) v4848;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4850 = *(const int8_t *)(v4849);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4851 = __riscv_vwmacc_vx_i16m1(v4812, v4847, v4826, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4852 = __riscv_vwmacc_vx_i16m1(v4813, v4850, v4828, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4853 = v32 + 823;
        const int8_t* v4854 = (const int8_t*) v4853;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4855 = *(const int8_t *)(v4854);
        const uint8_t* v4856 = v32 + 951;
        const int8_t* v4857 = (const int8_t*) v4856;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4858 = *(const int8_t *)(v4857);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4859 = __riscv_vwmacc_vx_i16m1(v4820, v4855, v4826, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4860 = __riscv_vwmacc_vx_i16m1(v4821, v4858, v4828, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4861 = v30 + 1952;
        const uint8_t* v4862 = (const uint8_t*) v4861;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4863 = __riscv_vle8_v_u8mf2(v4862, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4864 = __riscv_vand_vx_u8mf2(v4863, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4865 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4864);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4866 = __riscv_vsrl_vx_u8mf2(v4863, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4867 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4866);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4868 = v32 + 824;
        const int8_t* v4869 = (const int8_t*) v4868;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4870 = *(const int8_t *)(v4869);
        const uint8_t* v4871 = v32 + 952;
        const int8_t* v4872 = (const int8_t*) v4871;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4873 = *(const int8_t *)(v4872);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4874 = __riscv_vwmacc_vx_i16m1(v4835, v4870, v4865, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4875 = __riscv_vwmacc_vx_i16m1(v4836, v4873, v4867, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4876 = v32 + 825;
        const int8_t* v4877 = (const int8_t*) v4876;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4878 = *(const int8_t *)(v4877);
        const uint8_t* v4879 = v32 + 953;
        const int8_t* v4880 = (const int8_t*) v4879;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4881 = *(const int8_t *)(v4880);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4882 = __riscv_vwmacc_vx_i16m1(v4843, v4878, v4865, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4883 = __riscv_vwmacc_vx_i16m1(v4844, v4881, v4867, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4884 = v32 + 826;
        const int8_t* v4885 = (const int8_t*) v4884;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4886 = *(const int8_t *)(v4885);
        const uint8_t* v4887 = v32 + 954;
        const int8_t* v4888 = (const int8_t*) v4887;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4889 = *(const int8_t *)(v4888);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4890 = __riscv_vwmacc_vx_i16m1(v4851, v4886, v4865, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4891 = __riscv_vwmacc_vx_i16m1(v4852, v4889, v4867, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4892 = v32 + 827;
        const int8_t* v4893 = (const int8_t*) v4892;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4894 = *(const int8_t *)(v4893);
        const uint8_t* v4895 = v32 + 955;
        const int8_t* v4896 = (const int8_t*) v4895;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4897 = *(const int8_t *)(v4896);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4898 = __riscv_vwmacc_vx_i16m1(v4859, v4894, v4865, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4899 = __riscv_vwmacc_vx_i16m1(v4860, v4897, v4867, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4900 = v30 + 1968;
        const uint8_t* v4901 = (const uint8_t*) v4900;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4902 = __riscv_vle8_v_u8mf2(v4901, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4903 = __riscv_vand_vx_u8mf2(v4902, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4904 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4903);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4905 = __riscv_vsrl_vx_u8mf2(v4902, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4906 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4905);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4907 = v32 + 828;
        const int8_t* v4908 = (const int8_t*) v4907;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4909 = *(const int8_t *)(v4908);
        const uint8_t* v4910 = v32 + 956;
        const int8_t* v4911 = (const int8_t*) v4910;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4912 = *(const int8_t *)(v4911);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4913 = __riscv_vwmacc_vx_i16m1(v4874, v4909, v4904, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4914 = __riscv_vwmacc_vx_i16m1(v4875, v4912, v4906, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4915 = v32 + 829;
        const int8_t* v4916 = (const int8_t*) v4915;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4917 = *(const int8_t *)(v4916);
        const uint8_t* v4918 = v32 + 957;
        const int8_t* v4919 = (const int8_t*) v4918;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4920 = *(const int8_t *)(v4919);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4921 = __riscv_vwmacc_vx_i16m1(v4882, v4917, v4904, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4922 = __riscv_vwmacc_vx_i16m1(v4883, v4920, v4906, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4923 = v32 + 830;
        const int8_t* v4924 = (const int8_t*) v4923;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4925 = *(const int8_t *)(v4924);
        const uint8_t* v4926 = v32 + 958;
        const int8_t* v4927 = (const int8_t*) v4926;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4928 = *(const int8_t *)(v4927);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4929 = __riscv_vwmacc_vx_i16m1(v4890, v4925, v4904, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4930 = __riscv_vwmacc_vx_i16m1(v4891, v4928, v4906, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4931 = v32 + 831;
        const int8_t* v4932 = (const int8_t*) v4931;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4933 = *(const int8_t *)(v4932);
        const uint8_t* v4934 = v32 + 959;
        const int8_t* v4935 = (const int8_t*) v4934;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4936 = *(const int8_t *)(v4935);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4937 = __riscv_vwmacc_vx_i16m1(v4898, v4933, v4904, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4938 = __riscv_vwmacc_vx_i16m1(v4899, v4936, v4906, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4939 = v30 + 1984;
        const uint8_t* v4940 = (const uint8_t*) v4939;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4941 = __riscv_vle8_v_u8mf2(v4940, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4942 = __riscv_vand_vx_u8mf2(v4941, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4943 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4942);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4944 = __riscv_vsrl_vx_u8mf2(v4941, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4945 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4944);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4946 = v32 + 832;
        const int8_t* v4947 = (const int8_t*) v4946;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4948 = *(const int8_t *)(v4947);
        const uint8_t* v4949 = v32 + 960;
        const int8_t* v4950 = (const int8_t*) v4949;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4951 = *(const int8_t *)(v4950);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4952 = __riscv_vwmacc_vx_i16m1(v4913, v4948, v4943, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4953 = __riscv_vwmacc_vx_i16m1(v4914, v4951, v4945, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4954 = v32 + 833;
        const int8_t* v4955 = (const int8_t*) v4954;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4956 = *(const int8_t *)(v4955);
        const uint8_t* v4957 = v32 + 961;
        const int8_t* v4958 = (const int8_t*) v4957;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4959 = *(const int8_t *)(v4958);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4960 = __riscv_vwmacc_vx_i16m1(v4921, v4956, v4943, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4961 = __riscv_vwmacc_vx_i16m1(v4922, v4959, v4945, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4962 = v32 + 834;
        const int8_t* v4963 = (const int8_t*) v4962;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4964 = *(const int8_t *)(v4963);
        const uint8_t* v4965 = v32 + 962;
        const int8_t* v4966 = (const int8_t*) v4965;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4967 = *(const int8_t *)(v4966);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4968 = __riscv_vwmacc_vx_i16m1(v4929, v4964, v4943, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4969 = __riscv_vwmacc_vx_i16m1(v4930, v4967, v4945, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4970 = v32 + 835;
        const int8_t* v4971 = (const int8_t*) v4970;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4972 = *(const int8_t *)(v4971);
        const uint8_t* v4973 = v32 + 963;
        const int8_t* v4974 = (const int8_t*) v4973;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4975 = *(const int8_t *)(v4974);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4976 = __riscv_vwmacc_vx_i16m1(v4937, v4972, v4943, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4977 = __riscv_vwmacc_vx_i16m1(v4938, v4975, v4945, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v4978 = v30 + 2000;
        const uint8_t* v4979 = (const uint8_t*) v4978;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v4980 = __riscv_vle8_v_u8mf2(v4979, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v4981 = __riscv_vand_vx_u8mf2(v4980, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4982 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4981);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v4983 = __riscv_vsrl_vx_u8mf2(v4980, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v4984 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4983);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4985 = v32 + 836;
        const int8_t* v4986 = (const int8_t*) v4985;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4987 = *(const int8_t *)(v4986);
        const uint8_t* v4988 = v32 + 964;
        const int8_t* v4989 = (const int8_t*) v4988;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4990 = *(const int8_t *)(v4989);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4991 = __riscv_vwmacc_vx_i16m1(v4952, v4987, v4982, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4992 = __riscv_vwmacc_vx_i16m1(v4953, v4990, v4984, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v4993 = v32 + 837;
        const int8_t* v4994 = (const int8_t*) v4993;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4995 = *(const int8_t *)(v4994);
        const uint8_t* v4996 = v32 + 965;
        const int8_t* v4997 = (const int8_t*) v4996;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v4998 = *(const int8_t *)(v4997);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v4999 = __riscv_vwmacc_vx_i16m1(v4960, v4995, v4982, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5000 = __riscv_vwmacc_vx_i16m1(v4961, v4998, v4984, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5001 = v32 + 838;
        const int8_t* v5002 = (const int8_t*) v5001;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5003 = *(const int8_t *)(v5002);
        const uint8_t* v5004 = v32 + 966;
        const int8_t* v5005 = (const int8_t*) v5004;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5006 = *(const int8_t *)(v5005);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5007 = __riscv_vwmacc_vx_i16m1(v4968, v5003, v4982, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5008 = __riscv_vwmacc_vx_i16m1(v4969, v5006, v4984, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5009 = v32 + 839;
        const int8_t* v5010 = (const int8_t*) v5009;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5011 = *(const int8_t *)(v5010);
        const uint8_t* v5012 = v32 + 967;
        const int8_t* v5013 = (const int8_t*) v5012;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5014 = *(const int8_t *)(v5013);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5015 = __riscv_vwmacc_vx_i16m1(v4976, v5011, v4982, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5016 = __riscv_vwmacc_vx_i16m1(v4977, v5014, v4984, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5017 = v30 + 2016;
        const uint8_t* v5018 = (const uint8_t*) v5017;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5019 = __riscv_vle8_v_u8mf2(v5018, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5020 = __riscv_vand_vx_u8mf2(v5019, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5021 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5020);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5022 = __riscv_vsrl_vx_u8mf2(v5019, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5023 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5022);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5024 = v32 + 840;
        const int8_t* v5025 = (const int8_t*) v5024;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5026 = *(const int8_t *)(v5025);
        const uint8_t* v5027 = v32 + 968;
        const int8_t* v5028 = (const int8_t*) v5027;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5029 = *(const int8_t *)(v5028);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5030 = __riscv_vwmacc_vx_i16m1(v4991, v5026, v5021, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5031 = __riscv_vwmacc_vx_i16m1(v4992, v5029, v5023, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5032 = v32 + 841;
        const int8_t* v5033 = (const int8_t*) v5032;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5034 = *(const int8_t *)(v5033);
        const uint8_t* v5035 = v32 + 969;
        const int8_t* v5036 = (const int8_t*) v5035;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5037 = *(const int8_t *)(v5036);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5038 = __riscv_vwmacc_vx_i16m1(v4999, v5034, v5021, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5039 = __riscv_vwmacc_vx_i16m1(v5000, v5037, v5023, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5040 = v32 + 842;
        const int8_t* v5041 = (const int8_t*) v5040;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5042 = *(const int8_t *)(v5041);
        const uint8_t* v5043 = v32 + 970;
        const int8_t* v5044 = (const int8_t*) v5043;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5045 = *(const int8_t *)(v5044);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5046 = __riscv_vwmacc_vx_i16m1(v5007, v5042, v5021, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5047 = __riscv_vwmacc_vx_i16m1(v5008, v5045, v5023, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5048 = v32 + 843;
        const int8_t* v5049 = (const int8_t*) v5048;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5050 = *(const int8_t *)(v5049);
        const uint8_t* v5051 = v32 + 971;
        const int8_t* v5052 = (const int8_t*) v5051;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5053 = *(const int8_t *)(v5052);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5054 = __riscv_vwmacc_vx_i16m1(v5015, v5050, v5021, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5055 = __riscv_vwmacc_vx_i16m1(v5016, v5053, v5023, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5056 = v30 + 2032;
        const uint8_t* v5057 = (const uint8_t*) v5056;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5058 = __riscv_vle8_v_u8mf2(v5057, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5059 = __riscv_vand_vx_u8mf2(v5058, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5060 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5059);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5061 = __riscv_vsrl_vx_u8mf2(v5058, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5062 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5061);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5063 = v32 + 844;
        const int8_t* v5064 = (const int8_t*) v5063;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5065 = *(const int8_t *)(v5064);
        const uint8_t* v5066 = v32 + 972;
        const int8_t* v5067 = (const int8_t*) v5066;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5068 = *(const int8_t *)(v5067);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5069 = __riscv_vwmacc_vx_i16m1(v5030, v5065, v5060, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5070 = __riscv_vwmacc_vx_i16m1(v5031, v5068, v5062, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5071 = v32 + 845;
        const int8_t* v5072 = (const int8_t*) v5071;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5073 = *(const int8_t *)(v5072);
        const uint8_t* v5074 = v32 + 973;
        const int8_t* v5075 = (const int8_t*) v5074;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5076 = *(const int8_t *)(v5075);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5077 = __riscv_vwmacc_vx_i16m1(v5038, v5073, v5060, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5078 = __riscv_vwmacc_vx_i16m1(v5039, v5076, v5062, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5079 = v32 + 846;
        const int8_t* v5080 = (const int8_t*) v5079;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5081 = *(const int8_t *)(v5080);
        const uint8_t* v5082 = v32 + 974;
        const int8_t* v5083 = (const int8_t*) v5082;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5084 = *(const int8_t *)(v5083);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5085 = __riscv_vwmacc_vx_i16m1(v5046, v5081, v5060, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5086 = __riscv_vwmacc_vx_i16m1(v5047, v5084, v5062, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5087 = v32 + 847;
        const int8_t* v5088 = (const int8_t*) v5087;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5089 = *(const int8_t *)(v5088);
        const uint8_t* v5090 = v32 + 975;
        const int8_t* v5091 = (const int8_t*) v5090;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5092 = *(const int8_t *)(v5091);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5093 = __riscv_vwmacc_vx_i16m1(v5054, v5089, v5060, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5094 = __riscv_vwmacc_vx_i16m1(v5055, v5092, v5062, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v5095 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v5096 = __riscv_vle16_v_i16m1(v5095, 16);
        int16_t* v5097 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v5098 = __riscv_vle16_v_i16m1(v5097, 16);
        vint32m2_t v5099 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5100 = __riscv_vwmacc_vv_i32m2(v5099, v5096, v5069, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5101 = __riscv_vwmacc_vv_i32m2(v5100, v5098, v5070, 16);
        v47 = v5101;
        vint32m2_t v5102 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5103 = __riscv_vwmacc_vv_i32m2(v5102, v5096, v5077, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5104 = __riscv_vwmacc_vv_i32m2(v5103, v5098, v5078, 16);
        v49 = v5104;
        vint32m2_t v5105 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5106 = __riscv_vwmacc_vv_i32m2(v5105, v5096, v5085, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5107 = __riscv_vwmacc_vv_i32m2(v5106, v5098, v5086, 16);
        v51 = v5107;
        vint32m2_t v5108 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5109 = __riscv_vwmacc_vv_i32m2(v5108, v5096, v5093, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5110 = __riscv_vwmacc_vv_i32m2(v5109, v5098, v5094, 16);
        v53 = v5110;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v5111 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v5112 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v5113 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v5114 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v5115 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v5116 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v5117 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v5118 = __riscv_vmv_v_x_i16m1(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5119 = v30 + 2048;
        const uint8_t* v5120 = (const uint8_t*) v5119;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5121 = __riscv_vle8_v_u8mf2(v5120, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5122 = __riscv_vand_vx_u8mf2(v5121, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5123 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5122);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5124 = __riscv_vsrl_vx_u8mf2(v5121, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5125 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5124);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5126 = v32 + 848;
        const int8_t* v5127 = (const int8_t*) v5126;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5128 = *(const int8_t *)(v5127);
        const uint8_t* v5129 = v32 + 976;
        const int8_t* v5130 = (const int8_t*) v5129;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5131 = *(const int8_t *)(v5130);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5132 = __riscv_vwmacc_vx_i16m1(v5111, v5128, v5123, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5133 = __riscv_vwmacc_vx_i16m1(v5112, v5131, v5125, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5134 = v32 + 849;
        const int8_t* v5135 = (const int8_t*) v5134;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5136 = *(const int8_t *)(v5135);
        const uint8_t* v5137 = v32 + 977;
        const int8_t* v5138 = (const int8_t*) v5137;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5139 = *(const int8_t *)(v5138);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5140 = __riscv_vwmacc_vx_i16m1(v5113, v5136, v5123, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5141 = __riscv_vwmacc_vx_i16m1(v5114, v5139, v5125, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5142 = v32 + 850;
        const int8_t* v5143 = (const int8_t*) v5142;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5144 = *(const int8_t *)(v5143);
        const uint8_t* v5145 = v32 + 978;
        const int8_t* v5146 = (const int8_t*) v5145;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5147 = *(const int8_t *)(v5146);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5148 = __riscv_vwmacc_vx_i16m1(v5115, v5144, v5123, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5149 = __riscv_vwmacc_vx_i16m1(v5116, v5147, v5125, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5150 = v32 + 851;
        const int8_t* v5151 = (const int8_t*) v5150;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5152 = *(const int8_t *)(v5151);
        const uint8_t* v5153 = v32 + 979;
        const int8_t* v5154 = (const int8_t*) v5153;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5155 = *(const int8_t *)(v5154);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5156 = __riscv_vwmacc_vx_i16m1(v5117, v5152, v5123, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5157 = __riscv_vwmacc_vx_i16m1(v5118, v5155, v5125, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5158 = v30 + 2064;
        const uint8_t* v5159 = (const uint8_t*) v5158;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5160 = __riscv_vle8_v_u8mf2(v5159, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5161 = __riscv_vand_vx_u8mf2(v5160, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5162 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5161);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5163 = __riscv_vsrl_vx_u8mf2(v5160, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5164 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5163);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5165 = v32 + 852;
        const int8_t* v5166 = (const int8_t*) v5165;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5167 = *(const int8_t *)(v5166);
        const uint8_t* v5168 = v32 + 980;
        const int8_t* v5169 = (const int8_t*) v5168;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5170 = *(const int8_t *)(v5169);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5171 = __riscv_vwmacc_vx_i16m1(v5132, v5167, v5162, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5172 = __riscv_vwmacc_vx_i16m1(v5133, v5170, v5164, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5173 = v32 + 853;
        const int8_t* v5174 = (const int8_t*) v5173;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5175 = *(const int8_t *)(v5174);
        const uint8_t* v5176 = v32 + 981;
        const int8_t* v5177 = (const int8_t*) v5176;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5178 = *(const int8_t *)(v5177);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5179 = __riscv_vwmacc_vx_i16m1(v5140, v5175, v5162, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5180 = __riscv_vwmacc_vx_i16m1(v5141, v5178, v5164, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5181 = v32 + 854;
        const int8_t* v5182 = (const int8_t*) v5181;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5183 = *(const int8_t *)(v5182);
        const uint8_t* v5184 = v32 + 982;
        const int8_t* v5185 = (const int8_t*) v5184;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5186 = *(const int8_t *)(v5185);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5187 = __riscv_vwmacc_vx_i16m1(v5148, v5183, v5162, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5188 = __riscv_vwmacc_vx_i16m1(v5149, v5186, v5164, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5189 = v32 + 855;
        const int8_t* v5190 = (const int8_t*) v5189;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5191 = *(const int8_t *)(v5190);
        const uint8_t* v5192 = v32 + 983;
        const int8_t* v5193 = (const int8_t*) v5192;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5194 = *(const int8_t *)(v5193);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5195 = __riscv_vwmacc_vx_i16m1(v5156, v5191, v5162, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5196 = __riscv_vwmacc_vx_i16m1(v5157, v5194, v5164, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5197 = v30 + 2080;
        const uint8_t* v5198 = (const uint8_t*) v5197;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5199 = __riscv_vle8_v_u8mf2(v5198, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5200 = __riscv_vand_vx_u8mf2(v5199, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5201 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5200);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5202 = __riscv_vsrl_vx_u8mf2(v5199, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5203 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5202);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5204 = v32 + 856;
        const int8_t* v5205 = (const int8_t*) v5204;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5206 = *(const int8_t *)(v5205);
        const uint8_t* v5207 = v32 + 984;
        const int8_t* v5208 = (const int8_t*) v5207;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5209 = *(const int8_t *)(v5208);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5210 = __riscv_vwmacc_vx_i16m1(v5171, v5206, v5201, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5211 = __riscv_vwmacc_vx_i16m1(v5172, v5209, v5203, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5212 = v32 + 857;
        const int8_t* v5213 = (const int8_t*) v5212;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5214 = *(const int8_t *)(v5213);
        const uint8_t* v5215 = v32 + 985;
        const int8_t* v5216 = (const int8_t*) v5215;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5217 = *(const int8_t *)(v5216);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5218 = __riscv_vwmacc_vx_i16m1(v5179, v5214, v5201, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5219 = __riscv_vwmacc_vx_i16m1(v5180, v5217, v5203, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5220 = v32 + 858;
        const int8_t* v5221 = (const int8_t*) v5220;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5222 = *(const int8_t *)(v5221);
        const uint8_t* v5223 = v32 + 986;
        const int8_t* v5224 = (const int8_t*) v5223;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5225 = *(const int8_t *)(v5224);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5226 = __riscv_vwmacc_vx_i16m1(v5187, v5222, v5201, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5227 = __riscv_vwmacc_vx_i16m1(v5188, v5225, v5203, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5228 = v32 + 859;
        const int8_t* v5229 = (const int8_t*) v5228;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5230 = *(const int8_t *)(v5229);
        const uint8_t* v5231 = v32 + 987;
        const int8_t* v5232 = (const int8_t*) v5231;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5233 = *(const int8_t *)(v5232);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5234 = __riscv_vwmacc_vx_i16m1(v5195, v5230, v5201, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5235 = __riscv_vwmacc_vx_i16m1(v5196, v5233, v5203, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5236 = v30 + 2096;
        const uint8_t* v5237 = (const uint8_t*) v5236;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5238 = __riscv_vle8_v_u8mf2(v5237, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5239 = __riscv_vand_vx_u8mf2(v5238, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5240 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5239);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5241 = __riscv_vsrl_vx_u8mf2(v5238, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5242 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5241);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5243 = v32 + 860;
        const int8_t* v5244 = (const int8_t*) v5243;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5245 = *(const int8_t *)(v5244);
        const uint8_t* v5246 = v32 + 988;
        const int8_t* v5247 = (const int8_t*) v5246;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5248 = *(const int8_t *)(v5247);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5249 = __riscv_vwmacc_vx_i16m1(v5210, v5245, v5240, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5250 = __riscv_vwmacc_vx_i16m1(v5211, v5248, v5242, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5251 = v32 + 861;
        const int8_t* v5252 = (const int8_t*) v5251;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5253 = *(const int8_t *)(v5252);
        const uint8_t* v5254 = v32 + 989;
        const int8_t* v5255 = (const int8_t*) v5254;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5256 = *(const int8_t *)(v5255);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5257 = __riscv_vwmacc_vx_i16m1(v5218, v5253, v5240, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5258 = __riscv_vwmacc_vx_i16m1(v5219, v5256, v5242, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5259 = v32 + 862;
        const int8_t* v5260 = (const int8_t*) v5259;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5261 = *(const int8_t *)(v5260);
        const uint8_t* v5262 = v32 + 990;
        const int8_t* v5263 = (const int8_t*) v5262;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5264 = *(const int8_t *)(v5263);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5265 = __riscv_vwmacc_vx_i16m1(v5226, v5261, v5240, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5266 = __riscv_vwmacc_vx_i16m1(v5227, v5264, v5242, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5267 = v32 + 863;
        const int8_t* v5268 = (const int8_t*) v5267;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5269 = *(const int8_t *)(v5268);
        const uint8_t* v5270 = v32 + 991;
        const int8_t* v5271 = (const int8_t*) v5270;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5272 = *(const int8_t *)(v5271);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5273 = __riscv_vwmacc_vx_i16m1(v5234, v5269, v5240, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5274 = __riscv_vwmacc_vx_i16m1(v5235, v5272, v5242, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5275 = v30 + 2112;
        const uint8_t* v5276 = (const uint8_t*) v5275;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5277 = __riscv_vle8_v_u8mf2(v5276, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5278 = __riscv_vand_vx_u8mf2(v5277, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5279 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5278);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5280 = __riscv_vsrl_vx_u8mf2(v5277, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5281 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5280);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5282 = v32 + 864;
        const int8_t* v5283 = (const int8_t*) v5282;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5284 = *(const int8_t *)(v5283);
        const uint8_t* v5285 = v32 + 992;
        const int8_t* v5286 = (const int8_t*) v5285;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5287 = *(const int8_t *)(v5286);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5288 = __riscv_vwmacc_vx_i16m1(v5249, v5284, v5279, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5289 = __riscv_vwmacc_vx_i16m1(v5250, v5287, v5281, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5290 = v32 + 865;
        const int8_t* v5291 = (const int8_t*) v5290;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5292 = *(const int8_t *)(v5291);
        const uint8_t* v5293 = v32 + 993;
        const int8_t* v5294 = (const int8_t*) v5293;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5295 = *(const int8_t *)(v5294);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5296 = __riscv_vwmacc_vx_i16m1(v5257, v5292, v5279, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5297 = __riscv_vwmacc_vx_i16m1(v5258, v5295, v5281, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5298 = v32 + 866;
        const int8_t* v5299 = (const int8_t*) v5298;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5300 = *(const int8_t *)(v5299);
        const uint8_t* v5301 = v32 + 994;
        const int8_t* v5302 = (const int8_t*) v5301;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5303 = *(const int8_t *)(v5302);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5304 = __riscv_vwmacc_vx_i16m1(v5265, v5300, v5279, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5305 = __riscv_vwmacc_vx_i16m1(v5266, v5303, v5281, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5306 = v32 + 867;
        const int8_t* v5307 = (const int8_t*) v5306;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5308 = *(const int8_t *)(v5307);
        const uint8_t* v5309 = v32 + 995;
        const int8_t* v5310 = (const int8_t*) v5309;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5311 = *(const int8_t *)(v5310);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5312 = __riscv_vwmacc_vx_i16m1(v5273, v5308, v5279, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5313 = __riscv_vwmacc_vx_i16m1(v5274, v5311, v5281, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5314 = v30 + 2128;
        const uint8_t* v5315 = (const uint8_t*) v5314;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5316 = __riscv_vle8_v_u8mf2(v5315, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5317 = __riscv_vand_vx_u8mf2(v5316, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5318 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5317);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5319 = __riscv_vsrl_vx_u8mf2(v5316, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5320 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5319);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5321 = v32 + 868;
        const int8_t* v5322 = (const int8_t*) v5321;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5323 = *(const int8_t *)(v5322);
        const uint8_t* v5324 = v32 + 996;
        const int8_t* v5325 = (const int8_t*) v5324;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5326 = *(const int8_t *)(v5325);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5327 = __riscv_vwmacc_vx_i16m1(v5288, v5323, v5318, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5328 = __riscv_vwmacc_vx_i16m1(v5289, v5326, v5320, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5329 = v32 + 869;
        const int8_t* v5330 = (const int8_t*) v5329;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5331 = *(const int8_t *)(v5330);
        const uint8_t* v5332 = v32 + 997;
        const int8_t* v5333 = (const int8_t*) v5332;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5334 = *(const int8_t *)(v5333);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5335 = __riscv_vwmacc_vx_i16m1(v5296, v5331, v5318, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5336 = __riscv_vwmacc_vx_i16m1(v5297, v5334, v5320, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5337 = v32 + 870;
        const int8_t* v5338 = (const int8_t*) v5337;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5339 = *(const int8_t *)(v5338);
        const uint8_t* v5340 = v32 + 998;
        const int8_t* v5341 = (const int8_t*) v5340;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5342 = *(const int8_t *)(v5341);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5343 = __riscv_vwmacc_vx_i16m1(v5304, v5339, v5318, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5344 = __riscv_vwmacc_vx_i16m1(v5305, v5342, v5320, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5345 = v32 + 871;
        const int8_t* v5346 = (const int8_t*) v5345;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5347 = *(const int8_t *)(v5346);
        const uint8_t* v5348 = v32 + 999;
        const int8_t* v5349 = (const int8_t*) v5348;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5350 = *(const int8_t *)(v5349);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5351 = __riscv_vwmacc_vx_i16m1(v5312, v5347, v5318, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5352 = __riscv_vwmacc_vx_i16m1(v5313, v5350, v5320, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5353 = v30 + 2144;
        const uint8_t* v5354 = (const uint8_t*) v5353;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5355 = __riscv_vle8_v_u8mf2(v5354, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5356 = __riscv_vand_vx_u8mf2(v5355, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5357 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5356);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5358 = __riscv_vsrl_vx_u8mf2(v5355, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5359 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5358);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5360 = v32 + 872;
        const int8_t* v5361 = (const int8_t*) v5360;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5362 = *(const int8_t *)(v5361);
        const uint8_t* v5363 = v32 + 1000;
        const int8_t* v5364 = (const int8_t*) v5363;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5365 = *(const int8_t *)(v5364);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5366 = __riscv_vwmacc_vx_i16m1(v5327, v5362, v5357, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5367 = __riscv_vwmacc_vx_i16m1(v5328, v5365, v5359, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5368 = v32 + 873;
        const int8_t* v5369 = (const int8_t*) v5368;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5370 = *(const int8_t *)(v5369);
        const uint8_t* v5371 = v32 + 1001;
        const int8_t* v5372 = (const int8_t*) v5371;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5373 = *(const int8_t *)(v5372);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5374 = __riscv_vwmacc_vx_i16m1(v5335, v5370, v5357, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5375 = __riscv_vwmacc_vx_i16m1(v5336, v5373, v5359, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5376 = v32 + 874;
        const int8_t* v5377 = (const int8_t*) v5376;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5378 = *(const int8_t *)(v5377);
        const uint8_t* v5379 = v32 + 1002;
        const int8_t* v5380 = (const int8_t*) v5379;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5381 = *(const int8_t *)(v5380);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5382 = __riscv_vwmacc_vx_i16m1(v5343, v5378, v5357, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5383 = __riscv_vwmacc_vx_i16m1(v5344, v5381, v5359, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5384 = v32 + 875;
        const int8_t* v5385 = (const int8_t*) v5384;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5386 = *(const int8_t *)(v5385);
        const uint8_t* v5387 = v32 + 1003;
        const int8_t* v5388 = (const int8_t*) v5387;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5389 = *(const int8_t *)(v5388);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5390 = __riscv_vwmacc_vx_i16m1(v5351, v5386, v5357, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5391 = __riscv_vwmacc_vx_i16m1(v5352, v5389, v5359, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5392 = v30 + 2160;
        const uint8_t* v5393 = (const uint8_t*) v5392;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5394 = __riscv_vle8_v_u8mf2(v5393, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5395 = __riscv_vand_vx_u8mf2(v5394, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5396 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5395);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5397 = __riscv_vsrl_vx_u8mf2(v5394, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5398 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5397);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5399 = v32 + 876;
        const int8_t* v5400 = (const int8_t*) v5399;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5401 = *(const int8_t *)(v5400);
        const uint8_t* v5402 = v32 + 1004;
        const int8_t* v5403 = (const int8_t*) v5402;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5404 = *(const int8_t *)(v5403);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5405 = __riscv_vwmacc_vx_i16m1(v5366, v5401, v5396, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5406 = __riscv_vwmacc_vx_i16m1(v5367, v5404, v5398, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5407 = v32 + 877;
        const int8_t* v5408 = (const int8_t*) v5407;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5409 = *(const int8_t *)(v5408);
        const uint8_t* v5410 = v32 + 1005;
        const int8_t* v5411 = (const int8_t*) v5410;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5412 = *(const int8_t *)(v5411);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5413 = __riscv_vwmacc_vx_i16m1(v5374, v5409, v5396, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5414 = __riscv_vwmacc_vx_i16m1(v5375, v5412, v5398, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5415 = v32 + 878;
        const int8_t* v5416 = (const int8_t*) v5415;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5417 = *(const int8_t *)(v5416);
        const uint8_t* v5418 = v32 + 1006;
        const int8_t* v5419 = (const int8_t*) v5418;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5420 = *(const int8_t *)(v5419);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5421 = __riscv_vwmacc_vx_i16m1(v5382, v5417, v5396, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5422 = __riscv_vwmacc_vx_i16m1(v5383, v5420, v5398, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5423 = v32 + 879;
        const int8_t* v5424 = (const int8_t*) v5423;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5425 = *(const int8_t *)(v5424);
        const uint8_t* v5426 = v32 + 1007;
        const int8_t* v5427 = (const int8_t*) v5426;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5428 = *(const int8_t *)(v5427);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5429 = __riscv_vwmacc_vx_i16m1(v5390, v5425, v5396, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5430 = __riscv_vwmacc_vx_i16m1(v5391, v5428, v5398, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5431 = v30 + 2176;
        const uint8_t* v5432 = (const uint8_t*) v5431;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5433 = __riscv_vle8_v_u8mf2(v5432, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5434 = __riscv_vand_vx_u8mf2(v5433, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5435 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5434);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5436 = __riscv_vsrl_vx_u8mf2(v5433, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5437 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5436);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5438 = v32 + 880;
        const int8_t* v5439 = (const int8_t*) v5438;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5440 = *(const int8_t *)(v5439);
        const uint8_t* v5441 = v32 + 1008;
        const int8_t* v5442 = (const int8_t*) v5441;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5443 = *(const int8_t *)(v5442);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5444 = __riscv_vwmacc_vx_i16m1(v5405, v5440, v5435, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5445 = __riscv_vwmacc_vx_i16m1(v5406, v5443, v5437, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5446 = v32 + 881;
        const int8_t* v5447 = (const int8_t*) v5446;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5448 = *(const int8_t *)(v5447);
        const uint8_t* v5449 = v32 + 1009;
        const int8_t* v5450 = (const int8_t*) v5449;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5451 = *(const int8_t *)(v5450);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5452 = __riscv_vwmacc_vx_i16m1(v5413, v5448, v5435, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5453 = __riscv_vwmacc_vx_i16m1(v5414, v5451, v5437, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5454 = v32 + 882;
        const int8_t* v5455 = (const int8_t*) v5454;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5456 = *(const int8_t *)(v5455);
        const uint8_t* v5457 = v32 + 1010;
        const int8_t* v5458 = (const int8_t*) v5457;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5459 = *(const int8_t *)(v5458);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5460 = __riscv_vwmacc_vx_i16m1(v5421, v5456, v5435, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5461 = __riscv_vwmacc_vx_i16m1(v5422, v5459, v5437, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5462 = v32 + 883;
        const int8_t* v5463 = (const int8_t*) v5462;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5464 = *(const int8_t *)(v5463);
        const uint8_t* v5465 = v32 + 1011;
        const int8_t* v5466 = (const int8_t*) v5465;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5467 = *(const int8_t *)(v5466);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5468 = __riscv_vwmacc_vx_i16m1(v5429, v5464, v5435, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5469 = __riscv_vwmacc_vx_i16m1(v5430, v5467, v5437, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5470 = v30 + 2192;
        const uint8_t* v5471 = (const uint8_t*) v5470;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5472 = __riscv_vle8_v_u8mf2(v5471, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5473 = __riscv_vand_vx_u8mf2(v5472, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5474 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5473);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5475 = __riscv_vsrl_vx_u8mf2(v5472, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5476 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5475);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5477 = v32 + 884;
        const int8_t* v5478 = (const int8_t*) v5477;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5479 = *(const int8_t *)(v5478);
        const uint8_t* v5480 = v32 + 1012;
        const int8_t* v5481 = (const int8_t*) v5480;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5482 = *(const int8_t *)(v5481);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5483 = __riscv_vwmacc_vx_i16m1(v5444, v5479, v5474, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5484 = __riscv_vwmacc_vx_i16m1(v5445, v5482, v5476, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5485 = v32 + 885;
        const int8_t* v5486 = (const int8_t*) v5485;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5487 = *(const int8_t *)(v5486);
        const uint8_t* v5488 = v32 + 1013;
        const int8_t* v5489 = (const int8_t*) v5488;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5490 = *(const int8_t *)(v5489);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5491 = __riscv_vwmacc_vx_i16m1(v5452, v5487, v5474, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5492 = __riscv_vwmacc_vx_i16m1(v5453, v5490, v5476, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5493 = v32 + 886;
        const int8_t* v5494 = (const int8_t*) v5493;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5495 = *(const int8_t *)(v5494);
        const uint8_t* v5496 = v32 + 1014;
        const int8_t* v5497 = (const int8_t*) v5496;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5498 = *(const int8_t *)(v5497);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5499 = __riscv_vwmacc_vx_i16m1(v5460, v5495, v5474, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5500 = __riscv_vwmacc_vx_i16m1(v5461, v5498, v5476, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5501 = v32 + 887;
        const int8_t* v5502 = (const int8_t*) v5501;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5503 = *(const int8_t *)(v5502);
        const uint8_t* v5504 = v32 + 1015;
        const int8_t* v5505 = (const int8_t*) v5504;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5506 = *(const int8_t *)(v5505);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5507 = __riscv_vwmacc_vx_i16m1(v5468, v5503, v5474, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5508 = __riscv_vwmacc_vx_i16m1(v5469, v5506, v5476, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5509 = v30 + 2208;
        const uint8_t* v5510 = (const uint8_t*) v5509;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5511 = __riscv_vle8_v_u8mf2(v5510, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5512 = __riscv_vand_vx_u8mf2(v5511, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5513 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5512);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5514 = __riscv_vsrl_vx_u8mf2(v5511, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5515 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5514);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5516 = v32 + 888;
        const int8_t* v5517 = (const int8_t*) v5516;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5518 = *(const int8_t *)(v5517);
        const uint8_t* v5519 = v32 + 1016;
        const int8_t* v5520 = (const int8_t*) v5519;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5521 = *(const int8_t *)(v5520);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5522 = __riscv_vwmacc_vx_i16m1(v5483, v5518, v5513, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5523 = __riscv_vwmacc_vx_i16m1(v5484, v5521, v5515, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5524 = v32 + 889;
        const int8_t* v5525 = (const int8_t*) v5524;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5526 = *(const int8_t *)(v5525);
        const uint8_t* v5527 = v32 + 1017;
        const int8_t* v5528 = (const int8_t*) v5527;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5529 = *(const int8_t *)(v5528);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5530 = __riscv_vwmacc_vx_i16m1(v5491, v5526, v5513, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5531 = __riscv_vwmacc_vx_i16m1(v5492, v5529, v5515, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5532 = v32 + 890;
        const int8_t* v5533 = (const int8_t*) v5532;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5534 = *(const int8_t *)(v5533);
        const uint8_t* v5535 = v32 + 1018;
        const int8_t* v5536 = (const int8_t*) v5535;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5537 = *(const int8_t *)(v5536);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5538 = __riscv_vwmacc_vx_i16m1(v5499, v5534, v5513, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5539 = __riscv_vwmacc_vx_i16m1(v5500, v5537, v5515, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5540 = v32 + 891;
        const int8_t* v5541 = (const int8_t*) v5540;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5542 = *(const int8_t *)(v5541);
        const uint8_t* v5543 = v32 + 1019;
        const int8_t* v5544 = (const int8_t*) v5543;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5545 = *(const int8_t *)(v5544);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5546 = __riscv_vwmacc_vx_i16m1(v5507, v5542, v5513, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5547 = __riscv_vwmacc_vx_i16m1(v5508, v5545, v5515, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5548 = v30 + 2224;
        const uint8_t* v5549 = (const uint8_t*) v5548;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5550 = __riscv_vle8_v_u8mf2(v5549, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5551 = __riscv_vand_vx_u8mf2(v5550, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5552 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5551);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5553 = __riscv_vsrl_vx_u8mf2(v5550, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5554 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5553);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5555 = v32 + 892;
        const int8_t* v5556 = (const int8_t*) v5555;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5557 = *(const int8_t *)(v5556);
        const uint8_t* v5558 = v32 + 1020;
        const int8_t* v5559 = (const int8_t*) v5558;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5560 = *(const int8_t *)(v5559);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5561 = __riscv_vwmacc_vx_i16m1(v5522, v5557, v5552, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5562 = __riscv_vwmacc_vx_i16m1(v5523, v5560, v5554, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5563 = v32 + 893;
        const int8_t* v5564 = (const int8_t*) v5563;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5565 = *(const int8_t *)(v5564);
        const uint8_t* v5566 = v32 + 1021;
        const int8_t* v5567 = (const int8_t*) v5566;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5568 = *(const int8_t *)(v5567);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5569 = __riscv_vwmacc_vx_i16m1(v5530, v5565, v5552, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5570 = __riscv_vwmacc_vx_i16m1(v5531, v5568, v5554, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5571 = v32 + 894;
        const int8_t* v5572 = (const int8_t*) v5571;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5573 = *(const int8_t *)(v5572);
        const uint8_t* v5574 = v32 + 1022;
        const int8_t* v5575 = (const int8_t*) v5574;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5576 = *(const int8_t *)(v5575);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5577 = __riscv_vwmacc_vx_i16m1(v5538, v5573, v5552, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5578 = __riscv_vwmacc_vx_i16m1(v5539, v5576, v5554, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5579 = v32 + 895;
        const int8_t* v5580 = (const int8_t*) v5579;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5581 = *(const int8_t *)(v5580);
        const uint8_t* v5582 = v32 + 1023;
        const int8_t* v5583 = (const int8_t*) v5582;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5584 = *(const int8_t *)(v5583);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5585 = __riscv_vwmacc_vx_i16m1(v5546, v5581, v5552, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5586 = __riscv_vwmacc_vx_i16m1(v5547, v5584, v5554, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5587 = v30 + 2240;
        const uint8_t* v5588 = (const uint8_t*) v5587;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5589 = __riscv_vle8_v_u8mf2(v5588, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5590 = __riscv_vand_vx_u8mf2(v5589, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5591 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5590);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5592 = __riscv_vsrl_vx_u8mf2(v5589, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5593 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5592);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5594 = v32 + 896;
        const int8_t* v5595 = (const int8_t*) v5594;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5596 = *(const int8_t *)(v5595);
        const uint8_t* v5597 = v32 + 1024;
        const int8_t* v5598 = (const int8_t*) v5597;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5599 = *(const int8_t *)(v5598);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5600 = __riscv_vwmacc_vx_i16m1(v5561, v5596, v5591, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5601 = __riscv_vwmacc_vx_i16m1(v5562, v5599, v5593, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5602 = v32 + 897;
        const int8_t* v5603 = (const int8_t*) v5602;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5604 = *(const int8_t *)(v5603);
        const uint8_t* v5605 = v32 + 1025;
        const int8_t* v5606 = (const int8_t*) v5605;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5607 = *(const int8_t *)(v5606);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5608 = __riscv_vwmacc_vx_i16m1(v5569, v5604, v5591, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5609 = __riscv_vwmacc_vx_i16m1(v5570, v5607, v5593, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5610 = v32 + 898;
        const int8_t* v5611 = (const int8_t*) v5610;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5612 = *(const int8_t *)(v5611);
        const uint8_t* v5613 = v32 + 1026;
        const int8_t* v5614 = (const int8_t*) v5613;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5615 = *(const int8_t *)(v5614);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5616 = __riscv_vwmacc_vx_i16m1(v5577, v5612, v5591, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5617 = __riscv_vwmacc_vx_i16m1(v5578, v5615, v5593, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5618 = v32 + 899;
        const int8_t* v5619 = (const int8_t*) v5618;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5620 = *(const int8_t *)(v5619);
        const uint8_t* v5621 = v32 + 1027;
        const int8_t* v5622 = (const int8_t*) v5621;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5623 = *(const int8_t *)(v5622);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5624 = __riscv_vwmacc_vx_i16m1(v5585, v5620, v5591, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5625 = __riscv_vwmacc_vx_i16m1(v5586, v5623, v5593, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5626 = v30 + 2256;
        const uint8_t* v5627 = (const uint8_t*) v5626;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5628 = __riscv_vle8_v_u8mf2(v5627, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5629 = __riscv_vand_vx_u8mf2(v5628, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5630 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5629);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5631 = __riscv_vsrl_vx_u8mf2(v5628, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5632 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5631);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5633 = v32 + 900;
        const int8_t* v5634 = (const int8_t*) v5633;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5635 = *(const int8_t *)(v5634);
        const uint8_t* v5636 = v32 + 1028;
        const int8_t* v5637 = (const int8_t*) v5636;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5638 = *(const int8_t *)(v5637);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5639 = __riscv_vwmacc_vx_i16m1(v5600, v5635, v5630, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5640 = __riscv_vwmacc_vx_i16m1(v5601, v5638, v5632, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5641 = v32 + 901;
        const int8_t* v5642 = (const int8_t*) v5641;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5643 = *(const int8_t *)(v5642);
        const uint8_t* v5644 = v32 + 1029;
        const int8_t* v5645 = (const int8_t*) v5644;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5646 = *(const int8_t *)(v5645);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5647 = __riscv_vwmacc_vx_i16m1(v5608, v5643, v5630, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5648 = __riscv_vwmacc_vx_i16m1(v5609, v5646, v5632, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5649 = v32 + 902;
        const int8_t* v5650 = (const int8_t*) v5649;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5651 = *(const int8_t *)(v5650);
        const uint8_t* v5652 = v32 + 1030;
        const int8_t* v5653 = (const int8_t*) v5652;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5654 = *(const int8_t *)(v5653);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5655 = __riscv_vwmacc_vx_i16m1(v5616, v5651, v5630, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5656 = __riscv_vwmacc_vx_i16m1(v5617, v5654, v5632, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5657 = v32 + 903;
        const int8_t* v5658 = (const int8_t*) v5657;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5659 = *(const int8_t *)(v5658);
        const uint8_t* v5660 = v32 + 1031;
        const int8_t* v5661 = (const int8_t*) v5660;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5662 = *(const int8_t *)(v5661);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5663 = __riscv_vwmacc_vx_i16m1(v5624, v5659, v5630, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5664 = __riscv_vwmacc_vx_i16m1(v5625, v5662, v5632, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5665 = v30 + 2272;
        const uint8_t* v5666 = (const uint8_t*) v5665;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5667 = __riscv_vle8_v_u8mf2(v5666, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5668 = __riscv_vand_vx_u8mf2(v5667, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5669 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5668);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5670 = __riscv_vsrl_vx_u8mf2(v5667, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5671 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5670);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5672 = v32 + 904;
        const int8_t* v5673 = (const int8_t*) v5672;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5674 = *(const int8_t *)(v5673);
        const uint8_t* v5675 = v32 + 1032;
        const int8_t* v5676 = (const int8_t*) v5675;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5677 = *(const int8_t *)(v5676);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5678 = __riscv_vwmacc_vx_i16m1(v5639, v5674, v5669, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5679 = __riscv_vwmacc_vx_i16m1(v5640, v5677, v5671, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5680 = v32 + 905;
        const int8_t* v5681 = (const int8_t*) v5680;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5682 = *(const int8_t *)(v5681);
        const uint8_t* v5683 = v32 + 1033;
        const int8_t* v5684 = (const int8_t*) v5683;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5685 = *(const int8_t *)(v5684);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5686 = __riscv_vwmacc_vx_i16m1(v5647, v5682, v5669, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5687 = __riscv_vwmacc_vx_i16m1(v5648, v5685, v5671, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5688 = v32 + 906;
        const int8_t* v5689 = (const int8_t*) v5688;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5690 = *(const int8_t *)(v5689);
        const uint8_t* v5691 = v32 + 1034;
        const int8_t* v5692 = (const int8_t*) v5691;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5693 = *(const int8_t *)(v5692);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5694 = __riscv_vwmacc_vx_i16m1(v5655, v5690, v5669, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5695 = __riscv_vwmacc_vx_i16m1(v5656, v5693, v5671, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5696 = v32 + 907;
        const int8_t* v5697 = (const int8_t*) v5696;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5698 = *(const int8_t *)(v5697);
        const uint8_t* v5699 = v32 + 1035;
        const int8_t* v5700 = (const int8_t*) v5699;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5701 = *(const int8_t *)(v5700);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5702 = __riscv_vwmacc_vx_i16m1(v5663, v5698, v5669, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5703 = __riscv_vwmacc_vx_i16m1(v5664, v5701, v5671, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        const uint8_t* v5704 = v30 + 2288;
        const uint8_t* v5705 = (const uint8_t*) v5704;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v5706 = __riscv_vle8_v_u8mf2(v5705, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v5707 = __riscv_vand_vx_u8mf2(v5706, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5708 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5707);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v5709 = __riscv_vsrl_vx_u8mf2(v5706, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v5710 = __riscv_vreinterpret_v_u8mf2_i8mf2(v5709);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5711 = v32 + 908;
        const int8_t* v5712 = (const int8_t*) v5711;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5713 = *(const int8_t *)(v5712);
        const uint8_t* v5714 = v32 + 1036;
        const int8_t* v5715 = (const int8_t*) v5714;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5716 = *(const int8_t *)(v5715);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5717 = __riscv_vwmacc_vx_i16m1(v5678, v5713, v5708, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5718 = __riscv_vwmacc_vx_i16m1(v5679, v5716, v5710, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5719 = v32 + 909;
        const int8_t* v5720 = (const int8_t*) v5719;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5721 = *(const int8_t *)(v5720);
        const uint8_t* v5722 = v32 + 1037;
        const int8_t* v5723 = (const int8_t*) v5722;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5724 = *(const int8_t *)(v5723);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5725 = __riscv_vwmacc_vx_i16m1(v5686, v5721, v5708, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5726 = __riscv_vwmacc_vx_i16m1(v5687, v5724, v5710, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5727 = v32 + 910;
        const int8_t* v5728 = (const int8_t*) v5727;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5729 = *(const int8_t *)(v5728);
        const uint8_t* v5730 = v32 + 1038;
        const int8_t* v5731 = (const int8_t*) v5730;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5732 = *(const int8_t *)(v5731);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5733 = __riscv_vwmacc_vx_i16m1(v5694, v5729, v5708, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5734 = __riscv_vwmacc_vx_i16m1(v5695, v5732, v5710, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        const uint8_t* v5735 = v32 + 911;
        const int8_t* v5736 = (const int8_t*) v5735;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5737 = *(const int8_t *)(v5736);
        const uint8_t* v5738 = v32 + 1039;
        const int8_t* v5739 = (const int8_t*) v5738;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v5740 = *(const int8_t *)(v5739);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5741 = __riscv_vwmacc_vx_i16m1(v5702, v5737, v5708, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v5742 = __riscv_vwmacc_vx_i16m1(v5703, v5740, v5710, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v5743 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v5744 = __riscv_vle16_v_i16m1(v5743, 16);
        int16_t* v5745 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v5746 = __riscv_vle16_v_i16m1(v5745, 16);
        vint32m2_t v5747 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5748 = __riscv_vwmacc_vv_i32m2(v5747, v5744, v5717, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5749 = __riscv_vwmacc_vv_i32m2(v5748, v5746, v5718, 16);
        v47 = v5749;
        vint32m2_t v5750 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5751 = __riscv_vwmacc_vv_i32m2(v5750, v5744, v5725, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5752 = __riscv_vwmacc_vv_i32m2(v5751, v5746, v5726, 16);
        v49 = v5752;
        vint32m2_t v5753 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5754 = __riscv_vwmacc_vv_i32m2(v5753, v5744, v5733, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5755 = __riscv_vwmacc_vv_i32m2(v5754, v5746, v5734, 16);
        v51 = v5755;
        vint32m2_t v5756 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5757 = __riscv_vwmacc_vv_i32m2(v5756, v5744, v5741, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v5758 = __riscv_vwmacc_vv_i32m2(v5757, v5746, v5742, 16);
        v53 = v5758;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const uint8_t* v5759 = v30 + 32;
        const _Float16* v5760 = (const _Float16*) v5759;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v5761 = __riscv_vle16_v_f16m1(v5760, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v5762 = __riscv_vfwcvt_f_f_v_f32m2(v5761, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const _Float16* v5763 = (const _Float16*) v30;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v5764 = __riscv_vle16_v_f16m1(v5763, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v5765 = __riscv_vfwcvt_f_f_v_f32m2(v5764, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v5766 = __riscv_vfmul_vf_f32m2(v5765, v34, 16);
        vint32m2_t v5767 = v47;
        vfloat32m2_t v5768 = v20;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v5769 = __riscv_vfcvt_f_x_v_f32m2(v5767, 16);
        vfloat32m2_t v5770 = __riscv_vfmacc_vv_f32m2(v5768, v5769, v5766, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v5771 = __riscv_vfmul_vf_f32m2(v5762, v34, 16);
        int32_t* v5772 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v5773 = __riscv_vle32_v_i32m2(v5772, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v5774 = __riscv_vfcvt_f_x_v_f32m2(v5773, 16);
        vfloat32m2_t v5775 = __riscv_vfnmsac_vv_f32m2(v5770, v5771, v5774, 16);
        v20 = v5775;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v5776 = __riscv_vfmul_vf_f32m2(v5765, v37, 16);
        vint32m2_t v5777 = v49;
        vfloat32m2_t v5778 = v22;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v5779 = __riscv_vfcvt_f_x_v_f32m2(v5777, 16);
        vfloat32m2_t v5780 = __riscv_vfmacc_vv_f32m2(v5778, v5779, v5776, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v5781 = __riscv_vfmul_vf_f32m2(v5762, v37, 16);
        int32_t* v5782 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v5783 = __riscv_vle32_v_i32m2(v5782, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v5784 = __riscv_vfcvt_f_x_v_f32m2(v5783, 16);
        vfloat32m2_t v5785 = __riscv_vfnmsac_vv_f32m2(v5780, v5781, v5784, 16);
        v22 = v5785;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v5786 = __riscv_vfmul_vf_f32m2(v5765, v40, 16);
        vint32m2_t v5787 = v51;
        vfloat32m2_t v5788 = v24;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v5789 = __riscv_vfcvt_f_x_v_f32m2(v5787, 16);
        vfloat32m2_t v5790 = __riscv_vfmacc_vv_f32m2(v5788, v5789, v5786, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v5791 = __riscv_vfmul_vf_f32m2(v5762, v40, 16);
        int32_t* v5792 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v5793 = __riscv_vle32_v_i32m2(v5792, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v5794 = __riscv_vfcvt_f_x_v_f32m2(v5793, 16);
        vfloat32m2_t v5795 = __riscv_vfnmsac_vv_f32m2(v5790, v5791, v5794, 16);
        v24 = v5795;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v5796 = __riscv_vfmul_vf_f32m2(v5765, v43, 16);
        vint32m2_t v5797 = v53;
        vfloat32m2_t v5798 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v5799 = __riscv_vfcvt_f_x_v_f32m2(v5797, 16);
        vfloat32m2_t v5800 = __riscv_vfmacc_vv_f32m2(v5798, v5799, v5796, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v5801 = __riscv_vfmul_vf_f32m2(v5762, v43, 16);
        int32_t* v5802 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v5803 = __riscv_vle32_v_i32m2(v5802, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v5804 = __riscv_vfcvt_f_x_v_f32m2(v5803, 16);
        vfloat32m2_t v5805 = __riscv_vfnmsac_vv_f32m2(v5800, v5801, v5804, 16);
        v26 = v5805;
      }
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v5806 = v16 * 4;
      size_t v5807 = v5806 + 0;
      size_t v5808 = v5807 * v7;
      size_t v5809 = v12 * 16;
      size_t v5810 = v5808 + v5809;
      float* v5811 = v2 + v5810;
      vfloat32m2_t v5812 = v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v5811, v5812, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v5813 = v16 * 4;
      size_t v5814 = v5813 + 1;
      size_t v5815 = v5814 * v7;
      size_t v5816 = v12 * 16;
      size_t v5817 = v5815 + v5816;
      float* v5818 = v2 + v5817;
      vfloat32m2_t v5819 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v5818, v5819, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v5820 = v16 * 4;
      size_t v5821 = v5820 + 2;
      size_t v5822 = v5821 * v7;
      size_t v5823 = v12 * 16;
      size_t v5824 = v5822 + v5823;
      float* v5825 = v2 + v5824;
      vfloat32m2_t v5826 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v5825, v5826, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v5827 = v16 * 4;
      size_t v5828 = v5827 + 3;
      size_t v5829 = v5828 * v7;
      size_t v5830 = v12 * 16;
      size_t v5831 = v5829 + v5830;
      float* v5832 = v2 + v5831;
      vfloat32m2_t v5833 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v5832, v5833, 16);
    }
  }
  return;
}


