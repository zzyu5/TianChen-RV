#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5, size_t v6, size_t v7) {
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
      size_t v18 = v17 * 1344;
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
        size_t v29 = v28 * 1344;
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
        int16_t v44[128];
        int16_t v45[128];
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
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v58 = __riscv_vand_vx_u8mf2(v57, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v59 = __riscv_vsrl_vx_u8mf2(v57, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v60 = __riscv_vzext_vf2_u16m1(v58, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v61 = __riscv_vreinterpret_v_u16m1_i16m1(v60);
        int16_t* v62 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v62, v61, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v63 = __riscv_vzext_vf2_u16m1(v59, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v64 = __riscv_vreinterpret_v_u16m1_i16m1(v63);
        int16_t* v65 = &v45[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v65, v64, 16);
        const uint8_t* v66 = v30 + 80;
        const uint8_t* v67 = (const uint8_t*) v66;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v68 = __riscv_vle8_v_u8mf2(v67, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v69 = __riscv_vand_vx_u8mf2(v68, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v70 = __riscv_vsrl_vx_u8mf2(v68, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v71 = __riscv_vzext_vf2_u16m1(v69, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v72 = __riscv_vreinterpret_v_u16m1_i16m1(v71);
        int16_t* v73 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v73, v72, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v74 = __riscv_vzext_vf2_u16m1(v70, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v75 = __riscv_vreinterpret_v_u16m1_i16m1(v74);
        int16_t* v76 = &v45[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v76, v75, 16);
        const uint8_t* v77 = v30 + 96;
        const uint8_t* v78 = (const uint8_t*) v77;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v79 = __riscv_vle8_v_u8mf2(v78, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v80 = __riscv_vand_vx_u8mf2(v79, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v81 = __riscv_vsrl_vx_u8mf2(v79, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v82 = __riscv_vzext_vf2_u16m1(v80, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v83 = __riscv_vreinterpret_v_u16m1_i16m1(v82);
        int16_t* v84 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v84, v83, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v85 = __riscv_vzext_vf2_u16m1(v81, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v86 = __riscv_vreinterpret_v_u16m1_i16m1(v85);
        int16_t* v87 = &v45[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v87, v86, 16);
        const uint8_t* v88 = v30 + 112;
        const uint8_t* v89 = (const uint8_t*) v88;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v90 = __riscv_vle8_v_u8mf2(v89, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v91 = __riscv_vand_vx_u8mf2(v90, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v92 = __riscv_vsrl_vx_u8mf2(v90, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v93 = __riscv_vzext_vf2_u16m1(v91, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v94 = __riscv_vreinterpret_v_u16m1_i16m1(v93);
        int16_t* v95 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v95, v94, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v96 = __riscv_vzext_vf2_u16m1(v92, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v97 = __riscv_vreinterpret_v_u16m1_i16m1(v96);
        int16_t* v98 = &v45[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v98, v97, 16);
        const uint8_t* v99 = v30 + 128;
        const uint8_t* v100 = (const uint8_t*) v99;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v101 = __riscv_vle8_v_u8mf2(v100, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v102 = __riscv_vand_vx_u8mf2(v101, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v103 = __riscv_vsrl_vx_u8mf2(v101, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v104 = __riscv_vzext_vf2_u16m1(v102, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v105 = __riscv_vreinterpret_v_u16m1_i16m1(v104);
        int16_t* v106 = &v44[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v106, v105, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v107 = __riscv_vzext_vf2_u16m1(v103, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v108 = __riscv_vreinterpret_v_u16m1_i16m1(v107);
        int16_t* v109 = &v45[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v109, v108, 16);
        const uint8_t* v110 = v30 + 144;
        const uint8_t* v111 = (const uint8_t*) v110;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v112 = __riscv_vle8_v_u8mf2(v111, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v113 = __riscv_vand_vx_u8mf2(v112, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v114 = __riscv_vsrl_vx_u8mf2(v112, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v115 = __riscv_vzext_vf2_u16m1(v113, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v116 = __riscv_vreinterpret_v_u16m1_i16m1(v115);
        int16_t* v117 = &v44[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v117, v116, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v118 = __riscv_vzext_vf2_u16m1(v114, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v119 = __riscv_vreinterpret_v_u16m1_i16m1(v118);
        int16_t* v120 = &v45[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v120, v119, 16);
        const uint8_t* v121 = v30 + 160;
        const uint8_t* v122 = (const uint8_t*) v121;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v123 = __riscv_vle8_v_u8mf2(v122, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v124 = __riscv_vand_vx_u8mf2(v123, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v125 = __riscv_vsrl_vx_u8mf2(v123, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v126 = __riscv_vzext_vf2_u16m1(v124, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v127 = __riscv_vreinterpret_v_u16m1_i16m1(v126);
        int16_t* v128 = &v44[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v128, v127, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v129 = __riscv_vzext_vf2_u16m1(v125, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v130 = __riscv_vreinterpret_v_u16m1_i16m1(v129);
        int16_t* v131 = &v45[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v131, v130, 16);
        const uint8_t* v132 = v30 + 176;
        const uint8_t* v133 = (const uint8_t*) v132;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v134 = __riscv_vle8_v_u8mf2(v133, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v135 = __riscv_vand_vx_u8mf2(v134, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v136 = __riscv_vsrl_vx_u8mf2(v134, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v137 = __riscv_vzext_vf2_u16m1(v135, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v138 = __riscv_vreinterpret_v_u16m1_i16m1(v137);
        int16_t* v139 = &v44[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v139, v138, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v140 = __riscv_vzext_vf2_u16m1(v136, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v141 = __riscv_vreinterpret_v_u16m1_i16m1(v140);
        int16_t* v142 = &v45[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v142, v141, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold
        int16_t* v143 = &v45[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v144 = __riscv_vle16_v_i16m1(v143, 16);
        const uint8_t* v145 = v32 + 1040;
        const int16_t* v146 = (const int16_t*) v145;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v147 = *(const int16_t *)(v146);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v148 = __riscv_vmv_v_x_i32m2(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v149 = __riscv_vwmacc_vx_i32m2(v148, v147, v144, 16);
        int32_t* v150 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v150, v149, 16);
        const uint8_t* v151 = v32 + 1042;
        const int16_t* v152 = (const int16_t*) v151;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v153 = *(const int16_t *)(v152);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v154 = __riscv_vmv_v_x_i32m2(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v155 = __riscv_vwmacc_vx_i32m2(v154, v153, v144, 16);
        int32_t* v156 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v156, v155, 16);
        const uint8_t* v157 = v32 + 1044;
        const int16_t* v158 = (const int16_t*) v157;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v159 = *(const int16_t *)(v158);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v160 = __riscv_vmv_v_x_i32m2(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v161 = __riscv_vwmacc_vx_i32m2(v160, v159, v144, 16);
        int32_t* v162 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v162, v161, 16);
        const uint8_t* v163 = v32 + 1046;
        const int16_t* v164 = (const int16_t*) v163;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v165 = *(const int16_t *)(v164);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
        vint32m2_t v166 = __riscv_vmv_v_x_i32m2(0, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v167 = __riscv_vwmacc_vx_i32m2(v166, v165, v144, 16);
        int32_t* v168 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v168, v167, 16);
        int16_t* v169 = &v45[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v170 = __riscv_vle16_v_i16m1(v169, 16);
        const uint8_t* v171 = v32 + 1048;
        const int16_t* v172 = (const int16_t*) v171;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v173 = *(const int16_t *)(v172);
        int32_t* v174 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v175 = __riscv_vle32_v_i32m2(v174, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v176 = __riscv_vwmacc_vx_i32m2(v175, v173, v170, 16);
        int32_t* v177 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v177, v176, 16);
        const uint8_t* v178 = v32 + 1050;
        const int16_t* v179 = (const int16_t*) v178;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v180 = *(const int16_t *)(v179);
        int32_t* v181 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v182 = __riscv_vle32_v_i32m2(v181, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v183 = __riscv_vwmacc_vx_i32m2(v182, v180, v170, 16);
        int32_t* v184 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v184, v183, 16);
        const uint8_t* v185 = v32 + 1052;
        const int16_t* v186 = (const int16_t*) v185;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v187 = *(const int16_t *)(v186);
        int32_t* v188 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v189 = __riscv_vle32_v_i32m2(v188, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v190 = __riscv_vwmacc_vx_i32m2(v189, v187, v170, 16);
        int32_t* v191 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v191, v190, 16);
        const uint8_t* v192 = v32 + 1054;
        const int16_t* v193 = (const int16_t*) v192;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v194 = *(const int16_t *)(v193);
        int32_t* v195 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v196 = __riscv_vle32_v_i32m2(v195, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v197 = __riscv_vwmacc_vx_i32m2(v196, v194, v170, 16);
        int32_t* v198 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v198, v197, 16);
        int16_t* v199 = &v45[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v200 = __riscv_vle16_v_i16m1(v199, 16);
        const uint8_t* v201 = v32 + 1056;
        const int16_t* v202 = (const int16_t*) v201;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v203 = *(const int16_t *)(v202);
        int32_t* v204 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v205 = __riscv_vle32_v_i32m2(v204, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v206 = __riscv_vwmacc_vx_i32m2(v205, v203, v200, 16);
        int32_t* v207 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v207, v206, 16);
        const uint8_t* v208 = v32 + 1058;
        const int16_t* v209 = (const int16_t*) v208;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v210 = *(const int16_t *)(v209);
        int32_t* v211 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v212 = __riscv_vle32_v_i32m2(v211, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v213 = __riscv_vwmacc_vx_i32m2(v212, v210, v200, 16);
        int32_t* v214 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v214, v213, 16);
        const uint8_t* v215 = v32 + 1060;
        const int16_t* v216 = (const int16_t*) v215;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v217 = *(const int16_t *)(v216);
        int32_t* v218 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v219 = __riscv_vle32_v_i32m2(v218, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v220 = __riscv_vwmacc_vx_i32m2(v219, v217, v200, 16);
        int32_t* v221 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v221, v220, 16);
        const uint8_t* v222 = v32 + 1062;
        const int16_t* v223 = (const int16_t*) v222;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v224 = *(const int16_t *)(v223);
        int32_t* v225 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v226 = __riscv_vle32_v_i32m2(v225, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v227 = __riscv_vwmacc_vx_i32m2(v226, v224, v200, 16);
        int32_t* v228 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v228, v227, 16);
        int16_t* v229 = &v45[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v230 = __riscv_vle16_v_i16m1(v229, 16);
        const uint8_t* v231 = v32 + 1064;
        const int16_t* v232 = (const int16_t*) v231;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v233 = *(const int16_t *)(v232);
        int32_t* v234 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v235 = __riscv_vle32_v_i32m2(v234, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v236 = __riscv_vwmacc_vx_i32m2(v235, v233, v230, 16);
        int32_t* v237 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v237, v236, 16);
        const uint8_t* v238 = v32 + 1066;
        const int16_t* v239 = (const int16_t*) v238;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v240 = *(const int16_t *)(v239);
        int32_t* v241 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v242 = __riscv_vle32_v_i32m2(v241, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v243 = __riscv_vwmacc_vx_i32m2(v242, v240, v230, 16);
        int32_t* v244 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v244, v243, 16);
        const uint8_t* v245 = v32 + 1068;
        const int16_t* v246 = (const int16_t*) v245;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v247 = *(const int16_t *)(v246);
        int32_t* v248 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v249 = __riscv_vle32_v_i32m2(v248, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v250 = __riscv_vwmacc_vx_i32m2(v249, v247, v230, 16);
        int32_t* v251 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v251, v250, 16);
        const uint8_t* v252 = v32 + 1070;
        const int16_t* v253 = (const int16_t*) v252;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v254 = *(const int16_t *)(v253);
        int32_t* v255 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v256 = __riscv_vle32_v_i32m2(v255, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v257 = __riscv_vwmacc_vx_i32m2(v256, v254, v230, 16);
        int32_t* v258 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v258, v257, 16);
        int16_t* v259 = &v45[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v260 = __riscv_vle16_v_i16m1(v259, 16);
        const uint8_t* v261 = v32 + 1072;
        const int16_t* v262 = (const int16_t*) v261;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v263 = *(const int16_t *)(v262);
        int32_t* v264 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v265 = __riscv_vle32_v_i32m2(v264, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v266 = __riscv_vwmacc_vx_i32m2(v265, v263, v260, 16);
        int32_t* v267 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v267, v266, 16);
        const uint8_t* v268 = v32 + 1074;
        const int16_t* v269 = (const int16_t*) v268;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v270 = *(const int16_t *)(v269);
        int32_t* v271 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v272 = __riscv_vle32_v_i32m2(v271, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v273 = __riscv_vwmacc_vx_i32m2(v272, v270, v260, 16);
        int32_t* v274 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v274, v273, 16);
        const uint8_t* v275 = v32 + 1076;
        const int16_t* v276 = (const int16_t*) v275;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v277 = *(const int16_t *)(v276);
        int32_t* v278 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v279 = __riscv_vle32_v_i32m2(v278, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v280 = __riscv_vwmacc_vx_i32m2(v279, v277, v260, 16);
        int32_t* v281 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v281, v280, 16);
        const uint8_t* v282 = v32 + 1078;
        const int16_t* v283 = (const int16_t*) v282;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v284 = *(const int16_t *)(v283);
        int32_t* v285 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v286 = __riscv_vle32_v_i32m2(v285, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v287 = __riscv_vwmacc_vx_i32m2(v286, v284, v260, 16);
        int32_t* v288 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v288, v287, 16);
        int16_t* v289 = &v45[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v290 = __riscv_vle16_v_i16m1(v289, 16);
        const uint8_t* v291 = v32 + 1080;
        const int16_t* v292 = (const int16_t*) v291;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v293 = *(const int16_t *)(v292);
        int32_t* v294 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v295 = __riscv_vle32_v_i32m2(v294, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v296 = __riscv_vwmacc_vx_i32m2(v295, v293, v290, 16);
        int32_t* v297 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v297, v296, 16);
        const uint8_t* v298 = v32 + 1082;
        const int16_t* v299 = (const int16_t*) v298;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v300 = *(const int16_t *)(v299);
        int32_t* v301 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v302 = __riscv_vle32_v_i32m2(v301, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v303 = __riscv_vwmacc_vx_i32m2(v302, v300, v290, 16);
        int32_t* v304 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v304, v303, 16);
        const uint8_t* v305 = v32 + 1084;
        const int16_t* v306 = (const int16_t*) v305;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v307 = *(const int16_t *)(v306);
        int32_t* v308 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v309 = __riscv_vle32_v_i32m2(v308, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v310 = __riscv_vwmacc_vx_i32m2(v309, v307, v290, 16);
        int32_t* v311 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v311, v310, 16);
        const uint8_t* v312 = v32 + 1086;
        const int16_t* v313 = (const int16_t*) v312;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v314 = *(const int16_t *)(v313);
        int32_t* v315 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v316 = __riscv_vle32_v_i32m2(v315, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v317 = __riscv_vwmacc_vx_i32m2(v316, v314, v290, 16);
        int32_t* v318 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v318, v317, 16);
        int16_t* v319 = &v45[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v320 = __riscv_vle16_v_i16m1(v319, 16);
        const uint8_t* v321 = v32 + 1088;
        const int16_t* v322 = (const int16_t*) v321;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v323 = *(const int16_t *)(v322);
        int32_t* v324 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v325 = __riscv_vle32_v_i32m2(v324, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v326 = __riscv_vwmacc_vx_i32m2(v325, v323, v320, 16);
        int32_t* v327 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v327, v326, 16);
        const uint8_t* v328 = v32 + 1090;
        const int16_t* v329 = (const int16_t*) v328;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v330 = *(const int16_t *)(v329);
        int32_t* v331 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v332 = __riscv_vle32_v_i32m2(v331, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v333 = __riscv_vwmacc_vx_i32m2(v332, v330, v320, 16);
        int32_t* v334 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v334, v333, 16);
        const uint8_t* v335 = v32 + 1092;
        const int16_t* v336 = (const int16_t*) v335;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v337 = *(const int16_t *)(v336);
        int32_t* v338 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v339 = __riscv_vle32_v_i32m2(v338, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v340 = __riscv_vwmacc_vx_i32m2(v339, v337, v320, 16);
        int32_t* v341 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v341, v340, 16);
        const uint8_t* v342 = v32 + 1094;
        const int16_t* v343 = (const int16_t*) v342;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v344 = *(const int16_t *)(v343);
        int32_t* v345 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v346 = __riscv_vle32_v_i32m2(v345, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v347 = __riscv_vwmacc_vx_i32m2(v346, v344, v320, 16);
        int32_t* v348 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v348, v347, 16);
        int16_t* v349 = &v45[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v350 = __riscv_vle16_v_i16m1(v349, 16);
        const uint8_t* v351 = v32 + 1096;
        const int16_t* v352 = (const int16_t*) v351;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v353 = *(const int16_t *)(v352);
        int32_t* v354 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v355 = __riscv_vle32_v_i32m2(v354, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v356 = __riscv_vwmacc_vx_i32m2(v355, v353, v350, 16);
        int32_t* v357 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v357, v356, 16);
        const uint8_t* v358 = v32 + 1098;
        const int16_t* v359 = (const int16_t*) v358;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v360 = *(const int16_t *)(v359);
        int32_t* v361 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v362 = __riscv_vle32_v_i32m2(v361, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v363 = __riscv_vwmacc_vx_i32m2(v362, v360, v350, 16);
        int32_t* v364 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v364, v363, 16);
        const uint8_t* v365 = v32 + 1100;
        const int16_t* v366 = (const int16_t*) v365;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v367 = *(const int16_t *)(v366);
        int32_t* v368 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v369 = __riscv_vle32_v_i32m2(v368, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v370 = __riscv_vwmacc_vx_i32m2(v369, v367, v350, 16);
        int32_t* v371 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v371, v370, 16);
        const uint8_t* v372 = v32 + 1102;
        const int16_t* v373 = (const int16_t*) v372;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v374 = *(const int16_t *)(v373);
        int32_t* v375 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v376 = __riscv_vle32_v_i32m2(v375, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v377 = __riscv_vwmacc_vx_i32m2(v376, v374, v350, 16);
        int32_t* v378 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v378, v377, 16);
        int16_t v379[256];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v380 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v381 = &v379[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v381, v380, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v382 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v383 = &v379[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v383, v382, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v384 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v385 = &v379[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v385, v384, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v386 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v387 = &v379[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v387, v386, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v388 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v389 = &v379[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v389, v388, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v390 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v391 = &v379[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v391, v390, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v392 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v393 = &v379[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v393, v392, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v394 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v395 = &v379[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v395, v394, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v396 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v397 = &v379[128];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v397, v396, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v398 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v399 = &v379[144];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v399, v398, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v400 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v401 = &v379[160];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v401, v400, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v402 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v403 = &v379[176];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v403, v402, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v404 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v405 = &v379[192];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v405, v404, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v406 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v407 = &v379[208];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v407, v406, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v408 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v409 = &v379[224];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v409, v408, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v410 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v411 = &v379[240];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v411, v410, 16);
        for (size_t v412 = 0; v412 < 16; v412 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
          size_t v413 = v412 * 16;
          size_t v414 = 320 + v413;
          const uint8_t* v415 = v30 + v414;
          const uint8_t* v416 = (const uint8_t*) v415;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v417 = __riscv_vle8_v_u8mf2(v416, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v418 = __riscv_vand_vx_u8mf2(v417, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v419 = __riscv_vreinterpret_v_u8mf2_i8mf2(v418);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v420 = __riscv_vsrl_vx_u8mf2(v417, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v421 = __riscv_vand_vx_u8mf2(v420, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v422 = __riscv_vreinterpret_v_u8mf2_i8mf2(v421);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v423 = __riscv_vsrl_vx_u8mf2(v417, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v424 = __riscv_vand_vx_u8mf2(v423, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v425 = __riscv_vreinterpret_v_u8mf2_i8mf2(v424);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v426 = __riscv_vsrl_vx_u8mf2(v417, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v427 = __riscv_vand_vx_u8mf2(v426, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v428 = __riscv_vreinterpret_v_u8mf2_i8mf2(v427);
          size_t v429 = v412 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v430 = 16 + v429;
          const uint8_t* v431 = v32 + v430;
          const int8_t* v432 = (const int8_t*) v431;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v433 = *(const int8_t *)(v432);
          int16_t* v434 = &v379[0];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v435 = __riscv_vle16_v_i16m1(v434, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v436 = __riscv_vwmacc_vx_i16m1(v435, v433, v419, 16);
          int16_t* v437 = &v379[0];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v437, v436, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v438 = 144 + v429;
          const uint8_t* v439 = v32 + v438;
          const int8_t* v440 = (const int8_t*) v439;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v441 = *(const int8_t *)(v440);
          int16_t* v442 = &v379[16];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v443 = __riscv_vle16_v_i16m1(v442, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v444 = __riscv_vwmacc_vx_i16m1(v443, v441, v422, 16);
          int16_t* v445 = &v379[16];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v445, v444, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v446 = 272 + v429;
          const uint8_t* v447 = v32 + v446;
          const int8_t* v448 = (const int8_t*) v447;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v449 = *(const int8_t *)(v448);
          int16_t* v450 = &v379[32];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v451 = __riscv_vle16_v_i16m1(v450, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v452 = __riscv_vwmacc_vx_i16m1(v451, v449, v425, 16);
          int16_t* v453 = &v379[32];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v453, v452, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v454 = 400 + v429;
          const uint8_t* v455 = v32 + v454;
          const int8_t* v456 = (const int8_t*) v455;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v457 = *(const int8_t *)(v456);
          int16_t* v458 = &v379[48];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v459 = __riscv_vle16_v_i16m1(v458, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v460 = __riscv_vwmacc_vx_i16m1(v459, v457, v428, 16);
          int16_t* v461 = &v379[48];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v461, v460, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v462 = 17 + v429;
          const uint8_t* v463 = v32 + v462;
          const int8_t* v464 = (const int8_t*) v463;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v465 = *(const int8_t *)(v464);
          int16_t* v466 = &v379[64];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v467 = __riscv_vle16_v_i16m1(v466, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v468 = __riscv_vwmacc_vx_i16m1(v467, v465, v419, 16);
          int16_t* v469 = &v379[64];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v469, v468, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v470 = 145 + v429;
          const uint8_t* v471 = v32 + v470;
          const int8_t* v472 = (const int8_t*) v471;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v473 = *(const int8_t *)(v472);
          int16_t* v474 = &v379[80];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v475 = __riscv_vle16_v_i16m1(v474, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v476 = __riscv_vwmacc_vx_i16m1(v475, v473, v422, 16);
          int16_t* v477 = &v379[80];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v477, v476, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v478 = 273 + v429;
          const uint8_t* v479 = v32 + v478;
          const int8_t* v480 = (const int8_t*) v479;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v481 = *(const int8_t *)(v480);
          int16_t* v482 = &v379[96];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v483 = __riscv_vle16_v_i16m1(v482, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v484 = __riscv_vwmacc_vx_i16m1(v483, v481, v425, 16);
          int16_t* v485 = &v379[96];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v485, v484, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v486 = 401 + v429;
          const uint8_t* v487 = v32 + v486;
          const int8_t* v488 = (const int8_t*) v487;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v489 = *(const int8_t *)(v488);
          int16_t* v490 = &v379[112];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v491 = __riscv_vle16_v_i16m1(v490, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v492 = __riscv_vwmacc_vx_i16m1(v491, v489, v428, 16);
          int16_t* v493 = &v379[112];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v493, v492, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v494 = 18 + v429;
          const uint8_t* v495 = v32 + v494;
          const int8_t* v496 = (const int8_t*) v495;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v497 = *(const int8_t *)(v496);
          int16_t* v498 = &v379[128];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v499 = __riscv_vle16_v_i16m1(v498, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v500 = __riscv_vwmacc_vx_i16m1(v499, v497, v419, 16);
          int16_t* v501 = &v379[128];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v501, v500, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v502 = 146 + v429;
          const uint8_t* v503 = v32 + v502;
          const int8_t* v504 = (const int8_t*) v503;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v505 = *(const int8_t *)(v504);
          int16_t* v506 = &v379[144];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v507 = __riscv_vle16_v_i16m1(v506, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v508 = __riscv_vwmacc_vx_i16m1(v507, v505, v422, 16);
          int16_t* v509 = &v379[144];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v509, v508, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v510 = 274 + v429;
          const uint8_t* v511 = v32 + v510;
          const int8_t* v512 = (const int8_t*) v511;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v513 = *(const int8_t *)(v512);
          int16_t* v514 = &v379[160];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v515 = __riscv_vle16_v_i16m1(v514, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v516 = __riscv_vwmacc_vx_i16m1(v515, v513, v425, 16);
          int16_t* v517 = &v379[160];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v517, v516, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v518 = 402 + v429;
          const uint8_t* v519 = v32 + v518;
          const int8_t* v520 = (const int8_t*) v519;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v521 = *(const int8_t *)(v520);
          int16_t* v522 = &v379[176];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v523 = __riscv_vle16_v_i16m1(v522, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v524 = __riscv_vwmacc_vx_i16m1(v523, v521, v428, 16);
          int16_t* v525 = &v379[176];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v525, v524, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v526 = 19 + v429;
          const uint8_t* v527 = v32 + v526;
          const int8_t* v528 = (const int8_t*) v527;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v529 = *(const int8_t *)(v528);
          int16_t* v530 = &v379[192];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v531 = __riscv_vle16_v_i16m1(v530, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v532 = __riscv_vwmacc_vx_i16m1(v531, v529, v419, 16);
          int16_t* v533 = &v379[192];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v533, v532, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v534 = 147 + v429;
          const uint8_t* v535 = v32 + v534;
          const int8_t* v536 = (const int8_t*) v535;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v537 = *(const int8_t *)(v536);
          int16_t* v538 = &v379[208];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v539 = __riscv_vle16_v_i16m1(v538, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v540 = __riscv_vwmacc_vx_i16m1(v539, v537, v422, 16);
          int16_t* v541 = &v379[208];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v541, v540, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v542 = 275 + v429;
          const uint8_t* v543 = v32 + v542;
          const int8_t* v544 = (const int8_t*) v543;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v545 = *(const int8_t *)(v544);
          int16_t* v546 = &v379[224];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v547 = __riscv_vle16_v_i16m1(v546, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v548 = __riscv_vwmacc_vx_i16m1(v547, v545, v425, 16);
          int16_t* v549 = &v379[224];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v549, v548, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v550 = 403 + v429;
          const uint8_t* v551 = v32 + v550;
          const int8_t* v552 = (const int8_t*) v551;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v553 = *(const int8_t *)(v552);
          int16_t* v554 = &v379[240];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v555 = __riscv_vle16_v_i16m1(v554, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v556 = __riscv_vwmacc_vx_i16m1(v555, v553, v428, 16);
          int16_t* v557 = &v379[240];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v557, v556, 16);
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v558 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v559 = __riscv_vle16_v_i16m1(v558, 16);
        int16_t* v560 = &v379[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v561 = __riscv_vle16_v_i16m1(v560, 16);
        vint32m2_t v562 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v563 = __riscv_vwmacc_vv_i32m2(v562, v559, v561, 16);
        v47 = v563;
        int16_t* v564 = &v379[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v565 = __riscv_vle16_v_i16m1(v564, 16);
        vint32m2_t v566 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v567 = __riscv_vwmacc_vv_i32m2(v566, v559, v565, 16);
        v49 = v567;
        int16_t* v568 = &v379[128];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v569 = __riscv_vle16_v_i16m1(v568, 16);
        vint32m2_t v570 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v571 = __riscv_vwmacc_vv_i32m2(v570, v559, v569, 16);
        v51 = v571;
        int16_t* v572 = &v379[192];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v573 = __riscv_vle16_v_i16m1(v572, 16);
        vint32m2_t v574 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v575 = __riscv_vwmacc_vv_i32m2(v574, v559, v573, 16);
        v53 = v575;
        int16_t* v576 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v577 = __riscv_vle16_v_i16m1(v576, 16);
        int16_t* v578 = &v379[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v579 = __riscv_vle16_v_i16m1(v578, 16);
        vint32m2_t v580 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v581 = __riscv_vwmacc_vv_i32m2(v580, v577, v579, 16);
        v47 = v581;
        int16_t* v582 = &v379[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v583 = __riscv_vle16_v_i16m1(v582, 16);
        vint32m2_t v584 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v585 = __riscv_vwmacc_vv_i32m2(v584, v577, v583, 16);
        v49 = v585;
        int16_t* v586 = &v379[144];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v587 = __riscv_vle16_v_i16m1(v586, 16);
        vint32m2_t v588 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v589 = __riscv_vwmacc_vv_i32m2(v588, v577, v587, 16);
        v51 = v589;
        int16_t* v590 = &v379[208];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v591 = __riscv_vle16_v_i16m1(v590, 16);
        vint32m2_t v592 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v593 = __riscv_vwmacc_vv_i32m2(v592, v577, v591, 16);
        v53 = v593;
        int16_t* v594 = &v44[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v595 = __riscv_vle16_v_i16m1(v594, 16);
        int16_t* v596 = &v379[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v597 = __riscv_vle16_v_i16m1(v596, 16);
        vint32m2_t v598 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v599 = __riscv_vwmacc_vv_i32m2(v598, v595, v597, 16);
        v47 = v599;
        int16_t* v600 = &v379[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v601 = __riscv_vle16_v_i16m1(v600, 16);
        vint32m2_t v602 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v603 = __riscv_vwmacc_vv_i32m2(v602, v595, v601, 16);
        v49 = v603;
        int16_t* v604 = &v379[160];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v605 = __riscv_vle16_v_i16m1(v604, 16);
        vint32m2_t v606 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v607 = __riscv_vwmacc_vv_i32m2(v606, v595, v605, 16);
        v51 = v607;
        int16_t* v608 = &v379[224];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v609 = __riscv_vle16_v_i16m1(v608, 16);
        vint32m2_t v610 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v611 = __riscv_vwmacc_vv_i32m2(v610, v595, v609, 16);
        v53 = v611;
        int16_t* v612 = &v44[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v613 = __riscv_vle16_v_i16m1(v612, 16);
        int16_t* v614 = &v379[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v615 = __riscv_vle16_v_i16m1(v614, 16);
        vint32m2_t v616 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v617 = __riscv_vwmacc_vv_i32m2(v616, v613, v615, 16);
        v47 = v617;
        int16_t* v618 = &v379[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v619 = __riscv_vle16_v_i16m1(v618, 16);
        vint32m2_t v620 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v621 = __riscv_vwmacc_vv_i32m2(v620, v613, v619, 16);
        v49 = v621;
        int16_t* v622 = &v379[176];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v623 = __riscv_vle16_v_i16m1(v622, 16);
        vint32m2_t v624 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v625 = __riscv_vwmacc_vv_i32m2(v624, v613, v623, 16);
        v51 = v625;
        int16_t* v626 = &v379[240];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v627 = __riscv_vle16_v_i16m1(v626, 16);
        vint32m2_t v628 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v629 = __riscv_vwmacc_vv_i32m2(v628, v613, v627, 16);
        v53 = v629;
        int16_t v630[256];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v631 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v632 = &v630[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v632, v631, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v633 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v634 = &v630[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v634, v633, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v635 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v636 = &v630[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v636, v635, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v637 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v638 = &v630[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v638, v637, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v639 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v640 = &v630[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v640, v639, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v641 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v642 = &v630[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v642, v641, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v643 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v644 = &v630[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v644, v643, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v645 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v646 = &v630[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v646, v645, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v647 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v648 = &v630[128];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v648, v647, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v649 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v650 = &v630[144];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v650, v649, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v651 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v652 = &v630[160];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v652, v651, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v653 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v654 = &v630[176];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v654, v653, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v655 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v656 = &v630[192];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v656, v655, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v657 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v658 = &v630[208];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v658, v657, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v659 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v660 = &v630[224];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v660, v659, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v661 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v662 = &v630[240];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v662, v661, 16);
        for (size_t v663 = 0; v663 < 16; v663 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
          size_t v664 = v663 * 16;
          size_t v665 = 576 + v664;
          const uint8_t* v666 = v30 + v665;
          const uint8_t* v667 = (const uint8_t*) v666;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v668 = __riscv_vle8_v_u8mf2(v667, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v669 = __riscv_vand_vx_u8mf2(v668, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v670 = __riscv_vreinterpret_v_u8mf2_i8mf2(v669);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v671 = __riscv_vsrl_vx_u8mf2(v668, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v672 = __riscv_vand_vx_u8mf2(v671, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v673 = __riscv_vreinterpret_v_u8mf2_i8mf2(v672);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v674 = __riscv_vsrl_vx_u8mf2(v668, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v675 = __riscv_vand_vx_u8mf2(v674, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v676 = __riscv_vreinterpret_v_u8mf2_i8mf2(v675);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v677 = __riscv_vsrl_vx_u8mf2(v668, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v678 = __riscv_vand_vx_u8mf2(v677, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v679 = __riscv_vreinterpret_v_u8mf2_i8mf2(v678);
          size_t v680 = v663 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v681 = 80 + v680;
          const uint8_t* v682 = v32 + v681;
          const int8_t* v683 = (const int8_t*) v682;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v684 = *(const int8_t *)(v683);
          int16_t* v685 = &v630[0];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v686 = __riscv_vle16_v_i16m1(v685, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v687 = __riscv_vwmacc_vx_i16m1(v686, v684, v670, 16);
          int16_t* v688 = &v630[0];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v688, v687, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v689 = 208 + v680;
          const uint8_t* v690 = v32 + v689;
          const int8_t* v691 = (const int8_t*) v690;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v692 = *(const int8_t *)(v691);
          int16_t* v693 = &v630[16];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v694 = __riscv_vle16_v_i16m1(v693, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v695 = __riscv_vwmacc_vx_i16m1(v694, v692, v673, 16);
          int16_t* v696 = &v630[16];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v696, v695, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v697 = 336 + v680;
          const uint8_t* v698 = v32 + v697;
          const int8_t* v699 = (const int8_t*) v698;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v700 = *(const int8_t *)(v699);
          int16_t* v701 = &v630[32];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v702 = __riscv_vle16_v_i16m1(v701, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v703 = __riscv_vwmacc_vx_i16m1(v702, v700, v676, 16);
          int16_t* v704 = &v630[32];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v704, v703, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v705 = 464 + v680;
          const uint8_t* v706 = v32 + v705;
          const int8_t* v707 = (const int8_t*) v706;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v708 = *(const int8_t *)(v707);
          int16_t* v709 = &v630[48];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v710 = __riscv_vle16_v_i16m1(v709, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v711 = __riscv_vwmacc_vx_i16m1(v710, v708, v679, 16);
          int16_t* v712 = &v630[48];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v712, v711, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v713 = 81 + v680;
          const uint8_t* v714 = v32 + v713;
          const int8_t* v715 = (const int8_t*) v714;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v716 = *(const int8_t *)(v715);
          int16_t* v717 = &v630[64];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v718 = __riscv_vle16_v_i16m1(v717, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v719 = __riscv_vwmacc_vx_i16m1(v718, v716, v670, 16);
          int16_t* v720 = &v630[64];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v720, v719, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v721 = 209 + v680;
          const uint8_t* v722 = v32 + v721;
          const int8_t* v723 = (const int8_t*) v722;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v724 = *(const int8_t *)(v723);
          int16_t* v725 = &v630[80];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v726 = __riscv_vle16_v_i16m1(v725, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v727 = __riscv_vwmacc_vx_i16m1(v726, v724, v673, 16);
          int16_t* v728 = &v630[80];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v728, v727, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v729 = 337 + v680;
          const uint8_t* v730 = v32 + v729;
          const int8_t* v731 = (const int8_t*) v730;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v732 = *(const int8_t *)(v731);
          int16_t* v733 = &v630[96];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v734 = __riscv_vle16_v_i16m1(v733, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v735 = __riscv_vwmacc_vx_i16m1(v734, v732, v676, 16);
          int16_t* v736 = &v630[96];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v736, v735, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v737 = 465 + v680;
          const uint8_t* v738 = v32 + v737;
          const int8_t* v739 = (const int8_t*) v738;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v740 = *(const int8_t *)(v739);
          int16_t* v741 = &v630[112];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v742 = __riscv_vle16_v_i16m1(v741, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v743 = __riscv_vwmacc_vx_i16m1(v742, v740, v679, 16);
          int16_t* v744 = &v630[112];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v744, v743, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v745 = 82 + v680;
          const uint8_t* v746 = v32 + v745;
          const int8_t* v747 = (const int8_t*) v746;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v748 = *(const int8_t *)(v747);
          int16_t* v749 = &v630[128];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v750 = __riscv_vle16_v_i16m1(v749, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v751 = __riscv_vwmacc_vx_i16m1(v750, v748, v670, 16);
          int16_t* v752 = &v630[128];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v752, v751, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v753 = 210 + v680;
          const uint8_t* v754 = v32 + v753;
          const int8_t* v755 = (const int8_t*) v754;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v756 = *(const int8_t *)(v755);
          int16_t* v757 = &v630[144];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v758 = __riscv_vle16_v_i16m1(v757, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v759 = __riscv_vwmacc_vx_i16m1(v758, v756, v673, 16);
          int16_t* v760 = &v630[144];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v760, v759, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v761 = 338 + v680;
          const uint8_t* v762 = v32 + v761;
          const int8_t* v763 = (const int8_t*) v762;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v764 = *(const int8_t *)(v763);
          int16_t* v765 = &v630[160];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v766 = __riscv_vle16_v_i16m1(v765, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v767 = __riscv_vwmacc_vx_i16m1(v766, v764, v676, 16);
          int16_t* v768 = &v630[160];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v768, v767, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v769 = 466 + v680;
          const uint8_t* v770 = v32 + v769;
          const int8_t* v771 = (const int8_t*) v770;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v772 = *(const int8_t *)(v771);
          int16_t* v773 = &v630[176];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v774 = __riscv_vle16_v_i16m1(v773, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v775 = __riscv_vwmacc_vx_i16m1(v774, v772, v679, 16);
          int16_t* v776 = &v630[176];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v776, v775, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v777 = 83 + v680;
          const uint8_t* v778 = v32 + v777;
          const int8_t* v779 = (const int8_t*) v778;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v780 = *(const int8_t *)(v779);
          int16_t* v781 = &v630[192];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v782 = __riscv_vle16_v_i16m1(v781, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v783 = __riscv_vwmacc_vx_i16m1(v782, v780, v670, 16);
          int16_t* v784 = &v630[192];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v784, v783, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v785 = 211 + v680;
          const uint8_t* v786 = v32 + v785;
          const int8_t* v787 = (const int8_t*) v786;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v788 = *(const int8_t *)(v787);
          int16_t* v789 = &v630[208];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v790 = __riscv_vle16_v_i16m1(v789, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v791 = __riscv_vwmacc_vx_i16m1(v790, v788, v673, 16);
          int16_t* v792 = &v630[208];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v792, v791, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v793 = 339 + v680;
          const uint8_t* v794 = v32 + v793;
          const int8_t* v795 = (const int8_t*) v794;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v796 = *(const int8_t *)(v795);
          int16_t* v797 = &v630[224];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v798 = __riscv_vle16_v_i16m1(v797, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v799 = __riscv_vwmacc_vx_i16m1(v798, v796, v676, 16);
          int16_t* v800 = &v630[224];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v800, v799, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v801 = 467 + v680;
          const uint8_t* v802 = v32 + v801;
          const int8_t* v803 = (const int8_t*) v802;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v804 = *(const int8_t *)(v803);
          int16_t* v805 = &v630[240];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v806 = __riscv_vle16_v_i16m1(v805, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v807 = __riscv_vwmacc_vx_i16m1(v806, v804, v679, 16);
          int16_t* v808 = &v630[240];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v808, v807, 16);
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v809 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v810 = __riscv_vle16_v_i16m1(v809, 16);
        int16_t* v811 = &v630[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v812 = __riscv_vle16_v_i16m1(v811, 16);
        vint32m2_t v813 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v814 = __riscv_vwmacc_vv_i32m2(v813, v810, v812, 16);
        v47 = v814;
        int16_t* v815 = &v630[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v816 = __riscv_vle16_v_i16m1(v815, 16);
        vint32m2_t v817 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v818 = __riscv_vwmacc_vv_i32m2(v817, v810, v816, 16);
        v49 = v818;
        int16_t* v819 = &v630[128];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v820 = __riscv_vle16_v_i16m1(v819, 16);
        vint32m2_t v821 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v822 = __riscv_vwmacc_vv_i32m2(v821, v810, v820, 16);
        v51 = v822;
        int16_t* v823 = &v630[192];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v824 = __riscv_vle16_v_i16m1(v823, 16);
        vint32m2_t v825 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v826 = __riscv_vwmacc_vv_i32m2(v825, v810, v824, 16);
        v53 = v826;
        int16_t* v827 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v828 = __riscv_vle16_v_i16m1(v827, 16);
        int16_t* v829 = &v630[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v830 = __riscv_vle16_v_i16m1(v829, 16);
        vint32m2_t v831 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v832 = __riscv_vwmacc_vv_i32m2(v831, v828, v830, 16);
        v47 = v832;
        int16_t* v833 = &v630[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v834 = __riscv_vle16_v_i16m1(v833, 16);
        vint32m2_t v835 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v836 = __riscv_vwmacc_vv_i32m2(v835, v828, v834, 16);
        v49 = v836;
        int16_t* v837 = &v630[144];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v838 = __riscv_vle16_v_i16m1(v837, 16);
        vint32m2_t v839 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v840 = __riscv_vwmacc_vv_i32m2(v839, v828, v838, 16);
        v51 = v840;
        int16_t* v841 = &v630[208];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v842 = __riscv_vle16_v_i16m1(v841, 16);
        vint32m2_t v843 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v844 = __riscv_vwmacc_vv_i32m2(v843, v828, v842, 16);
        v53 = v844;
        int16_t* v845 = &v44[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v846 = __riscv_vle16_v_i16m1(v845, 16);
        int16_t* v847 = &v630[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v848 = __riscv_vle16_v_i16m1(v847, 16);
        vint32m2_t v849 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v850 = __riscv_vwmacc_vv_i32m2(v849, v846, v848, 16);
        v47 = v850;
        int16_t* v851 = &v630[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v852 = __riscv_vle16_v_i16m1(v851, 16);
        vint32m2_t v853 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v854 = __riscv_vwmacc_vv_i32m2(v853, v846, v852, 16);
        v49 = v854;
        int16_t* v855 = &v630[160];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v856 = __riscv_vle16_v_i16m1(v855, 16);
        vint32m2_t v857 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v858 = __riscv_vwmacc_vv_i32m2(v857, v846, v856, 16);
        v51 = v858;
        int16_t* v859 = &v630[224];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v860 = __riscv_vle16_v_i16m1(v859, 16);
        vint32m2_t v861 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v862 = __riscv_vwmacc_vv_i32m2(v861, v846, v860, 16);
        v53 = v862;
        int16_t* v863 = &v44[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v864 = __riscv_vle16_v_i16m1(v863, 16);
        int16_t* v865 = &v630[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v866 = __riscv_vle16_v_i16m1(v865, 16);
        vint32m2_t v867 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v868 = __riscv_vwmacc_vv_i32m2(v867, v864, v866, 16);
        v47 = v868;
        int16_t* v869 = &v630[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v870 = __riscv_vle16_v_i16m1(v869, 16);
        vint32m2_t v871 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v872 = __riscv_vwmacc_vv_i32m2(v871, v864, v870, 16);
        v49 = v872;
        int16_t* v873 = &v630[176];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v874 = __riscv_vle16_v_i16m1(v873, 16);
        vint32m2_t v875 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v876 = __riscv_vwmacc_vv_i32m2(v875, v864, v874, 16);
        v51 = v876;
        int16_t* v877 = &v630[240];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v878 = __riscv_vle16_v_i16m1(v877, 16);
        vint32m2_t v879 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v880 = __riscv_vwmacc_vv_i32m2(v879, v864, v878, 16);
        v53 = v880;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
        const uint8_t* v881 = v30 + 192;
        const uint8_t* v882 = (const uint8_t*) v881;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v883 = __riscv_vle8_v_u8mf2(v882, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v884 = __riscv_vand_vx_u8mf2(v883, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v885 = __riscv_vsrl_vx_u8mf2(v883, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v886 = __riscv_vzext_vf2_u16m1(v884, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v887 = __riscv_vreinterpret_v_u16m1_i16m1(v886);
        int16_t* v888 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v888, v887, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v889 = __riscv_vzext_vf2_u16m1(v885, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v890 = __riscv_vreinterpret_v_u16m1_i16m1(v889);
        int16_t* v891 = &v45[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v891, v890, 16);
        const uint8_t* v892 = v30 + 208;
        const uint8_t* v893 = (const uint8_t*) v892;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v894 = __riscv_vle8_v_u8mf2(v893, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v895 = __riscv_vand_vx_u8mf2(v894, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v896 = __riscv_vsrl_vx_u8mf2(v894, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v897 = __riscv_vzext_vf2_u16m1(v895, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v898 = __riscv_vreinterpret_v_u16m1_i16m1(v897);
        int16_t* v899 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v899, v898, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v900 = __riscv_vzext_vf2_u16m1(v896, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v901 = __riscv_vreinterpret_v_u16m1_i16m1(v900);
        int16_t* v902 = &v45[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v902, v901, 16);
        const uint8_t* v903 = v30 + 224;
        const uint8_t* v904 = (const uint8_t*) v903;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v905 = __riscv_vle8_v_u8mf2(v904, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v906 = __riscv_vand_vx_u8mf2(v905, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v907 = __riscv_vsrl_vx_u8mf2(v905, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v908 = __riscv_vzext_vf2_u16m1(v906, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v909 = __riscv_vreinterpret_v_u16m1_i16m1(v908);
        int16_t* v910 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v910, v909, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v911 = __riscv_vzext_vf2_u16m1(v907, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v912 = __riscv_vreinterpret_v_u16m1_i16m1(v911);
        int16_t* v913 = &v45[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v913, v912, 16);
        const uint8_t* v914 = v30 + 240;
        const uint8_t* v915 = (const uint8_t*) v914;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v916 = __riscv_vle8_v_u8mf2(v915, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v917 = __riscv_vand_vx_u8mf2(v916, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v918 = __riscv_vsrl_vx_u8mf2(v916, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v919 = __riscv_vzext_vf2_u16m1(v917, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v920 = __riscv_vreinterpret_v_u16m1_i16m1(v919);
        int16_t* v921 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v921, v920, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v922 = __riscv_vzext_vf2_u16m1(v918, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v923 = __riscv_vreinterpret_v_u16m1_i16m1(v922);
        int16_t* v924 = &v45[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v924, v923, 16);
        const uint8_t* v925 = v30 + 256;
        const uint8_t* v926 = (const uint8_t*) v925;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v927 = __riscv_vle8_v_u8mf2(v926, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v928 = __riscv_vand_vx_u8mf2(v927, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v929 = __riscv_vsrl_vx_u8mf2(v927, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v930 = __riscv_vzext_vf2_u16m1(v928, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v931 = __riscv_vreinterpret_v_u16m1_i16m1(v930);
        int16_t* v932 = &v44[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v932, v931, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v933 = __riscv_vzext_vf2_u16m1(v929, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v934 = __riscv_vreinterpret_v_u16m1_i16m1(v933);
        int16_t* v935 = &v45[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v935, v934, 16);
        const uint8_t* v936 = v30 + 272;
        const uint8_t* v937 = (const uint8_t*) v936;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v938 = __riscv_vle8_v_u8mf2(v937, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v939 = __riscv_vand_vx_u8mf2(v938, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v940 = __riscv_vsrl_vx_u8mf2(v938, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v941 = __riscv_vzext_vf2_u16m1(v939, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v942 = __riscv_vreinterpret_v_u16m1_i16m1(v941);
        int16_t* v943 = &v44[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v943, v942, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v944 = __riscv_vzext_vf2_u16m1(v940, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v945 = __riscv_vreinterpret_v_u16m1_i16m1(v944);
        int16_t* v946 = &v45[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v946, v945, 16);
        const uint8_t* v947 = v30 + 288;
        const uint8_t* v948 = (const uint8_t*) v947;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v949 = __riscv_vle8_v_u8mf2(v948, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v950 = __riscv_vand_vx_u8mf2(v949, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v951 = __riscv_vsrl_vx_u8mf2(v949, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v952 = __riscv_vzext_vf2_u16m1(v950, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v953 = __riscv_vreinterpret_v_u16m1_i16m1(v952);
        int16_t* v954 = &v44[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v954, v953, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v955 = __riscv_vzext_vf2_u16m1(v951, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v956 = __riscv_vreinterpret_v_u16m1_i16m1(v955);
        int16_t* v957 = &v45[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v957, v956, 16);
        const uint8_t* v958 = v30 + 304;
        const uint8_t* v959 = (const uint8_t*) v958;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v960 = __riscv_vle8_v_u8mf2(v959, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v961 = __riscv_vand_vx_u8mf2(v960, 0x0F, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v962 = __riscv_vsrl_vx_u8mf2(v960, 4, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v963 = __riscv_vzext_vf2_u16m1(v961, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v964 = __riscv_vreinterpret_v_u16m1_i16m1(v963);
        int16_t* v965 = &v44[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v965, v964, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
        vuint16m1_t v966 = __riscv_vzext_vf2_u16m1(v962, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
        vint16m1_t v967 = __riscv_vreinterpret_v_u16m1_i16m1(v966);
        int16_t* v968 = &v45[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v968, v967, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold
        int16_t* v969 = &v45[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v970 = __riscv_vle16_v_i16m1(v969, 16);
        const uint8_t* v971 = v32 + 1104;
        const int16_t* v972 = (const int16_t*) v971;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v973 = *(const int16_t *)(v972);
        int32_t* v974 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v975 = __riscv_vle32_v_i32m2(v974, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v976 = __riscv_vwmacc_vx_i32m2(v975, v973, v970, 16);
        int32_t* v977 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v977, v976, 16);
        const uint8_t* v978 = v32 + 1106;
        const int16_t* v979 = (const int16_t*) v978;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v980 = *(const int16_t *)(v979);
        int32_t* v981 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v982 = __riscv_vle32_v_i32m2(v981, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v983 = __riscv_vwmacc_vx_i32m2(v982, v980, v970, 16);
        int32_t* v984 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v984, v983, 16);
        const uint8_t* v985 = v32 + 1108;
        const int16_t* v986 = (const int16_t*) v985;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v987 = *(const int16_t *)(v986);
        int32_t* v988 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v989 = __riscv_vle32_v_i32m2(v988, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v990 = __riscv_vwmacc_vx_i32m2(v989, v987, v970, 16);
        int32_t* v991 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v991, v990, 16);
        const uint8_t* v992 = v32 + 1110;
        const int16_t* v993 = (const int16_t*) v992;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v994 = *(const int16_t *)(v993);
        int32_t* v995 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v996 = __riscv_vle32_v_i32m2(v995, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v997 = __riscv_vwmacc_vx_i32m2(v996, v994, v970, 16);
        int32_t* v998 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v998, v997, 16);
        int16_t* v999 = &v45[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1000 = __riscv_vle16_v_i16m1(v999, 16);
        const uint8_t* v1001 = v32 + 1112;
        const int16_t* v1002 = (const int16_t*) v1001;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1003 = *(const int16_t *)(v1002);
        int32_t* v1004 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1005 = __riscv_vle32_v_i32m2(v1004, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1006 = __riscv_vwmacc_vx_i32m2(v1005, v1003, v1000, 16);
        int32_t* v1007 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1007, v1006, 16);
        const uint8_t* v1008 = v32 + 1114;
        const int16_t* v1009 = (const int16_t*) v1008;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1010 = *(const int16_t *)(v1009);
        int32_t* v1011 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1012 = __riscv_vle32_v_i32m2(v1011, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1013 = __riscv_vwmacc_vx_i32m2(v1012, v1010, v1000, 16);
        int32_t* v1014 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1014, v1013, 16);
        const uint8_t* v1015 = v32 + 1116;
        const int16_t* v1016 = (const int16_t*) v1015;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1017 = *(const int16_t *)(v1016);
        int32_t* v1018 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1019 = __riscv_vle32_v_i32m2(v1018, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1020 = __riscv_vwmacc_vx_i32m2(v1019, v1017, v1000, 16);
        int32_t* v1021 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1021, v1020, 16);
        const uint8_t* v1022 = v32 + 1118;
        const int16_t* v1023 = (const int16_t*) v1022;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1024 = *(const int16_t *)(v1023);
        int32_t* v1025 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1026 = __riscv_vle32_v_i32m2(v1025, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1027 = __riscv_vwmacc_vx_i32m2(v1026, v1024, v1000, 16);
        int32_t* v1028 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1028, v1027, 16);
        int16_t* v1029 = &v45[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1030 = __riscv_vle16_v_i16m1(v1029, 16);
        const uint8_t* v1031 = v32 + 1120;
        const int16_t* v1032 = (const int16_t*) v1031;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1033 = *(const int16_t *)(v1032);
        int32_t* v1034 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1035 = __riscv_vle32_v_i32m2(v1034, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1036 = __riscv_vwmacc_vx_i32m2(v1035, v1033, v1030, 16);
        int32_t* v1037 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1037, v1036, 16);
        const uint8_t* v1038 = v32 + 1122;
        const int16_t* v1039 = (const int16_t*) v1038;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1040 = *(const int16_t *)(v1039);
        int32_t* v1041 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1042 = __riscv_vle32_v_i32m2(v1041, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1043 = __riscv_vwmacc_vx_i32m2(v1042, v1040, v1030, 16);
        int32_t* v1044 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1044, v1043, 16);
        const uint8_t* v1045 = v32 + 1124;
        const int16_t* v1046 = (const int16_t*) v1045;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1047 = *(const int16_t *)(v1046);
        int32_t* v1048 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1049 = __riscv_vle32_v_i32m2(v1048, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1050 = __riscv_vwmacc_vx_i32m2(v1049, v1047, v1030, 16);
        int32_t* v1051 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1051, v1050, 16);
        const uint8_t* v1052 = v32 + 1126;
        const int16_t* v1053 = (const int16_t*) v1052;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1054 = *(const int16_t *)(v1053);
        int32_t* v1055 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1056 = __riscv_vle32_v_i32m2(v1055, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1057 = __riscv_vwmacc_vx_i32m2(v1056, v1054, v1030, 16);
        int32_t* v1058 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1058, v1057, 16);
        int16_t* v1059 = &v45[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1060 = __riscv_vle16_v_i16m1(v1059, 16);
        const uint8_t* v1061 = v32 + 1128;
        const int16_t* v1062 = (const int16_t*) v1061;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1063 = *(const int16_t *)(v1062);
        int32_t* v1064 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1065 = __riscv_vle32_v_i32m2(v1064, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1066 = __riscv_vwmacc_vx_i32m2(v1065, v1063, v1060, 16);
        int32_t* v1067 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1067, v1066, 16);
        const uint8_t* v1068 = v32 + 1130;
        const int16_t* v1069 = (const int16_t*) v1068;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1070 = *(const int16_t *)(v1069);
        int32_t* v1071 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1072 = __riscv_vle32_v_i32m2(v1071, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1073 = __riscv_vwmacc_vx_i32m2(v1072, v1070, v1060, 16);
        int32_t* v1074 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1074, v1073, 16);
        const uint8_t* v1075 = v32 + 1132;
        const int16_t* v1076 = (const int16_t*) v1075;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1077 = *(const int16_t *)(v1076);
        int32_t* v1078 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1079 = __riscv_vle32_v_i32m2(v1078, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1080 = __riscv_vwmacc_vx_i32m2(v1079, v1077, v1060, 16);
        int32_t* v1081 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1081, v1080, 16);
        const uint8_t* v1082 = v32 + 1134;
        const int16_t* v1083 = (const int16_t*) v1082;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1084 = *(const int16_t *)(v1083);
        int32_t* v1085 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1086 = __riscv_vle32_v_i32m2(v1085, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1087 = __riscv_vwmacc_vx_i32m2(v1086, v1084, v1060, 16);
        int32_t* v1088 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1088, v1087, 16);
        int16_t* v1089 = &v45[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1090 = __riscv_vle16_v_i16m1(v1089, 16);
        const uint8_t* v1091 = v32 + 1136;
        const int16_t* v1092 = (const int16_t*) v1091;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1093 = *(const int16_t *)(v1092);
        int32_t* v1094 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1095 = __riscv_vle32_v_i32m2(v1094, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1096 = __riscv_vwmacc_vx_i32m2(v1095, v1093, v1090, 16);
        int32_t* v1097 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1097, v1096, 16);
        const uint8_t* v1098 = v32 + 1138;
        const int16_t* v1099 = (const int16_t*) v1098;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1100 = *(const int16_t *)(v1099);
        int32_t* v1101 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1102 = __riscv_vle32_v_i32m2(v1101, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1103 = __riscv_vwmacc_vx_i32m2(v1102, v1100, v1090, 16);
        int32_t* v1104 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1104, v1103, 16);
        const uint8_t* v1105 = v32 + 1140;
        const int16_t* v1106 = (const int16_t*) v1105;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1107 = *(const int16_t *)(v1106);
        int32_t* v1108 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1109 = __riscv_vle32_v_i32m2(v1108, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1110 = __riscv_vwmacc_vx_i32m2(v1109, v1107, v1090, 16);
        int32_t* v1111 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1111, v1110, 16);
        const uint8_t* v1112 = v32 + 1142;
        const int16_t* v1113 = (const int16_t*) v1112;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1114 = *(const int16_t *)(v1113);
        int32_t* v1115 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1116 = __riscv_vle32_v_i32m2(v1115, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1117 = __riscv_vwmacc_vx_i32m2(v1116, v1114, v1090, 16);
        int32_t* v1118 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1118, v1117, 16);
        int16_t* v1119 = &v45[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1120 = __riscv_vle16_v_i16m1(v1119, 16);
        const uint8_t* v1121 = v32 + 1144;
        const int16_t* v1122 = (const int16_t*) v1121;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1123 = *(const int16_t *)(v1122);
        int32_t* v1124 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1125 = __riscv_vle32_v_i32m2(v1124, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1126 = __riscv_vwmacc_vx_i32m2(v1125, v1123, v1120, 16);
        int32_t* v1127 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1127, v1126, 16);
        const uint8_t* v1128 = v32 + 1146;
        const int16_t* v1129 = (const int16_t*) v1128;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1130 = *(const int16_t *)(v1129);
        int32_t* v1131 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1132 = __riscv_vle32_v_i32m2(v1131, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1133 = __riscv_vwmacc_vx_i32m2(v1132, v1130, v1120, 16);
        int32_t* v1134 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1134, v1133, 16);
        const uint8_t* v1135 = v32 + 1148;
        const int16_t* v1136 = (const int16_t*) v1135;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1137 = *(const int16_t *)(v1136);
        int32_t* v1138 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1139 = __riscv_vle32_v_i32m2(v1138, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1140 = __riscv_vwmacc_vx_i32m2(v1139, v1137, v1120, 16);
        int32_t* v1141 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1141, v1140, 16);
        const uint8_t* v1142 = v32 + 1150;
        const int16_t* v1143 = (const int16_t*) v1142;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1144 = *(const int16_t *)(v1143);
        int32_t* v1145 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1146 = __riscv_vle32_v_i32m2(v1145, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1147 = __riscv_vwmacc_vx_i32m2(v1146, v1144, v1120, 16);
        int32_t* v1148 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1148, v1147, 16);
        int16_t* v1149 = &v45[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1150 = __riscv_vle16_v_i16m1(v1149, 16);
        const uint8_t* v1151 = v32 + 1152;
        const int16_t* v1152 = (const int16_t*) v1151;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1153 = *(const int16_t *)(v1152);
        int32_t* v1154 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1155 = __riscv_vle32_v_i32m2(v1154, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1156 = __riscv_vwmacc_vx_i32m2(v1155, v1153, v1150, 16);
        int32_t* v1157 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1157, v1156, 16);
        const uint8_t* v1158 = v32 + 1154;
        const int16_t* v1159 = (const int16_t*) v1158;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1160 = *(const int16_t *)(v1159);
        int32_t* v1161 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1162 = __riscv_vle32_v_i32m2(v1161, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1163 = __riscv_vwmacc_vx_i32m2(v1162, v1160, v1150, 16);
        int32_t* v1164 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1164, v1163, 16);
        const uint8_t* v1165 = v32 + 1156;
        const int16_t* v1166 = (const int16_t*) v1165;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1167 = *(const int16_t *)(v1166);
        int32_t* v1168 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1169 = __riscv_vle32_v_i32m2(v1168, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1170 = __riscv_vwmacc_vx_i32m2(v1169, v1167, v1150, 16);
        int32_t* v1171 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1171, v1170, 16);
        const uint8_t* v1172 = v32 + 1158;
        const int16_t* v1173 = (const int16_t*) v1172;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1174 = *(const int16_t *)(v1173);
        int32_t* v1175 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1176 = __riscv_vle32_v_i32m2(v1175, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1177 = __riscv_vwmacc_vx_i32m2(v1176, v1174, v1150, 16);
        int32_t* v1178 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1178, v1177, 16);
        int16_t* v1179 = &v45[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1180 = __riscv_vle16_v_i16m1(v1179, 16);
        const uint8_t* v1181 = v32 + 1160;
        const int16_t* v1182 = (const int16_t*) v1181;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1183 = *(const int16_t *)(v1182);
        int32_t* v1184 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1185 = __riscv_vle32_v_i32m2(v1184, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1186 = __riscv_vwmacc_vx_i32m2(v1185, v1183, v1180, 16);
        int32_t* v1187 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1187, v1186, 16);
        const uint8_t* v1188 = v32 + 1162;
        const int16_t* v1189 = (const int16_t*) v1188;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1190 = *(const int16_t *)(v1189);
        int32_t* v1191 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1192 = __riscv_vle32_v_i32m2(v1191, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1193 = __riscv_vwmacc_vx_i32m2(v1192, v1190, v1180, 16);
        int32_t* v1194 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1194, v1193, 16);
        const uint8_t* v1195 = v32 + 1164;
        const int16_t* v1196 = (const int16_t*) v1195;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1197 = *(const int16_t *)(v1196);
        int32_t* v1198 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1199 = __riscv_vle32_v_i32m2(v1198, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1200 = __riscv_vwmacc_vx_i32m2(v1199, v1197, v1180, 16);
        int32_t* v1201 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1201, v1200, 16);
        const uint8_t* v1202 = v32 + 1166;
        const int16_t* v1203 = (const int16_t*) v1202;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
        int32_t v1204 = *(const int16_t *)(v1203);
        int32_t* v1205 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1206 = __riscv_vle32_v_i32m2(v1205, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
        vint32m2_t v1207 = __riscv_vwmacc_vx_i32m2(v1206, v1204, v1180, 16);
        int32_t* v1208 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
        __riscv_vse32_v_i32m2(v1208, v1207, 16);
        int16_t v1209[256];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1210 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1211 = &v1209[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1211, v1210, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1212 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1213 = &v1209[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1213, v1212, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1214 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1215 = &v1209[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1215, v1214, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1216 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1217 = &v1209[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1217, v1216, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1218 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1219 = &v1209[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1219, v1218, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1220 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1221 = &v1209[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1221, v1220, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1222 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1223 = &v1209[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1223, v1222, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1224 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1225 = &v1209[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1225, v1224, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1226 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1227 = &v1209[128];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1227, v1226, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1228 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1229 = &v1209[144];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1229, v1228, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1230 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1231 = &v1209[160];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1231, v1230, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1232 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1233 = &v1209[176];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1233, v1232, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1234 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1235 = &v1209[192];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1235, v1234, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1236 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1237 = &v1209[208];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1237, v1236, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1238 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1239 = &v1209[224];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1239, v1238, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1240 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1241 = &v1209[240];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1241, v1240, 16);
        for (size_t v1242 = 0; v1242 < 16; v1242 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
          size_t v1243 = v1242 * 16;
          size_t v1244 = 832 + v1243;
          const uint8_t* v1245 = v30 + v1244;
          const uint8_t* v1246 = (const uint8_t*) v1245;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1247 = __riscv_vle8_v_u8mf2(v1246, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1248 = __riscv_vand_vx_u8mf2(v1247, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1249 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1248);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1250 = __riscv_vsrl_vx_u8mf2(v1247, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1251 = __riscv_vand_vx_u8mf2(v1250, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1252 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1251);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1253 = __riscv_vsrl_vx_u8mf2(v1247, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1254 = __riscv_vand_vx_u8mf2(v1253, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1255 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1254);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1256 = __riscv_vsrl_vx_u8mf2(v1247, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1257 = __riscv_vand_vx_u8mf2(v1256, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1258 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1257);
          size_t v1259 = v1242 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1260 = 528 + v1259;
          const uint8_t* v1261 = v32 + v1260;
          const int8_t* v1262 = (const int8_t*) v1261;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1263 = *(const int8_t *)(v1262);
          int16_t* v1264 = &v1209[0];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1265 = __riscv_vle16_v_i16m1(v1264, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1266 = __riscv_vwmacc_vx_i16m1(v1265, v1263, v1249, 16);
          int16_t* v1267 = &v1209[0];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1267, v1266, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1268 = 656 + v1259;
          const uint8_t* v1269 = v32 + v1268;
          const int8_t* v1270 = (const int8_t*) v1269;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1271 = *(const int8_t *)(v1270);
          int16_t* v1272 = &v1209[16];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1273 = __riscv_vle16_v_i16m1(v1272, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1274 = __riscv_vwmacc_vx_i16m1(v1273, v1271, v1252, 16);
          int16_t* v1275 = &v1209[16];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1275, v1274, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1276 = 784 + v1259;
          const uint8_t* v1277 = v32 + v1276;
          const int8_t* v1278 = (const int8_t*) v1277;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1279 = *(const int8_t *)(v1278);
          int16_t* v1280 = &v1209[32];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1281 = __riscv_vle16_v_i16m1(v1280, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1282 = __riscv_vwmacc_vx_i16m1(v1281, v1279, v1255, 16);
          int16_t* v1283 = &v1209[32];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1283, v1282, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1284 = 912 + v1259;
          const uint8_t* v1285 = v32 + v1284;
          const int8_t* v1286 = (const int8_t*) v1285;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1287 = *(const int8_t *)(v1286);
          int16_t* v1288 = &v1209[48];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1289 = __riscv_vle16_v_i16m1(v1288, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1290 = __riscv_vwmacc_vx_i16m1(v1289, v1287, v1258, 16);
          int16_t* v1291 = &v1209[48];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1291, v1290, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1292 = 529 + v1259;
          const uint8_t* v1293 = v32 + v1292;
          const int8_t* v1294 = (const int8_t*) v1293;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1295 = *(const int8_t *)(v1294);
          int16_t* v1296 = &v1209[64];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1297 = __riscv_vle16_v_i16m1(v1296, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1298 = __riscv_vwmacc_vx_i16m1(v1297, v1295, v1249, 16);
          int16_t* v1299 = &v1209[64];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1299, v1298, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1300 = 657 + v1259;
          const uint8_t* v1301 = v32 + v1300;
          const int8_t* v1302 = (const int8_t*) v1301;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1303 = *(const int8_t *)(v1302);
          int16_t* v1304 = &v1209[80];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1305 = __riscv_vle16_v_i16m1(v1304, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1306 = __riscv_vwmacc_vx_i16m1(v1305, v1303, v1252, 16);
          int16_t* v1307 = &v1209[80];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1307, v1306, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1308 = 785 + v1259;
          const uint8_t* v1309 = v32 + v1308;
          const int8_t* v1310 = (const int8_t*) v1309;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1311 = *(const int8_t *)(v1310);
          int16_t* v1312 = &v1209[96];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1313 = __riscv_vle16_v_i16m1(v1312, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1314 = __riscv_vwmacc_vx_i16m1(v1313, v1311, v1255, 16);
          int16_t* v1315 = &v1209[96];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1315, v1314, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1316 = 913 + v1259;
          const uint8_t* v1317 = v32 + v1316;
          const int8_t* v1318 = (const int8_t*) v1317;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1319 = *(const int8_t *)(v1318);
          int16_t* v1320 = &v1209[112];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1321 = __riscv_vle16_v_i16m1(v1320, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1322 = __riscv_vwmacc_vx_i16m1(v1321, v1319, v1258, 16);
          int16_t* v1323 = &v1209[112];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1323, v1322, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1324 = 530 + v1259;
          const uint8_t* v1325 = v32 + v1324;
          const int8_t* v1326 = (const int8_t*) v1325;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1327 = *(const int8_t *)(v1326);
          int16_t* v1328 = &v1209[128];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1329 = __riscv_vle16_v_i16m1(v1328, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1330 = __riscv_vwmacc_vx_i16m1(v1329, v1327, v1249, 16);
          int16_t* v1331 = &v1209[128];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1331, v1330, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1332 = 658 + v1259;
          const uint8_t* v1333 = v32 + v1332;
          const int8_t* v1334 = (const int8_t*) v1333;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1335 = *(const int8_t *)(v1334);
          int16_t* v1336 = &v1209[144];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1337 = __riscv_vle16_v_i16m1(v1336, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1338 = __riscv_vwmacc_vx_i16m1(v1337, v1335, v1252, 16);
          int16_t* v1339 = &v1209[144];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1339, v1338, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1340 = 786 + v1259;
          const uint8_t* v1341 = v32 + v1340;
          const int8_t* v1342 = (const int8_t*) v1341;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1343 = *(const int8_t *)(v1342);
          int16_t* v1344 = &v1209[160];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1345 = __riscv_vle16_v_i16m1(v1344, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1346 = __riscv_vwmacc_vx_i16m1(v1345, v1343, v1255, 16);
          int16_t* v1347 = &v1209[160];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1347, v1346, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1348 = 914 + v1259;
          const uint8_t* v1349 = v32 + v1348;
          const int8_t* v1350 = (const int8_t*) v1349;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1351 = *(const int8_t *)(v1350);
          int16_t* v1352 = &v1209[176];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1353 = __riscv_vle16_v_i16m1(v1352, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1354 = __riscv_vwmacc_vx_i16m1(v1353, v1351, v1258, 16);
          int16_t* v1355 = &v1209[176];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1355, v1354, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1356 = 531 + v1259;
          const uint8_t* v1357 = v32 + v1356;
          const int8_t* v1358 = (const int8_t*) v1357;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1359 = *(const int8_t *)(v1358);
          int16_t* v1360 = &v1209[192];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1361 = __riscv_vle16_v_i16m1(v1360, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1362 = __riscv_vwmacc_vx_i16m1(v1361, v1359, v1249, 16);
          int16_t* v1363 = &v1209[192];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1363, v1362, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1364 = 659 + v1259;
          const uint8_t* v1365 = v32 + v1364;
          const int8_t* v1366 = (const int8_t*) v1365;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1367 = *(const int8_t *)(v1366);
          int16_t* v1368 = &v1209[208];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1369 = __riscv_vle16_v_i16m1(v1368, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1370 = __riscv_vwmacc_vx_i16m1(v1369, v1367, v1252, 16);
          int16_t* v1371 = &v1209[208];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1371, v1370, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1372 = 787 + v1259;
          const uint8_t* v1373 = v32 + v1372;
          const int8_t* v1374 = (const int8_t*) v1373;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1375 = *(const int8_t *)(v1374);
          int16_t* v1376 = &v1209[224];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1377 = __riscv_vle16_v_i16m1(v1376, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1378 = __riscv_vwmacc_vx_i16m1(v1377, v1375, v1255, 16);
          int16_t* v1379 = &v1209[224];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1379, v1378, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1380 = 915 + v1259;
          const uint8_t* v1381 = v32 + v1380;
          const int8_t* v1382 = (const int8_t*) v1381;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1383 = *(const int8_t *)(v1382);
          int16_t* v1384 = &v1209[240];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1385 = __riscv_vle16_v_i16m1(v1384, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1386 = __riscv_vwmacc_vx_i16m1(v1385, v1383, v1258, 16);
          int16_t* v1387 = &v1209[240];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1387, v1386, 16);
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v1388 = &v44[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1389 = __riscv_vle16_v_i16m1(v1388, 16);
        int16_t* v1390 = &v1209[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1391 = __riscv_vle16_v_i16m1(v1390, 16);
        vint32m2_t v1392 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1393 = __riscv_vwmacc_vv_i32m2(v1392, v1389, v1391, 16);
        v47 = v1393;
        int16_t* v1394 = &v1209[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1395 = __riscv_vle16_v_i16m1(v1394, 16);
        vint32m2_t v1396 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1397 = __riscv_vwmacc_vv_i32m2(v1396, v1389, v1395, 16);
        v49 = v1397;
        int16_t* v1398 = &v1209[128];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1399 = __riscv_vle16_v_i16m1(v1398, 16);
        vint32m2_t v1400 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1401 = __riscv_vwmacc_vv_i32m2(v1400, v1389, v1399, 16);
        v51 = v1401;
        int16_t* v1402 = &v1209[192];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1403 = __riscv_vle16_v_i16m1(v1402, 16);
        vint32m2_t v1404 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1405 = __riscv_vwmacc_vv_i32m2(v1404, v1389, v1403, 16);
        v53 = v1405;
        int16_t* v1406 = &v44[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1407 = __riscv_vle16_v_i16m1(v1406, 16);
        int16_t* v1408 = &v1209[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1409 = __riscv_vle16_v_i16m1(v1408, 16);
        vint32m2_t v1410 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1411 = __riscv_vwmacc_vv_i32m2(v1410, v1407, v1409, 16);
        v47 = v1411;
        int16_t* v1412 = &v1209[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1413 = __riscv_vle16_v_i16m1(v1412, 16);
        vint32m2_t v1414 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1415 = __riscv_vwmacc_vv_i32m2(v1414, v1407, v1413, 16);
        v49 = v1415;
        int16_t* v1416 = &v1209[144];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1417 = __riscv_vle16_v_i16m1(v1416, 16);
        vint32m2_t v1418 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1419 = __riscv_vwmacc_vv_i32m2(v1418, v1407, v1417, 16);
        v51 = v1419;
        int16_t* v1420 = &v1209[208];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1421 = __riscv_vle16_v_i16m1(v1420, 16);
        vint32m2_t v1422 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1423 = __riscv_vwmacc_vv_i32m2(v1422, v1407, v1421, 16);
        v53 = v1423;
        int16_t* v1424 = &v44[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1425 = __riscv_vle16_v_i16m1(v1424, 16);
        int16_t* v1426 = &v1209[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1427 = __riscv_vle16_v_i16m1(v1426, 16);
        vint32m2_t v1428 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1429 = __riscv_vwmacc_vv_i32m2(v1428, v1425, v1427, 16);
        v47 = v1429;
        int16_t* v1430 = &v1209[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1431 = __riscv_vle16_v_i16m1(v1430, 16);
        vint32m2_t v1432 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1433 = __riscv_vwmacc_vv_i32m2(v1432, v1425, v1431, 16);
        v49 = v1433;
        int16_t* v1434 = &v1209[160];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1435 = __riscv_vle16_v_i16m1(v1434, 16);
        vint32m2_t v1436 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1437 = __riscv_vwmacc_vv_i32m2(v1436, v1425, v1435, 16);
        v51 = v1437;
        int16_t* v1438 = &v1209[224];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1439 = __riscv_vle16_v_i16m1(v1438, 16);
        vint32m2_t v1440 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1441 = __riscv_vwmacc_vv_i32m2(v1440, v1425, v1439, 16);
        v53 = v1441;
        int16_t* v1442 = &v44[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1443 = __riscv_vle16_v_i16m1(v1442, 16);
        int16_t* v1444 = &v1209[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1445 = __riscv_vle16_v_i16m1(v1444, 16);
        vint32m2_t v1446 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1447 = __riscv_vwmacc_vv_i32m2(v1446, v1443, v1445, 16);
        v47 = v1447;
        int16_t* v1448 = &v1209[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1449 = __riscv_vle16_v_i16m1(v1448, 16);
        vint32m2_t v1450 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1451 = __riscv_vwmacc_vv_i32m2(v1450, v1443, v1449, 16);
        v49 = v1451;
        int16_t* v1452 = &v1209[176];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1453 = __riscv_vle16_v_i16m1(v1452, 16);
        vint32m2_t v1454 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1455 = __riscv_vwmacc_vv_i32m2(v1454, v1443, v1453, 16);
        v51 = v1455;
        int16_t* v1456 = &v1209[240];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1457 = __riscv_vle16_v_i16m1(v1456, 16);
        vint32m2_t v1458 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1459 = __riscv_vwmacc_vv_i32m2(v1458, v1443, v1457, 16);
        v53 = v1459;
        int16_t v1460[256];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1461 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1462 = &v1460[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1462, v1461, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1463 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1464 = &v1460[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1464, v1463, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1465 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1466 = &v1460[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1466, v1465, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1467 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1468 = &v1460[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1468, v1467, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1469 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1470 = &v1460[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1470, v1469, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1471 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1472 = &v1460[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1472, v1471, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1473 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1474 = &v1460[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1474, v1473, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1475 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1476 = &v1460[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1476, v1475, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1477 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1478 = &v1460[128];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1478, v1477, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1479 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1480 = &v1460[144];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1480, v1479, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1481 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1482 = &v1460[160];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1482, v1481, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1483 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1484 = &v1460[176];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1484, v1483, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1485 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1486 = &v1460[192];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1486, v1485, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1487 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1488 = &v1460[208];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1488, v1487, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1489 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1490 = &v1460[224];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1490, v1489, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
        vint16m1_t v1491 = __riscv_vmv_v_x_i16m1(0, 16);
        int16_t* v1492 = &v1460[240];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
        __riscv_vse16_v_i16m1(v1492, v1491, 16);
        for (size_t v1493 = 0; v1493 < 16; v1493 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
          size_t v1494 = v1493 * 16;
          size_t v1495 = 1088 + v1494;
          const uint8_t* v1496 = v30 + v1495;
          const uint8_t* v1497 = (const uint8_t*) v1496;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
          vuint8mf2_t v1498 = __riscv_vle8_v_u8mf2(v1497, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1499 = __riscv_vand_vx_u8mf2(v1498, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1500 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1499);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1501 = __riscv_vsrl_vx_u8mf2(v1498, 2, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1502 = __riscv_vand_vx_u8mf2(v1501, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1503 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1502);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1504 = __riscv_vsrl_vx_u8mf2(v1498, 4, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1505 = __riscv_vand_vx_u8mf2(v1504, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1506 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1505);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
          vuint8mf2_t v1507 = __riscv_vsrl_vx_u8mf2(v1498, 6, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
          vuint8mf2_t v1508 = __riscv_vand_vx_u8mf2(v1507, 0x03, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
          vint8mf2_t v1509 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1508);
          size_t v1510 = v1493 * 4;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1511 = 592 + v1510;
          const uint8_t* v1512 = v32 + v1511;
          const int8_t* v1513 = (const int8_t*) v1512;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1514 = *(const int8_t *)(v1513);
          int16_t* v1515 = &v1460[0];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1516 = __riscv_vle16_v_i16m1(v1515, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1517 = __riscv_vwmacc_vx_i16m1(v1516, v1514, v1500, 16);
          int16_t* v1518 = &v1460[0];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1518, v1517, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1519 = 720 + v1510;
          const uint8_t* v1520 = v32 + v1519;
          const int8_t* v1521 = (const int8_t*) v1520;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1522 = *(const int8_t *)(v1521);
          int16_t* v1523 = &v1460[16];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1524 = __riscv_vle16_v_i16m1(v1523, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1525 = __riscv_vwmacc_vx_i16m1(v1524, v1522, v1503, 16);
          int16_t* v1526 = &v1460[16];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1526, v1525, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1527 = 848 + v1510;
          const uint8_t* v1528 = v32 + v1527;
          const int8_t* v1529 = (const int8_t*) v1528;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1530 = *(const int8_t *)(v1529);
          int16_t* v1531 = &v1460[32];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1532 = __riscv_vle16_v_i16m1(v1531, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1533 = __riscv_vwmacc_vx_i16m1(v1532, v1530, v1506, 16);
          int16_t* v1534 = &v1460[32];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1534, v1533, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1535 = 976 + v1510;
          const uint8_t* v1536 = v32 + v1535;
          const int8_t* v1537 = (const int8_t*) v1536;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1538 = *(const int8_t *)(v1537);
          int16_t* v1539 = &v1460[48];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1540 = __riscv_vle16_v_i16m1(v1539, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1541 = __riscv_vwmacc_vx_i16m1(v1540, v1538, v1509, 16);
          int16_t* v1542 = &v1460[48];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1542, v1541, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1543 = 593 + v1510;
          const uint8_t* v1544 = v32 + v1543;
          const int8_t* v1545 = (const int8_t*) v1544;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1546 = *(const int8_t *)(v1545);
          int16_t* v1547 = &v1460[64];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1548 = __riscv_vle16_v_i16m1(v1547, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1549 = __riscv_vwmacc_vx_i16m1(v1548, v1546, v1500, 16);
          int16_t* v1550 = &v1460[64];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1550, v1549, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1551 = 721 + v1510;
          const uint8_t* v1552 = v32 + v1551;
          const int8_t* v1553 = (const int8_t*) v1552;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1554 = *(const int8_t *)(v1553);
          int16_t* v1555 = &v1460[80];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1556 = __riscv_vle16_v_i16m1(v1555, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1557 = __riscv_vwmacc_vx_i16m1(v1556, v1554, v1503, 16);
          int16_t* v1558 = &v1460[80];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1558, v1557, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1559 = 849 + v1510;
          const uint8_t* v1560 = v32 + v1559;
          const int8_t* v1561 = (const int8_t*) v1560;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1562 = *(const int8_t *)(v1561);
          int16_t* v1563 = &v1460[96];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1564 = __riscv_vle16_v_i16m1(v1563, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1565 = __riscv_vwmacc_vx_i16m1(v1564, v1562, v1506, 16);
          int16_t* v1566 = &v1460[96];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1566, v1565, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1567 = 977 + v1510;
          const uint8_t* v1568 = v32 + v1567;
          const int8_t* v1569 = (const int8_t*) v1568;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1570 = *(const int8_t *)(v1569);
          int16_t* v1571 = &v1460[112];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1572 = __riscv_vle16_v_i16m1(v1571, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1573 = __riscv_vwmacc_vx_i16m1(v1572, v1570, v1509, 16);
          int16_t* v1574 = &v1460[112];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1574, v1573, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1575 = 594 + v1510;
          const uint8_t* v1576 = v32 + v1575;
          const int8_t* v1577 = (const int8_t*) v1576;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1578 = *(const int8_t *)(v1577);
          int16_t* v1579 = &v1460[128];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1580 = __riscv_vle16_v_i16m1(v1579, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1581 = __riscv_vwmacc_vx_i16m1(v1580, v1578, v1500, 16);
          int16_t* v1582 = &v1460[128];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1582, v1581, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1583 = 722 + v1510;
          const uint8_t* v1584 = v32 + v1583;
          const int8_t* v1585 = (const int8_t*) v1584;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1586 = *(const int8_t *)(v1585);
          int16_t* v1587 = &v1460[144];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1588 = __riscv_vle16_v_i16m1(v1587, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1589 = __riscv_vwmacc_vx_i16m1(v1588, v1586, v1503, 16);
          int16_t* v1590 = &v1460[144];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1590, v1589, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1591 = 850 + v1510;
          const uint8_t* v1592 = v32 + v1591;
          const int8_t* v1593 = (const int8_t*) v1592;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1594 = *(const int8_t *)(v1593);
          int16_t* v1595 = &v1460[160];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1596 = __riscv_vle16_v_i16m1(v1595, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1597 = __riscv_vwmacc_vx_i16m1(v1596, v1594, v1506, 16);
          int16_t* v1598 = &v1460[160];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1598, v1597, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1599 = 978 + v1510;
          const uint8_t* v1600 = v32 + v1599;
          const int8_t* v1601 = (const int8_t*) v1600;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1602 = *(const int8_t *)(v1601);
          int16_t* v1603 = &v1460[176];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1604 = __riscv_vle16_v_i16m1(v1603, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1605 = __riscv_vwmacc_vx_i16m1(v1604, v1602, v1509, 16);
          int16_t* v1606 = &v1460[176];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1606, v1605, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1607 = 595 + v1510;
          const uint8_t* v1608 = v32 + v1607;
          const int8_t* v1609 = (const int8_t*) v1608;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1610 = *(const int8_t *)(v1609);
          int16_t* v1611 = &v1460[192];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1612 = __riscv_vle16_v_i16m1(v1611, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1613 = __riscv_vwmacc_vx_i16m1(v1612, v1610, v1500, 16);
          int16_t* v1614 = &v1460[192];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1614, v1613, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1615 = 723 + v1510;
          const uint8_t* v1616 = v32 + v1615;
          const int8_t* v1617 = (const int8_t*) v1616;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1618 = *(const int8_t *)(v1617);
          int16_t* v1619 = &v1460[208];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1620 = __riscv_vle16_v_i16m1(v1619, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1621 = __riscv_vwmacc_vx_i16m1(v1620, v1618, v1503, 16);
          int16_t* v1622 = &v1460[208];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1622, v1621, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1623 = 851 + v1510;
          const uint8_t* v1624 = v32 + v1623;
          const int8_t* v1625 = (const int8_t*) v1624;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1626 = *(const int8_t *)(v1625);
          int16_t* v1627 = &v1460[224];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1628 = __riscv_vle16_v_i16m1(v1627, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1629 = __riscv_vwmacc_vx_i16m1(v1628, v1626, v1506, 16);
          int16_t* v1630 = &v1460[224];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1630, v1629, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
          size_t v1631 = 979 + v1510;
          const uint8_t* v1632 = v32 + v1631;
          const int8_t* v1633 = (const int8_t*) v1632;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
          int32_t v1634 = *(const int8_t *)(v1633);
          int16_t* v1635 = &v1460[240];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
          vint16m1_t v1636 = __riscv_vle16_v_i16m1(v1635, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
          vint16m1_t v1637 = __riscv_vwmacc_vx_i16m1(v1636, v1634, v1509, 16);
          int16_t* v1638 = &v1460[240];
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse16_v_i16m1
          __riscv_vse16_v_i16m1(v1638, v1637, 16);
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
        int16_t* v1639 = &v44[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1640 = __riscv_vle16_v_i16m1(v1639, 16);
        int16_t* v1641 = &v1460[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1642 = __riscv_vle16_v_i16m1(v1641, 16);
        vint32m2_t v1643 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1644 = __riscv_vwmacc_vv_i32m2(v1643, v1640, v1642, 16);
        v47 = v1644;
        int16_t* v1645 = &v1460[64];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1646 = __riscv_vle16_v_i16m1(v1645, 16);
        vint32m2_t v1647 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1648 = __riscv_vwmacc_vv_i32m2(v1647, v1640, v1646, 16);
        v49 = v1648;
        int16_t* v1649 = &v1460[128];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1650 = __riscv_vle16_v_i16m1(v1649, 16);
        vint32m2_t v1651 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1652 = __riscv_vwmacc_vv_i32m2(v1651, v1640, v1650, 16);
        v51 = v1652;
        int16_t* v1653 = &v1460[192];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1654 = __riscv_vle16_v_i16m1(v1653, 16);
        vint32m2_t v1655 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1656 = __riscv_vwmacc_vv_i32m2(v1655, v1640, v1654, 16);
        v53 = v1656;
        int16_t* v1657 = &v44[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1658 = __riscv_vle16_v_i16m1(v1657, 16);
        int16_t* v1659 = &v1460[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1660 = __riscv_vle16_v_i16m1(v1659, 16);
        vint32m2_t v1661 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1662 = __riscv_vwmacc_vv_i32m2(v1661, v1658, v1660, 16);
        v47 = v1662;
        int16_t* v1663 = &v1460[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1664 = __riscv_vle16_v_i16m1(v1663, 16);
        vint32m2_t v1665 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1666 = __riscv_vwmacc_vv_i32m2(v1665, v1658, v1664, 16);
        v49 = v1666;
        int16_t* v1667 = &v1460[144];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1668 = __riscv_vle16_v_i16m1(v1667, 16);
        vint32m2_t v1669 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1670 = __riscv_vwmacc_vv_i32m2(v1669, v1658, v1668, 16);
        v51 = v1670;
        int16_t* v1671 = &v1460[208];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1672 = __riscv_vle16_v_i16m1(v1671, 16);
        vint32m2_t v1673 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1674 = __riscv_vwmacc_vv_i32m2(v1673, v1658, v1672, 16);
        v53 = v1674;
        int16_t* v1675 = &v44[80];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1676 = __riscv_vle16_v_i16m1(v1675, 16);
        int16_t* v1677 = &v1460[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1678 = __riscv_vle16_v_i16m1(v1677, 16);
        vint32m2_t v1679 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1680 = __riscv_vwmacc_vv_i32m2(v1679, v1676, v1678, 16);
        v47 = v1680;
        int16_t* v1681 = &v1460[96];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1682 = __riscv_vle16_v_i16m1(v1681, 16);
        vint32m2_t v1683 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1684 = __riscv_vwmacc_vv_i32m2(v1683, v1676, v1682, 16);
        v49 = v1684;
        int16_t* v1685 = &v1460[160];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1686 = __riscv_vle16_v_i16m1(v1685, 16);
        vint32m2_t v1687 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1688 = __riscv_vwmacc_vv_i32m2(v1687, v1676, v1686, 16);
        v51 = v1688;
        int16_t* v1689 = &v1460[224];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1690 = __riscv_vle16_v_i16m1(v1689, 16);
        vint32m2_t v1691 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1692 = __riscv_vwmacc_vv_i32m2(v1691, v1676, v1690, 16);
        v53 = v1692;
        int16_t* v1693 = &v44[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1694 = __riscv_vle16_v_i16m1(v1693, 16);
        int16_t* v1695 = &v1460[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1696 = __riscv_vle16_v_i16m1(v1695, 16);
        vint32m2_t v1697 = v47;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1698 = __riscv_vwmacc_vv_i32m2(v1697, v1694, v1696, 16);
        v47 = v1698;
        int16_t* v1699 = &v1460[112];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1700 = __riscv_vle16_v_i16m1(v1699, 16);
        vint32m2_t v1701 = v49;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1702 = __riscv_vwmacc_vv_i32m2(v1701, v1694, v1700, 16);
        v49 = v1702;
        int16_t* v1703 = &v1460[176];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1704 = __riscv_vle16_v_i16m1(v1703, 16);
        vint32m2_t v1705 = v51;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1706 = __riscv_vwmacc_vv_i32m2(v1705, v1694, v1704, 16);
        v51 = v1706;
        int16_t* v1707 = &v1460[240];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_i16m1
        vint16m1_t v1708 = __riscv_vle16_v_i16m1(v1707, 16);
        vint32m2_t v1709 = v53;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
        vint32m2_t v1710 = __riscv_vwmacc_vv_i32m2(v1709, v1694, v1708, 16);
        v53 = v1710;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const uint8_t* v1711 = v30 + 32;
        const _Float16* v1712 = (const _Float16*) v1711;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v1713 = __riscv_vle16_v_f16m1(v1712, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v1714 = __riscv_vfwcvt_f_f_v_f32m2(v1713, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
        const _Float16* v1715 = (const _Float16*) v30;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
        vfloat16m1_t v1716 = __riscv_vle16_v_f16m1(v1715, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
        vfloat32m2_t v1717 = __riscv_vfwcvt_f_f_v_f32m2(v1716, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1718 = __riscv_vfmul_vf_f32m2(v1717, v34, 16);
        vint32m2_t v1719 = v47;
        vfloat32m2_t v1720 = v20;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1721 = __riscv_vfcvt_f_x_v_f32m2(v1719, 16);
        vfloat32m2_t v1722 = __riscv_vfmacc_vv_f32m2(v1720, v1721, v1718, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1723 = __riscv_vfmul_vf_f32m2(v1714, v34, 16);
        int32_t* v1724 = &v46[0];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1725 = __riscv_vle32_v_i32m2(v1724, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1726 = __riscv_vfcvt_f_x_v_f32m2(v1725, 16);
        vfloat32m2_t v1727 = __riscv_vfnmsac_vv_f32m2(v1722, v1723, v1726, 16);
        v20 = v1727;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1728 = __riscv_vfmul_vf_f32m2(v1717, v37, 16);
        vint32m2_t v1729 = v49;
        vfloat32m2_t v1730 = v22;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1731 = __riscv_vfcvt_f_x_v_f32m2(v1729, 16);
        vfloat32m2_t v1732 = __riscv_vfmacc_vv_f32m2(v1730, v1731, v1728, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1733 = __riscv_vfmul_vf_f32m2(v1714, v37, 16);
        int32_t* v1734 = &v46[16];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1735 = __riscv_vle32_v_i32m2(v1734, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1736 = __riscv_vfcvt_f_x_v_f32m2(v1735, 16);
        vfloat32m2_t v1737 = __riscv_vfnmsac_vv_f32m2(v1732, v1733, v1736, 16);
        v22 = v1737;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1738 = __riscv_vfmul_vf_f32m2(v1717, v40, 16);
        vint32m2_t v1739 = v51;
        vfloat32m2_t v1740 = v24;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1741 = __riscv_vfcvt_f_x_v_f32m2(v1739, 16);
        vfloat32m2_t v1742 = __riscv_vfmacc_vv_f32m2(v1740, v1741, v1738, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1743 = __riscv_vfmul_vf_f32m2(v1714, v40, 16);
        int32_t* v1744 = &v46[32];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1745 = __riscv_vle32_v_i32m2(v1744, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1746 = __riscv_vfcvt_f_x_v_f32m2(v1745, 16);
        vfloat32m2_t v1747 = __riscv_vfnmsac_vv_f32m2(v1742, v1743, v1746, 16);
        v24 = v1747;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1748 = __riscv_vfmul_vf_f32m2(v1717, v43, 16);
        vint32m2_t v1749 = v53;
        vfloat32m2_t v1750 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1751 = __riscv_vfcvt_f_x_v_f32m2(v1749, 16);
        vfloat32m2_t v1752 = __riscv_vfmacc_vv_f32m2(v1750, v1751, v1748, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
        vfloat32m2_t v1753 = __riscv_vfmul_vf_f32m2(v1714, v43, 16);
        int32_t* v1754 = &v46[48];
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
        vint32m2_t v1755 = __riscv_vle32_v_i32m2(v1754, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
        vfloat32m2_t v1756 = __riscv_vfcvt_f_x_v_f32m2(v1755, 16);
        vfloat32m2_t v1757 = __riscv_vfnmsac_vv_f32m2(v1752, v1753, v1756, 16);
        v26 = v1757;
      }
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1758 = v12 * 4;
      size_t v1759 = v1758 + 0;
      size_t v1760 = v1759 * v7;
      size_t v1761 = v16 * 16;
      size_t v1762 = v1760 + v1761;
      float* v1763 = v2 + v1762;
      vfloat32m2_t v1764 = v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1763, v1764, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1765 = v12 * 4;
      size_t v1766 = v1765 + 1;
      size_t v1767 = v1766 * v7;
      size_t v1768 = v16 * 16;
      size_t v1769 = v1767 + v1768;
      float* v1770 = v2 + v1769;
      vfloat32m2_t v1771 = v22;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1770, v1771, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1772 = v12 * 4;
      size_t v1773 = v1772 + 2;
      size_t v1774 = v1773 * v7;
      size_t v1775 = v16 * 16;
      size_t v1776 = v1774 + v1775;
      float* v1777 = v2 + v1776;
      vfloat32m2_t v1778 = v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1777, v1778, 16);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
      size_t v1779 = v12 * 4;
      size_t v1780 = v1779 + 3;
      size_t v1781 = v1780 * v7;
      size_t v1782 = v16 * 16;
      size_t v1783 = v1781 + v1782;
      float* v1784 = v2 + v1783;
      vfloat32m2_t v1785 = v26;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
      __riscv_vse32_v_f32m2(v1784, v1785, 16);
    }
  }
  return;
}


