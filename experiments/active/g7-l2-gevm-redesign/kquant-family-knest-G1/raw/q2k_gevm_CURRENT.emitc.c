#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemv_q2_K_q8_K_kernel_ggml_repack_gemv_q2_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
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
    size_t v11 = v10 * 1344;
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
      size_t v18 = v17 * 1344;
      const uint8_t* v19 = v12 + v18;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
      size_t v20 = v17 * 292;
      const uint8_t* v21 = v4 + v20;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
      const float* v22 = (const float*) v21;
      float v23 = *(const float *)(v22);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v24 = v19 + 32;
      const _Float16* v25 = (const _Float16*) v24;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v26 = __riscv_vle16_v_f16m1(v25, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v27 = __riscv_vfwcvt_f_f_v_f32m2(v26, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v28 = __riscv_vfmul_vf_f32m2(v27, v23, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v29 = v19 + 48;
      const _Float16* v30 = (const _Float16*) v29;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v31 = __riscv_vle16_v_f16m1(v30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v32 = __riscv_vfwcvt_f_f_v_f32m2(v31, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v33 = __riscv_vfmul_vf_f32m2(v32, v23, 8);
      vint32m2_t v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v35 = __riscv_vmv_v_x_i32m2(0, 8);
      v34 = v35;
      vint32m2_t v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v37 = __riscv_vmv_v_x_i32m2(0, 8);
      v36 = v37;
      vint32m2_t v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v39 = __riscv_vmv_v_x_i32m2(0, 8);
      v38 = v39;
      vint32m2_t v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v41 = __riscv_vmv_v_x_i32m2(0, 8);
      v40 = v41;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
      const uint8_t* v42 = v19 + 64;
      const uint8_t* v43 = (const uint8_t*) v42;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v44 = __riscv_vle8_v_u8mf2(v43, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v45 = __riscv_vand_vx_u8mf2(v44, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v46 = __riscv_vsrl_vx_u8mf2(v44, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v47 = __riscv_vzext_vf2_u16m1(v45, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v48 = __riscv_vreinterpret_v_u16m1_i16m1(v47);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v49 = __riscv_vzext_vf2_u16m1(v46, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v50 = __riscv_vreinterpret_v_u16m1_i16m1(v49);
      const uint8_t* v51 = v19 + 80;
      const uint8_t* v52 = (const uint8_t*) v51;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v53 = __riscv_vle8_v_u8mf2(v52, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v54 = __riscv_vand_vx_u8mf2(v53, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v55 = __riscv_vsrl_vx_u8mf2(v53, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v56 = __riscv_vzext_vf2_u16m1(v54, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v57 = __riscv_vreinterpret_v_u16m1_i16m1(v56);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v58 = __riscv_vzext_vf2_u16m1(v55, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v59 = __riscv_vreinterpret_v_u16m1_i16m1(v58);
      const uint8_t* v60 = v19 + 96;
      const uint8_t* v61 = (const uint8_t*) v60;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v62 = __riscv_vle8_v_u8mf2(v61, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v63 = __riscv_vand_vx_u8mf2(v62, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v64 = __riscv_vsrl_vx_u8mf2(v62, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v65 = __riscv_vzext_vf2_u16m1(v63, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v66 = __riscv_vreinterpret_v_u16m1_i16m1(v65);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v67 = __riscv_vzext_vf2_u16m1(v64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v68 = __riscv_vreinterpret_v_u16m1_i16m1(v67);
      const uint8_t* v69 = v19 + 112;
      const uint8_t* v70 = (const uint8_t*) v69;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v71 = __riscv_vle8_v_u8mf2(v70, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v72 = __riscv_vand_vx_u8mf2(v71, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v73 = __riscv_vsrl_vx_u8mf2(v71, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v74 = __riscv_vzext_vf2_u16m1(v72, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v75 = __riscv_vreinterpret_v_u16m1_i16m1(v74);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v76 = __riscv_vzext_vf2_u16m1(v73, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v77 = __riscv_vreinterpret_v_u16m1_i16m1(v76);
      const uint8_t* v78 = v19 + 128;
      const uint8_t* v79 = (const uint8_t*) v78;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v80 = __riscv_vle8_v_u8mf2(v79, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v81 = __riscv_vand_vx_u8mf2(v80, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v82 = __riscv_vsrl_vx_u8mf2(v80, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v83 = __riscv_vzext_vf2_u16m1(v81, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v84 = __riscv_vreinterpret_v_u16m1_i16m1(v83);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v85 = __riscv_vzext_vf2_u16m1(v82, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v86 = __riscv_vreinterpret_v_u16m1_i16m1(v85);
      const uint8_t* v87 = v19 + 144;
      const uint8_t* v88 = (const uint8_t*) v87;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v89 = __riscv_vle8_v_u8mf2(v88, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v90 = __riscv_vand_vx_u8mf2(v89, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v91 = __riscv_vsrl_vx_u8mf2(v89, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v92 = __riscv_vzext_vf2_u16m1(v90, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v93 = __riscv_vreinterpret_v_u16m1_i16m1(v92);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v94 = __riscv_vzext_vf2_u16m1(v91, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v95 = __riscv_vreinterpret_v_u16m1_i16m1(v94);
      const uint8_t* v96 = v19 + 160;
      const uint8_t* v97 = (const uint8_t*) v96;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v98 = __riscv_vle8_v_u8mf2(v97, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v99 = __riscv_vand_vx_u8mf2(v98, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v100 = __riscv_vsrl_vx_u8mf2(v98, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v101 = __riscv_vzext_vf2_u16m1(v99, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v102 = __riscv_vreinterpret_v_u16m1_i16m1(v101);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v103 = __riscv_vzext_vf2_u16m1(v100, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v104 = __riscv_vreinterpret_v_u16m1_i16m1(v103);
      const uint8_t* v105 = v19 + 176;
      const uint8_t* v106 = (const uint8_t*) v105;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v107 = __riscv_vle8_v_u8mf2(v106, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v108 = __riscv_vand_vx_u8mf2(v107, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v109 = __riscv_vsrl_vx_u8mf2(v107, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v110 = __riscv_vzext_vf2_u16m1(v108, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v111 = __riscv_vreinterpret_v_u16m1_i16m1(v110);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v112 = __riscv_vzext_vf2_u16m1(v109, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v113 = __riscv_vreinterpret_v_u16m1_i16m1(v112);
      const uint8_t* v114 = v19 + 72;
      const uint8_t* v115 = (const uint8_t*) v114;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v116 = __riscv_vle8_v_u8mf2(v115, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v117 = __riscv_vand_vx_u8mf2(v116, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v118 = __riscv_vsrl_vx_u8mf2(v116, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v119 = __riscv_vzext_vf2_u16m1(v117, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v120 = __riscv_vreinterpret_v_u16m1_i16m1(v119);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v121 = __riscv_vzext_vf2_u16m1(v118, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v122 = __riscv_vreinterpret_v_u16m1_i16m1(v121);
      const uint8_t* v123 = v19 + 88;
      const uint8_t* v124 = (const uint8_t*) v123;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v125 = __riscv_vle8_v_u8mf2(v124, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v126 = __riscv_vand_vx_u8mf2(v125, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v127 = __riscv_vsrl_vx_u8mf2(v125, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v128 = __riscv_vzext_vf2_u16m1(v126, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v129 = __riscv_vreinterpret_v_u16m1_i16m1(v128);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v130 = __riscv_vzext_vf2_u16m1(v127, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v131 = __riscv_vreinterpret_v_u16m1_i16m1(v130);
      const uint8_t* v132 = v19 + 104;
      const uint8_t* v133 = (const uint8_t*) v132;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v134 = __riscv_vle8_v_u8mf2(v133, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v135 = __riscv_vand_vx_u8mf2(v134, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v136 = __riscv_vsrl_vx_u8mf2(v134, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v137 = __riscv_vzext_vf2_u16m1(v135, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v138 = __riscv_vreinterpret_v_u16m1_i16m1(v137);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v139 = __riscv_vzext_vf2_u16m1(v136, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v140 = __riscv_vreinterpret_v_u16m1_i16m1(v139);
      const uint8_t* v141 = v19 + 120;
      const uint8_t* v142 = (const uint8_t*) v141;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v143 = __riscv_vle8_v_u8mf2(v142, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v144 = __riscv_vand_vx_u8mf2(v143, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v145 = __riscv_vsrl_vx_u8mf2(v143, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v146 = __riscv_vzext_vf2_u16m1(v144, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v147 = __riscv_vreinterpret_v_u16m1_i16m1(v146);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v148 = __riscv_vzext_vf2_u16m1(v145, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v149 = __riscv_vreinterpret_v_u16m1_i16m1(v148);
      const uint8_t* v150 = v19 + 136;
      const uint8_t* v151 = (const uint8_t*) v150;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v152 = __riscv_vle8_v_u8mf2(v151, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v153 = __riscv_vand_vx_u8mf2(v152, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v154 = __riscv_vsrl_vx_u8mf2(v152, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v155 = __riscv_vzext_vf2_u16m1(v153, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v156 = __riscv_vreinterpret_v_u16m1_i16m1(v155);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v157 = __riscv_vzext_vf2_u16m1(v154, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v158 = __riscv_vreinterpret_v_u16m1_i16m1(v157);
      const uint8_t* v159 = v19 + 152;
      const uint8_t* v160 = (const uint8_t*) v159;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v161 = __riscv_vle8_v_u8mf2(v160, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v162 = __riscv_vand_vx_u8mf2(v161, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v163 = __riscv_vsrl_vx_u8mf2(v161, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v164 = __riscv_vzext_vf2_u16m1(v162, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v165 = __riscv_vreinterpret_v_u16m1_i16m1(v164);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v166 = __riscv_vzext_vf2_u16m1(v163, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v167 = __riscv_vreinterpret_v_u16m1_i16m1(v166);
      const uint8_t* v168 = v19 + 168;
      const uint8_t* v169 = (const uint8_t*) v168;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v170 = __riscv_vle8_v_u8mf2(v169, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v171 = __riscv_vand_vx_u8mf2(v170, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v172 = __riscv_vsrl_vx_u8mf2(v170, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v173 = __riscv_vzext_vf2_u16m1(v171, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v174 = __riscv_vreinterpret_v_u16m1_i16m1(v173);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v175 = __riscv_vzext_vf2_u16m1(v172, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v176 = __riscv_vreinterpret_v_u16m1_i16m1(v175);
      const uint8_t* v177 = v19 + 184;
      const uint8_t* v178 = (const uint8_t*) v177;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v179 = __riscv_vle8_v_u8mf2(v178, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v180 = __riscv_vand_vx_u8mf2(v179, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v181 = __riscv_vsrl_vx_u8mf2(v179, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v182 = __riscv_vzext_vf2_u16m1(v180, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v183 = __riscv_vreinterpret_v_u16m1_i16m1(v182);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v184 = __riscv_vzext_vf2_u16m1(v181, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v185 = __riscv_vreinterpret_v_u16m1_i16m1(v184);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold
      const uint8_t* v186 = v21 + 260;
      const int16_t* v187 = (const int16_t*) v186;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v188 = *(const int16_t *)(v187);
      vint32m2_t v189 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v190 = __riscv_vwmacc_vx_i32m2(v189, v188, v50, 8);
      v36 = v190;
      vint32m2_t v191 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v192 = __riscv_vwmacc_vx_i32m2(v191, v188, v122, 8);
      v40 = v192;
      const uint8_t* v193 = v21 + 262;
      const int16_t* v194 = (const int16_t*) v193;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v195 = *(const int16_t *)(v194);
      vint32m2_t v196 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v197 = __riscv_vwmacc_vx_i32m2(v196, v195, v59, 8);
      v36 = v197;
      vint32m2_t v198 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v199 = __riscv_vwmacc_vx_i32m2(v198, v195, v131, 8);
      v40 = v199;
      const uint8_t* v200 = v21 + 264;
      const int16_t* v201 = (const int16_t*) v200;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v202 = *(const int16_t *)(v201);
      vint32m2_t v203 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v204 = __riscv_vwmacc_vx_i32m2(v203, v202, v68, 8);
      v36 = v204;
      vint32m2_t v205 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v206 = __riscv_vwmacc_vx_i32m2(v205, v202, v140, 8);
      v40 = v206;
      const uint8_t* v207 = v21 + 266;
      const int16_t* v208 = (const int16_t*) v207;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v209 = *(const int16_t *)(v208);
      vint32m2_t v210 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v211 = __riscv_vwmacc_vx_i32m2(v210, v209, v77, 8);
      v36 = v211;
      vint32m2_t v212 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v213 = __riscv_vwmacc_vx_i32m2(v212, v209, v149, 8);
      v40 = v213;
      const uint8_t* v214 = v21 + 268;
      const int16_t* v215 = (const int16_t*) v214;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v216 = *(const int16_t *)(v215);
      vint32m2_t v217 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v218 = __riscv_vwmacc_vx_i32m2(v217, v216, v86, 8);
      v36 = v218;
      vint32m2_t v219 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v220 = __riscv_vwmacc_vx_i32m2(v219, v216, v158, 8);
      v40 = v220;
      const uint8_t* v221 = v21 + 270;
      const int16_t* v222 = (const int16_t*) v221;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v223 = *(const int16_t *)(v222);
      vint32m2_t v224 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v225 = __riscv_vwmacc_vx_i32m2(v224, v223, v95, 8);
      v36 = v225;
      vint32m2_t v226 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v227 = __riscv_vwmacc_vx_i32m2(v226, v223, v167, 8);
      v40 = v227;
      const uint8_t* v228 = v21 + 272;
      const int16_t* v229 = (const int16_t*) v228;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v230 = *(const int16_t *)(v229);
      vint32m2_t v231 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v232 = __riscv_vwmacc_vx_i32m2(v231, v230, v104, 8);
      v36 = v232;
      vint32m2_t v233 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v234 = __riscv_vwmacc_vx_i32m2(v233, v230, v176, 8);
      v40 = v234;
      const uint8_t* v235 = v21 + 274;
      const int16_t* v236 = (const int16_t*) v235;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v237 = *(const int16_t *)(v236);
      vint32m2_t v238 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v239 = __riscv_vwmacc_vx_i32m2(v238, v237, v113, 8);
      v36 = v239;
      vint32m2_t v240 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v241 = __riscv_vwmacc_vx_i32m2(v240, v237, v185, 8);
      v40 = v241;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v242 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v243 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v244 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v245 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v246 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v247 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v248 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v249 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v250 = v19 + 320;
      const uint8_t* v251 = (const uint8_t*) v250;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v252 = __riscv_vle8_v_u8mf2(v251, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v253 = __riscv_vand_vx_u8mf2(v252, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v254 = __riscv_vreinterpret_v_u8mf2_i8mf2(v253);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v255 = v21 + 4;
      const int8_t* v256 = (const int8_t*) v255;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v257 = *(const int8_t *)(v256);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v258 = __riscv_vwmacc_vx_i16m1(v242, v257, v254, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v259 = __riscv_vsrl_vx_u8mf2(v252, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v260 = __riscv_vand_vx_u8mf2(v259, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v261 = __riscv_vreinterpret_v_u8mf2_i8mf2(v260);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v262 = v21 + 36;
      const int8_t* v263 = (const int8_t*) v262;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v264 = *(const int8_t *)(v263);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v265 = __riscv_vwmacc_vx_i16m1(v243, v264, v261, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v266 = __riscv_vsrl_vx_u8mf2(v252, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v267 = __riscv_vand_vx_u8mf2(v266, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v268 = __riscv_vreinterpret_v_u8mf2_i8mf2(v267);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v269 = v21 + 68;
      const int8_t* v270 = (const int8_t*) v269;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v271 = *(const int8_t *)(v270);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v272 = __riscv_vwmacc_vx_i16m1(v244, v271, v268, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v273 = __riscv_vsrl_vx_u8mf2(v252, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v274 = __riscv_vand_vx_u8mf2(v273, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v275 = __riscv_vreinterpret_v_u8mf2_i8mf2(v274);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v276 = v21 + 100;
      const int8_t* v277 = (const int8_t*) v276;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v278 = *(const int8_t *)(v277);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v279 = __riscv_vwmacc_vx_i16m1(v245, v278, v275, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v280 = v19 + 328;
      const uint8_t* v281 = (const uint8_t*) v280;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v282 = __riscv_vle8_v_u8mf2(v281, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v283 = __riscv_vand_vx_u8mf2(v282, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v284 = __riscv_vreinterpret_v_u8mf2_i8mf2(v283);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v285 = v21 + 4;
      const int8_t* v286 = (const int8_t*) v285;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v287 = *(const int8_t *)(v286);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v288 = __riscv_vwmacc_vx_i16m1(v246, v287, v284, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v289 = __riscv_vsrl_vx_u8mf2(v282, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v290 = __riscv_vand_vx_u8mf2(v289, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v291 = __riscv_vreinterpret_v_u8mf2_i8mf2(v290);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v292 = v21 + 36;
      const int8_t* v293 = (const int8_t*) v292;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v294 = *(const int8_t *)(v293);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v295 = __riscv_vwmacc_vx_i16m1(v247, v294, v291, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v296 = __riscv_vsrl_vx_u8mf2(v282, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v297 = __riscv_vand_vx_u8mf2(v296, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v298 = __riscv_vreinterpret_v_u8mf2_i8mf2(v297);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v299 = v21 + 68;
      const int8_t* v300 = (const int8_t*) v299;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v301 = *(const int8_t *)(v300);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v302 = __riscv_vwmacc_vx_i16m1(v248, v301, v298, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v303 = __riscv_vsrl_vx_u8mf2(v282, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v304 = __riscv_vand_vx_u8mf2(v303, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v305 = __riscv_vreinterpret_v_u8mf2_i8mf2(v304);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v306 = v21 + 100;
      const int8_t* v307 = (const int8_t*) v306;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v308 = *(const int8_t *)(v307);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v309 = __riscv_vwmacc_vx_i16m1(v249, v308, v305, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v310 = v19 + 336;
      const uint8_t* v311 = (const uint8_t*) v310;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v312 = __riscv_vle8_v_u8mf2(v311, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v313 = __riscv_vand_vx_u8mf2(v312, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v314 = __riscv_vreinterpret_v_u8mf2_i8mf2(v313);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v315 = v21 + 5;
      const int8_t* v316 = (const int8_t*) v315;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v317 = *(const int8_t *)(v316);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v318 = __riscv_vwmacc_vx_i16m1(v258, v317, v314, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v319 = __riscv_vsrl_vx_u8mf2(v312, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v320 = __riscv_vand_vx_u8mf2(v319, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v321 = __riscv_vreinterpret_v_u8mf2_i8mf2(v320);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v322 = v21 + 37;
      const int8_t* v323 = (const int8_t*) v322;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v324 = *(const int8_t *)(v323);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v325 = __riscv_vwmacc_vx_i16m1(v265, v324, v321, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v326 = __riscv_vsrl_vx_u8mf2(v312, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v327 = __riscv_vand_vx_u8mf2(v326, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v328 = __riscv_vreinterpret_v_u8mf2_i8mf2(v327);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v329 = v21 + 69;
      const int8_t* v330 = (const int8_t*) v329;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v331 = *(const int8_t *)(v330);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v332 = __riscv_vwmacc_vx_i16m1(v272, v331, v328, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v333 = __riscv_vsrl_vx_u8mf2(v312, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v334 = __riscv_vand_vx_u8mf2(v333, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v335 = __riscv_vreinterpret_v_u8mf2_i8mf2(v334);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v336 = v21 + 101;
      const int8_t* v337 = (const int8_t*) v336;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v338 = *(const int8_t *)(v337);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v339 = __riscv_vwmacc_vx_i16m1(v279, v338, v335, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v340 = v19 + 344;
      const uint8_t* v341 = (const uint8_t*) v340;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v342 = __riscv_vle8_v_u8mf2(v341, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v343 = __riscv_vand_vx_u8mf2(v342, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v344 = __riscv_vreinterpret_v_u8mf2_i8mf2(v343);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v345 = v21 + 5;
      const int8_t* v346 = (const int8_t*) v345;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v347 = *(const int8_t *)(v346);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v348 = __riscv_vwmacc_vx_i16m1(v288, v347, v344, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v349 = __riscv_vsrl_vx_u8mf2(v342, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v350 = __riscv_vand_vx_u8mf2(v349, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v351 = __riscv_vreinterpret_v_u8mf2_i8mf2(v350);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v352 = v21 + 37;
      const int8_t* v353 = (const int8_t*) v352;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v354 = *(const int8_t *)(v353);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v355 = __riscv_vwmacc_vx_i16m1(v295, v354, v351, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v356 = __riscv_vsrl_vx_u8mf2(v342, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v357 = __riscv_vand_vx_u8mf2(v356, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v358 = __riscv_vreinterpret_v_u8mf2_i8mf2(v357);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v359 = v21 + 69;
      const int8_t* v360 = (const int8_t*) v359;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v361 = *(const int8_t *)(v360);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v362 = __riscv_vwmacc_vx_i16m1(v302, v361, v358, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v363 = __riscv_vsrl_vx_u8mf2(v342, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v364 = __riscv_vand_vx_u8mf2(v363, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v365 = __riscv_vreinterpret_v_u8mf2_i8mf2(v364);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v366 = v21 + 101;
      const int8_t* v367 = (const int8_t*) v366;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v368 = *(const int8_t *)(v367);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v369 = __riscv_vwmacc_vx_i16m1(v309, v368, v365, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v370 = v19 + 352;
      const uint8_t* v371 = (const uint8_t*) v370;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v372 = __riscv_vle8_v_u8mf2(v371, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v373 = __riscv_vand_vx_u8mf2(v372, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v374 = __riscv_vreinterpret_v_u8mf2_i8mf2(v373);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v375 = v21 + 6;
      const int8_t* v376 = (const int8_t*) v375;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v377 = *(const int8_t *)(v376);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v378 = __riscv_vwmacc_vx_i16m1(v318, v377, v374, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v379 = __riscv_vsrl_vx_u8mf2(v372, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v380 = __riscv_vand_vx_u8mf2(v379, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v381 = __riscv_vreinterpret_v_u8mf2_i8mf2(v380);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v382 = v21 + 38;
      const int8_t* v383 = (const int8_t*) v382;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v384 = *(const int8_t *)(v383);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v385 = __riscv_vwmacc_vx_i16m1(v325, v384, v381, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v386 = __riscv_vsrl_vx_u8mf2(v372, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v387 = __riscv_vand_vx_u8mf2(v386, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v388 = __riscv_vreinterpret_v_u8mf2_i8mf2(v387);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v389 = v21 + 70;
      const int8_t* v390 = (const int8_t*) v389;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v391 = *(const int8_t *)(v390);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v392 = __riscv_vwmacc_vx_i16m1(v332, v391, v388, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v393 = __riscv_vsrl_vx_u8mf2(v372, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v394 = __riscv_vand_vx_u8mf2(v393, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v395 = __riscv_vreinterpret_v_u8mf2_i8mf2(v394);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v396 = v21 + 102;
      const int8_t* v397 = (const int8_t*) v396;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v398 = *(const int8_t *)(v397);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v399 = __riscv_vwmacc_vx_i16m1(v339, v398, v395, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v400 = v19 + 360;
      const uint8_t* v401 = (const uint8_t*) v400;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v402 = __riscv_vle8_v_u8mf2(v401, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v403 = __riscv_vand_vx_u8mf2(v402, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v404 = __riscv_vreinterpret_v_u8mf2_i8mf2(v403);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v405 = v21 + 6;
      const int8_t* v406 = (const int8_t*) v405;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v407 = *(const int8_t *)(v406);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v408 = __riscv_vwmacc_vx_i16m1(v348, v407, v404, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v409 = __riscv_vsrl_vx_u8mf2(v402, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v410 = __riscv_vand_vx_u8mf2(v409, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v411 = __riscv_vreinterpret_v_u8mf2_i8mf2(v410);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v412 = v21 + 38;
      const int8_t* v413 = (const int8_t*) v412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v414 = *(const int8_t *)(v413);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v415 = __riscv_vwmacc_vx_i16m1(v355, v414, v411, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v416 = __riscv_vsrl_vx_u8mf2(v402, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v417 = __riscv_vand_vx_u8mf2(v416, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v418 = __riscv_vreinterpret_v_u8mf2_i8mf2(v417);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v419 = v21 + 70;
      const int8_t* v420 = (const int8_t*) v419;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v421 = *(const int8_t *)(v420);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v422 = __riscv_vwmacc_vx_i16m1(v362, v421, v418, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v423 = __riscv_vsrl_vx_u8mf2(v402, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v424 = __riscv_vand_vx_u8mf2(v423, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v425 = __riscv_vreinterpret_v_u8mf2_i8mf2(v424);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v426 = v21 + 102;
      const int8_t* v427 = (const int8_t*) v426;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v428 = *(const int8_t *)(v427);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v429 = __riscv_vwmacc_vx_i16m1(v369, v428, v425, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v430 = v19 + 368;
      const uint8_t* v431 = (const uint8_t*) v430;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v432 = __riscv_vle8_v_u8mf2(v431, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v433 = __riscv_vand_vx_u8mf2(v432, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v434 = __riscv_vreinterpret_v_u8mf2_i8mf2(v433);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v435 = v21 + 7;
      const int8_t* v436 = (const int8_t*) v435;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v437 = *(const int8_t *)(v436);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v438 = __riscv_vwmacc_vx_i16m1(v378, v437, v434, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v439 = __riscv_vsrl_vx_u8mf2(v432, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v440 = __riscv_vand_vx_u8mf2(v439, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v441 = __riscv_vreinterpret_v_u8mf2_i8mf2(v440);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v442 = v21 + 39;
      const int8_t* v443 = (const int8_t*) v442;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v444 = *(const int8_t *)(v443);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v445 = __riscv_vwmacc_vx_i16m1(v385, v444, v441, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v446 = __riscv_vsrl_vx_u8mf2(v432, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v447 = __riscv_vand_vx_u8mf2(v446, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v448 = __riscv_vreinterpret_v_u8mf2_i8mf2(v447);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v449 = v21 + 71;
      const int8_t* v450 = (const int8_t*) v449;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v451 = *(const int8_t *)(v450);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v452 = __riscv_vwmacc_vx_i16m1(v392, v451, v448, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v453 = __riscv_vsrl_vx_u8mf2(v432, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v454 = __riscv_vand_vx_u8mf2(v453, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v455 = __riscv_vreinterpret_v_u8mf2_i8mf2(v454);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v456 = v21 + 103;
      const int8_t* v457 = (const int8_t*) v456;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v458 = *(const int8_t *)(v457);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v459 = __riscv_vwmacc_vx_i16m1(v399, v458, v455, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v460 = v19 + 376;
      const uint8_t* v461 = (const uint8_t*) v460;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v462 = __riscv_vle8_v_u8mf2(v461, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v463 = __riscv_vand_vx_u8mf2(v462, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v464 = __riscv_vreinterpret_v_u8mf2_i8mf2(v463);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v465 = v21 + 7;
      const int8_t* v466 = (const int8_t*) v465;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v467 = *(const int8_t *)(v466);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v468 = __riscv_vwmacc_vx_i16m1(v408, v467, v464, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v469 = __riscv_vsrl_vx_u8mf2(v462, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v470 = __riscv_vand_vx_u8mf2(v469, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v471 = __riscv_vreinterpret_v_u8mf2_i8mf2(v470);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v472 = v21 + 39;
      const int8_t* v473 = (const int8_t*) v472;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v474 = *(const int8_t *)(v473);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v475 = __riscv_vwmacc_vx_i16m1(v415, v474, v471, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v476 = __riscv_vsrl_vx_u8mf2(v462, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v477 = __riscv_vand_vx_u8mf2(v476, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v478 = __riscv_vreinterpret_v_u8mf2_i8mf2(v477);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v479 = v21 + 71;
      const int8_t* v480 = (const int8_t*) v479;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v481 = *(const int8_t *)(v480);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v482 = __riscv_vwmacc_vx_i16m1(v422, v481, v478, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v483 = __riscv_vsrl_vx_u8mf2(v462, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v484 = __riscv_vand_vx_u8mf2(v483, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v485 = __riscv_vreinterpret_v_u8mf2_i8mf2(v484);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v486 = v21 + 103;
      const int8_t* v487 = (const int8_t*) v486;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v488 = *(const int8_t *)(v487);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v489 = __riscv_vwmacc_vx_i16m1(v429, v488, v485, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v490 = v19 + 384;
      const uint8_t* v491 = (const uint8_t*) v490;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v492 = __riscv_vle8_v_u8mf2(v491, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v493 = __riscv_vand_vx_u8mf2(v492, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v494 = __riscv_vreinterpret_v_u8mf2_i8mf2(v493);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v495 = v21 + 8;
      const int8_t* v496 = (const int8_t*) v495;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v497 = *(const int8_t *)(v496);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v498 = __riscv_vwmacc_vx_i16m1(v438, v497, v494, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v499 = __riscv_vsrl_vx_u8mf2(v492, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v500 = __riscv_vand_vx_u8mf2(v499, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v501 = __riscv_vreinterpret_v_u8mf2_i8mf2(v500);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v502 = v21 + 40;
      const int8_t* v503 = (const int8_t*) v502;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v504 = *(const int8_t *)(v503);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v505 = __riscv_vwmacc_vx_i16m1(v445, v504, v501, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v506 = __riscv_vsrl_vx_u8mf2(v492, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v507 = __riscv_vand_vx_u8mf2(v506, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v508 = __riscv_vreinterpret_v_u8mf2_i8mf2(v507);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v509 = v21 + 72;
      const int8_t* v510 = (const int8_t*) v509;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v511 = *(const int8_t *)(v510);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v512 = __riscv_vwmacc_vx_i16m1(v452, v511, v508, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v513 = __riscv_vsrl_vx_u8mf2(v492, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v514 = __riscv_vand_vx_u8mf2(v513, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v515 = __riscv_vreinterpret_v_u8mf2_i8mf2(v514);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v516 = v21 + 104;
      const int8_t* v517 = (const int8_t*) v516;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v518 = *(const int8_t *)(v517);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v519 = __riscv_vwmacc_vx_i16m1(v459, v518, v515, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v520 = v19 + 392;
      const uint8_t* v521 = (const uint8_t*) v520;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v522 = __riscv_vle8_v_u8mf2(v521, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v523 = __riscv_vand_vx_u8mf2(v522, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v524 = __riscv_vreinterpret_v_u8mf2_i8mf2(v523);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v525 = v21 + 8;
      const int8_t* v526 = (const int8_t*) v525;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v527 = *(const int8_t *)(v526);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v528 = __riscv_vwmacc_vx_i16m1(v468, v527, v524, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v529 = __riscv_vsrl_vx_u8mf2(v522, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v530 = __riscv_vand_vx_u8mf2(v529, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v531 = __riscv_vreinterpret_v_u8mf2_i8mf2(v530);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v532 = v21 + 40;
      const int8_t* v533 = (const int8_t*) v532;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v534 = *(const int8_t *)(v533);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v535 = __riscv_vwmacc_vx_i16m1(v475, v534, v531, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v536 = __riscv_vsrl_vx_u8mf2(v522, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v537 = __riscv_vand_vx_u8mf2(v536, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v538 = __riscv_vreinterpret_v_u8mf2_i8mf2(v537);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v539 = v21 + 72;
      const int8_t* v540 = (const int8_t*) v539;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v541 = *(const int8_t *)(v540);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v542 = __riscv_vwmacc_vx_i16m1(v482, v541, v538, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v543 = __riscv_vsrl_vx_u8mf2(v522, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v544 = __riscv_vand_vx_u8mf2(v543, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v545 = __riscv_vreinterpret_v_u8mf2_i8mf2(v544);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v546 = v21 + 104;
      const int8_t* v547 = (const int8_t*) v546;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v548 = *(const int8_t *)(v547);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v549 = __riscv_vwmacc_vx_i16m1(v489, v548, v545, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v550 = v19 + 400;
      const uint8_t* v551 = (const uint8_t*) v550;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v552 = __riscv_vle8_v_u8mf2(v551, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v553 = __riscv_vand_vx_u8mf2(v552, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v554 = __riscv_vreinterpret_v_u8mf2_i8mf2(v553);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v555 = v21 + 9;
      const int8_t* v556 = (const int8_t*) v555;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v557 = *(const int8_t *)(v556);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v558 = __riscv_vwmacc_vx_i16m1(v498, v557, v554, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v559 = __riscv_vsrl_vx_u8mf2(v552, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v560 = __riscv_vand_vx_u8mf2(v559, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v561 = __riscv_vreinterpret_v_u8mf2_i8mf2(v560);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v562 = v21 + 41;
      const int8_t* v563 = (const int8_t*) v562;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v564 = *(const int8_t *)(v563);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v565 = __riscv_vwmacc_vx_i16m1(v505, v564, v561, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v566 = __riscv_vsrl_vx_u8mf2(v552, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v567 = __riscv_vand_vx_u8mf2(v566, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v568 = __riscv_vreinterpret_v_u8mf2_i8mf2(v567);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v569 = v21 + 73;
      const int8_t* v570 = (const int8_t*) v569;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v571 = *(const int8_t *)(v570);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v572 = __riscv_vwmacc_vx_i16m1(v512, v571, v568, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v573 = __riscv_vsrl_vx_u8mf2(v552, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v574 = __riscv_vand_vx_u8mf2(v573, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v575 = __riscv_vreinterpret_v_u8mf2_i8mf2(v574);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v576 = v21 + 105;
      const int8_t* v577 = (const int8_t*) v576;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v578 = *(const int8_t *)(v577);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v579 = __riscv_vwmacc_vx_i16m1(v519, v578, v575, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v580 = v19 + 408;
      const uint8_t* v581 = (const uint8_t*) v580;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v582 = __riscv_vle8_v_u8mf2(v581, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v583 = __riscv_vand_vx_u8mf2(v582, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v584 = __riscv_vreinterpret_v_u8mf2_i8mf2(v583);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v585 = v21 + 9;
      const int8_t* v586 = (const int8_t*) v585;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v587 = *(const int8_t *)(v586);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v588 = __riscv_vwmacc_vx_i16m1(v528, v587, v584, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v589 = __riscv_vsrl_vx_u8mf2(v582, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v590 = __riscv_vand_vx_u8mf2(v589, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v591 = __riscv_vreinterpret_v_u8mf2_i8mf2(v590);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v592 = v21 + 41;
      const int8_t* v593 = (const int8_t*) v592;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v594 = *(const int8_t *)(v593);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v595 = __riscv_vwmacc_vx_i16m1(v535, v594, v591, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v596 = __riscv_vsrl_vx_u8mf2(v582, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v597 = __riscv_vand_vx_u8mf2(v596, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v598 = __riscv_vreinterpret_v_u8mf2_i8mf2(v597);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v599 = v21 + 73;
      const int8_t* v600 = (const int8_t*) v599;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v601 = *(const int8_t *)(v600);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v602 = __riscv_vwmacc_vx_i16m1(v542, v601, v598, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v603 = __riscv_vsrl_vx_u8mf2(v582, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v604 = __riscv_vand_vx_u8mf2(v603, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v605 = __riscv_vreinterpret_v_u8mf2_i8mf2(v604);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v606 = v21 + 105;
      const int8_t* v607 = (const int8_t*) v606;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v608 = *(const int8_t *)(v607);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v609 = __riscv_vwmacc_vx_i16m1(v549, v608, v605, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v610 = v19 + 416;
      const uint8_t* v611 = (const uint8_t*) v610;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v612 = __riscv_vle8_v_u8mf2(v611, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v613 = __riscv_vand_vx_u8mf2(v612, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v614 = __riscv_vreinterpret_v_u8mf2_i8mf2(v613);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v615 = v21 + 10;
      const int8_t* v616 = (const int8_t*) v615;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v617 = *(const int8_t *)(v616);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v618 = __riscv_vwmacc_vx_i16m1(v558, v617, v614, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v619 = __riscv_vsrl_vx_u8mf2(v612, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v620 = __riscv_vand_vx_u8mf2(v619, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v621 = __riscv_vreinterpret_v_u8mf2_i8mf2(v620);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v622 = v21 + 42;
      const int8_t* v623 = (const int8_t*) v622;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v624 = *(const int8_t *)(v623);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v625 = __riscv_vwmacc_vx_i16m1(v565, v624, v621, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v626 = __riscv_vsrl_vx_u8mf2(v612, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v627 = __riscv_vand_vx_u8mf2(v626, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v628 = __riscv_vreinterpret_v_u8mf2_i8mf2(v627);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v629 = v21 + 74;
      const int8_t* v630 = (const int8_t*) v629;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v631 = *(const int8_t *)(v630);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v632 = __riscv_vwmacc_vx_i16m1(v572, v631, v628, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v633 = __riscv_vsrl_vx_u8mf2(v612, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v634 = __riscv_vand_vx_u8mf2(v633, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v635 = __riscv_vreinterpret_v_u8mf2_i8mf2(v634);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v636 = v21 + 106;
      const int8_t* v637 = (const int8_t*) v636;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v638 = *(const int8_t *)(v637);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v639 = __riscv_vwmacc_vx_i16m1(v579, v638, v635, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v640 = v19 + 424;
      const uint8_t* v641 = (const uint8_t*) v640;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v642 = __riscv_vle8_v_u8mf2(v641, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v643 = __riscv_vand_vx_u8mf2(v642, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v644 = __riscv_vreinterpret_v_u8mf2_i8mf2(v643);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v645 = v21 + 10;
      const int8_t* v646 = (const int8_t*) v645;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v647 = *(const int8_t *)(v646);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v648 = __riscv_vwmacc_vx_i16m1(v588, v647, v644, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v649 = __riscv_vsrl_vx_u8mf2(v642, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v650 = __riscv_vand_vx_u8mf2(v649, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v651 = __riscv_vreinterpret_v_u8mf2_i8mf2(v650);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v652 = v21 + 42;
      const int8_t* v653 = (const int8_t*) v652;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v654 = *(const int8_t *)(v653);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v655 = __riscv_vwmacc_vx_i16m1(v595, v654, v651, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v656 = __riscv_vsrl_vx_u8mf2(v642, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v657 = __riscv_vand_vx_u8mf2(v656, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v658 = __riscv_vreinterpret_v_u8mf2_i8mf2(v657);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v659 = v21 + 74;
      const int8_t* v660 = (const int8_t*) v659;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v661 = *(const int8_t *)(v660);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v662 = __riscv_vwmacc_vx_i16m1(v602, v661, v658, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v663 = __riscv_vsrl_vx_u8mf2(v642, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v664 = __riscv_vand_vx_u8mf2(v663, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v665 = __riscv_vreinterpret_v_u8mf2_i8mf2(v664);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v666 = v21 + 106;
      const int8_t* v667 = (const int8_t*) v666;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v668 = *(const int8_t *)(v667);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v669 = __riscv_vwmacc_vx_i16m1(v609, v668, v665, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v670 = v19 + 432;
      const uint8_t* v671 = (const uint8_t*) v670;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v672 = __riscv_vle8_v_u8mf2(v671, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v673 = __riscv_vand_vx_u8mf2(v672, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v674 = __riscv_vreinterpret_v_u8mf2_i8mf2(v673);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v675 = v21 + 11;
      const int8_t* v676 = (const int8_t*) v675;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v677 = *(const int8_t *)(v676);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v678 = __riscv_vwmacc_vx_i16m1(v618, v677, v674, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v679 = __riscv_vsrl_vx_u8mf2(v672, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v680 = __riscv_vand_vx_u8mf2(v679, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v681 = __riscv_vreinterpret_v_u8mf2_i8mf2(v680);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v682 = v21 + 43;
      const int8_t* v683 = (const int8_t*) v682;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v684 = *(const int8_t *)(v683);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v685 = __riscv_vwmacc_vx_i16m1(v625, v684, v681, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v686 = __riscv_vsrl_vx_u8mf2(v672, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v687 = __riscv_vand_vx_u8mf2(v686, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v688 = __riscv_vreinterpret_v_u8mf2_i8mf2(v687);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v689 = v21 + 75;
      const int8_t* v690 = (const int8_t*) v689;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v691 = *(const int8_t *)(v690);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v692 = __riscv_vwmacc_vx_i16m1(v632, v691, v688, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v693 = __riscv_vsrl_vx_u8mf2(v672, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v694 = __riscv_vand_vx_u8mf2(v693, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v695 = __riscv_vreinterpret_v_u8mf2_i8mf2(v694);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v696 = v21 + 107;
      const int8_t* v697 = (const int8_t*) v696;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v698 = *(const int8_t *)(v697);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v699 = __riscv_vwmacc_vx_i16m1(v639, v698, v695, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v700 = v19 + 440;
      const uint8_t* v701 = (const uint8_t*) v700;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v702 = __riscv_vle8_v_u8mf2(v701, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v703 = __riscv_vand_vx_u8mf2(v702, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v704 = __riscv_vreinterpret_v_u8mf2_i8mf2(v703);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v705 = v21 + 11;
      const int8_t* v706 = (const int8_t*) v705;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v707 = *(const int8_t *)(v706);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v708 = __riscv_vwmacc_vx_i16m1(v648, v707, v704, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v709 = __riscv_vsrl_vx_u8mf2(v702, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v710 = __riscv_vand_vx_u8mf2(v709, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v711 = __riscv_vreinterpret_v_u8mf2_i8mf2(v710);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v712 = v21 + 43;
      const int8_t* v713 = (const int8_t*) v712;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v714 = *(const int8_t *)(v713);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v715 = __riscv_vwmacc_vx_i16m1(v655, v714, v711, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v716 = __riscv_vsrl_vx_u8mf2(v702, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v717 = __riscv_vand_vx_u8mf2(v716, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v718 = __riscv_vreinterpret_v_u8mf2_i8mf2(v717);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v719 = v21 + 75;
      const int8_t* v720 = (const int8_t*) v719;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v721 = *(const int8_t *)(v720);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v722 = __riscv_vwmacc_vx_i16m1(v662, v721, v718, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v723 = __riscv_vsrl_vx_u8mf2(v702, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v724 = __riscv_vand_vx_u8mf2(v723, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v725 = __riscv_vreinterpret_v_u8mf2_i8mf2(v724);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v726 = v21 + 107;
      const int8_t* v727 = (const int8_t*) v726;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v728 = *(const int8_t *)(v727);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v729 = __riscv_vwmacc_vx_i16m1(v669, v728, v725, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v730 = v19 + 448;
      const uint8_t* v731 = (const uint8_t*) v730;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v732 = __riscv_vle8_v_u8mf2(v731, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v733 = __riscv_vand_vx_u8mf2(v732, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v734 = __riscv_vreinterpret_v_u8mf2_i8mf2(v733);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v735 = v21 + 12;
      const int8_t* v736 = (const int8_t*) v735;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v737 = *(const int8_t *)(v736);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v738 = __riscv_vwmacc_vx_i16m1(v678, v737, v734, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v739 = __riscv_vsrl_vx_u8mf2(v732, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v740 = __riscv_vand_vx_u8mf2(v739, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v741 = __riscv_vreinterpret_v_u8mf2_i8mf2(v740);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v742 = v21 + 44;
      const int8_t* v743 = (const int8_t*) v742;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v744 = *(const int8_t *)(v743);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v745 = __riscv_vwmacc_vx_i16m1(v685, v744, v741, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v746 = __riscv_vsrl_vx_u8mf2(v732, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v747 = __riscv_vand_vx_u8mf2(v746, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v748 = __riscv_vreinterpret_v_u8mf2_i8mf2(v747);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v749 = v21 + 76;
      const int8_t* v750 = (const int8_t*) v749;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v751 = *(const int8_t *)(v750);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v752 = __riscv_vwmacc_vx_i16m1(v692, v751, v748, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v753 = __riscv_vsrl_vx_u8mf2(v732, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v754 = __riscv_vand_vx_u8mf2(v753, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v755 = __riscv_vreinterpret_v_u8mf2_i8mf2(v754);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v756 = v21 + 108;
      const int8_t* v757 = (const int8_t*) v756;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v758 = *(const int8_t *)(v757);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v759 = __riscv_vwmacc_vx_i16m1(v699, v758, v755, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v760 = v19 + 456;
      const uint8_t* v761 = (const uint8_t*) v760;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v762 = __riscv_vle8_v_u8mf2(v761, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v763 = __riscv_vand_vx_u8mf2(v762, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v764 = __riscv_vreinterpret_v_u8mf2_i8mf2(v763);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v765 = v21 + 12;
      const int8_t* v766 = (const int8_t*) v765;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v767 = *(const int8_t *)(v766);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v768 = __riscv_vwmacc_vx_i16m1(v708, v767, v764, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v769 = __riscv_vsrl_vx_u8mf2(v762, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v770 = __riscv_vand_vx_u8mf2(v769, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v771 = __riscv_vreinterpret_v_u8mf2_i8mf2(v770);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v772 = v21 + 44;
      const int8_t* v773 = (const int8_t*) v772;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v774 = *(const int8_t *)(v773);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v775 = __riscv_vwmacc_vx_i16m1(v715, v774, v771, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v776 = __riscv_vsrl_vx_u8mf2(v762, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v777 = __riscv_vand_vx_u8mf2(v776, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v778 = __riscv_vreinterpret_v_u8mf2_i8mf2(v777);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v779 = v21 + 76;
      const int8_t* v780 = (const int8_t*) v779;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v781 = *(const int8_t *)(v780);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v782 = __riscv_vwmacc_vx_i16m1(v722, v781, v778, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v783 = __riscv_vsrl_vx_u8mf2(v762, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v784 = __riscv_vand_vx_u8mf2(v783, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v785 = __riscv_vreinterpret_v_u8mf2_i8mf2(v784);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v786 = v21 + 108;
      const int8_t* v787 = (const int8_t*) v786;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v788 = *(const int8_t *)(v787);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v789 = __riscv_vwmacc_vx_i16m1(v729, v788, v785, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v790 = v19 + 464;
      const uint8_t* v791 = (const uint8_t*) v790;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v792 = __riscv_vle8_v_u8mf2(v791, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v793 = __riscv_vand_vx_u8mf2(v792, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v794 = __riscv_vreinterpret_v_u8mf2_i8mf2(v793);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v795 = v21 + 13;
      const int8_t* v796 = (const int8_t*) v795;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v797 = *(const int8_t *)(v796);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v798 = __riscv_vwmacc_vx_i16m1(v738, v797, v794, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v799 = __riscv_vsrl_vx_u8mf2(v792, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v800 = __riscv_vand_vx_u8mf2(v799, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v801 = __riscv_vreinterpret_v_u8mf2_i8mf2(v800);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v802 = v21 + 45;
      const int8_t* v803 = (const int8_t*) v802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v804 = *(const int8_t *)(v803);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v805 = __riscv_vwmacc_vx_i16m1(v745, v804, v801, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v806 = __riscv_vsrl_vx_u8mf2(v792, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v807 = __riscv_vand_vx_u8mf2(v806, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v808 = __riscv_vreinterpret_v_u8mf2_i8mf2(v807);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v809 = v21 + 77;
      const int8_t* v810 = (const int8_t*) v809;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v811 = *(const int8_t *)(v810);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v812 = __riscv_vwmacc_vx_i16m1(v752, v811, v808, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v813 = __riscv_vsrl_vx_u8mf2(v792, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v814 = __riscv_vand_vx_u8mf2(v813, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v815 = __riscv_vreinterpret_v_u8mf2_i8mf2(v814);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v816 = v21 + 109;
      const int8_t* v817 = (const int8_t*) v816;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v818 = *(const int8_t *)(v817);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v819 = __riscv_vwmacc_vx_i16m1(v759, v818, v815, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v820 = v19 + 472;
      const uint8_t* v821 = (const uint8_t*) v820;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v822 = __riscv_vle8_v_u8mf2(v821, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v823 = __riscv_vand_vx_u8mf2(v822, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v824 = __riscv_vreinterpret_v_u8mf2_i8mf2(v823);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v825 = v21 + 13;
      const int8_t* v826 = (const int8_t*) v825;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v827 = *(const int8_t *)(v826);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v828 = __riscv_vwmacc_vx_i16m1(v768, v827, v824, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v829 = __riscv_vsrl_vx_u8mf2(v822, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v830 = __riscv_vand_vx_u8mf2(v829, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v831 = __riscv_vreinterpret_v_u8mf2_i8mf2(v830);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v832 = v21 + 45;
      const int8_t* v833 = (const int8_t*) v832;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v834 = *(const int8_t *)(v833);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v835 = __riscv_vwmacc_vx_i16m1(v775, v834, v831, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v836 = __riscv_vsrl_vx_u8mf2(v822, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v837 = __riscv_vand_vx_u8mf2(v836, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v838 = __riscv_vreinterpret_v_u8mf2_i8mf2(v837);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v839 = v21 + 77;
      const int8_t* v840 = (const int8_t*) v839;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v841 = *(const int8_t *)(v840);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v842 = __riscv_vwmacc_vx_i16m1(v782, v841, v838, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v843 = __riscv_vsrl_vx_u8mf2(v822, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v844 = __riscv_vand_vx_u8mf2(v843, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v845 = __riscv_vreinterpret_v_u8mf2_i8mf2(v844);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v846 = v21 + 109;
      const int8_t* v847 = (const int8_t*) v846;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v848 = *(const int8_t *)(v847);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v849 = __riscv_vwmacc_vx_i16m1(v789, v848, v845, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v850 = v19 + 480;
      const uint8_t* v851 = (const uint8_t*) v850;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v852 = __riscv_vle8_v_u8mf2(v851, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v853 = __riscv_vand_vx_u8mf2(v852, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v854 = __riscv_vreinterpret_v_u8mf2_i8mf2(v853);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v855 = v21 + 14;
      const int8_t* v856 = (const int8_t*) v855;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v857 = *(const int8_t *)(v856);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v858 = __riscv_vwmacc_vx_i16m1(v798, v857, v854, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v859 = __riscv_vsrl_vx_u8mf2(v852, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v860 = __riscv_vand_vx_u8mf2(v859, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v861 = __riscv_vreinterpret_v_u8mf2_i8mf2(v860);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v862 = v21 + 46;
      const int8_t* v863 = (const int8_t*) v862;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v864 = *(const int8_t *)(v863);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v865 = __riscv_vwmacc_vx_i16m1(v805, v864, v861, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v866 = __riscv_vsrl_vx_u8mf2(v852, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v867 = __riscv_vand_vx_u8mf2(v866, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v868 = __riscv_vreinterpret_v_u8mf2_i8mf2(v867);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v869 = v21 + 78;
      const int8_t* v870 = (const int8_t*) v869;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v871 = *(const int8_t *)(v870);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v872 = __riscv_vwmacc_vx_i16m1(v812, v871, v868, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v873 = __riscv_vsrl_vx_u8mf2(v852, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v874 = __riscv_vand_vx_u8mf2(v873, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v875 = __riscv_vreinterpret_v_u8mf2_i8mf2(v874);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v876 = v21 + 110;
      const int8_t* v877 = (const int8_t*) v876;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v878 = *(const int8_t *)(v877);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v879 = __riscv_vwmacc_vx_i16m1(v819, v878, v875, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v880 = v19 + 488;
      const uint8_t* v881 = (const uint8_t*) v880;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v882 = __riscv_vle8_v_u8mf2(v881, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v883 = __riscv_vand_vx_u8mf2(v882, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v884 = __riscv_vreinterpret_v_u8mf2_i8mf2(v883);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v885 = v21 + 14;
      const int8_t* v886 = (const int8_t*) v885;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v887 = *(const int8_t *)(v886);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v888 = __riscv_vwmacc_vx_i16m1(v828, v887, v884, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v889 = __riscv_vsrl_vx_u8mf2(v882, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v890 = __riscv_vand_vx_u8mf2(v889, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v891 = __riscv_vreinterpret_v_u8mf2_i8mf2(v890);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v892 = v21 + 46;
      const int8_t* v893 = (const int8_t*) v892;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v894 = *(const int8_t *)(v893);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v895 = __riscv_vwmacc_vx_i16m1(v835, v894, v891, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v896 = __riscv_vsrl_vx_u8mf2(v882, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v897 = __riscv_vand_vx_u8mf2(v896, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v898 = __riscv_vreinterpret_v_u8mf2_i8mf2(v897);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v899 = v21 + 78;
      const int8_t* v900 = (const int8_t*) v899;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v901 = *(const int8_t *)(v900);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v902 = __riscv_vwmacc_vx_i16m1(v842, v901, v898, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v903 = __riscv_vsrl_vx_u8mf2(v882, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v904 = __riscv_vand_vx_u8mf2(v903, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v905 = __riscv_vreinterpret_v_u8mf2_i8mf2(v904);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v906 = v21 + 110;
      const int8_t* v907 = (const int8_t*) v906;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v908 = *(const int8_t *)(v907);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v909 = __riscv_vwmacc_vx_i16m1(v849, v908, v905, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v910 = v19 + 496;
      const uint8_t* v911 = (const uint8_t*) v910;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v912 = __riscv_vle8_v_u8mf2(v911, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v913 = __riscv_vand_vx_u8mf2(v912, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v914 = __riscv_vreinterpret_v_u8mf2_i8mf2(v913);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v915 = v21 + 15;
      const int8_t* v916 = (const int8_t*) v915;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v917 = *(const int8_t *)(v916);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v918 = __riscv_vwmacc_vx_i16m1(v858, v917, v914, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v919 = __riscv_vsrl_vx_u8mf2(v912, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v920 = __riscv_vand_vx_u8mf2(v919, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v921 = __riscv_vreinterpret_v_u8mf2_i8mf2(v920);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v922 = v21 + 47;
      const int8_t* v923 = (const int8_t*) v922;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v924 = *(const int8_t *)(v923);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v925 = __riscv_vwmacc_vx_i16m1(v865, v924, v921, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v926 = __riscv_vsrl_vx_u8mf2(v912, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v927 = __riscv_vand_vx_u8mf2(v926, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v928 = __riscv_vreinterpret_v_u8mf2_i8mf2(v927);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v929 = v21 + 79;
      const int8_t* v930 = (const int8_t*) v929;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v931 = *(const int8_t *)(v930);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v932 = __riscv_vwmacc_vx_i16m1(v872, v931, v928, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v933 = __riscv_vsrl_vx_u8mf2(v912, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v934 = __riscv_vand_vx_u8mf2(v933, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v935 = __riscv_vreinterpret_v_u8mf2_i8mf2(v934);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v936 = v21 + 111;
      const int8_t* v937 = (const int8_t*) v936;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v938 = *(const int8_t *)(v937);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v939 = __riscv_vwmacc_vx_i16m1(v879, v938, v935, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v940 = v19 + 504;
      const uint8_t* v941 = (const uint8_t*) v940;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v942 = __riscv_vle8_v_u8mf2(v941, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v943 = __riscv_vand_vx_u8mf2(v942, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v944 = __riscv_vreinterpret_v_u8mf2_i8mf2(v943);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v945 = v21 + 15;
      const int8_t* v946 = (const int8_t*) v945;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v947 = *(const int8_t *)(v946);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v948 = __riscv_vwmacc_vx_i16m1(v888, v947, v944, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v949 = __riscv_vsrl_vx_u8mf2(v942, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v950 = __riscv_vand_vx_u8mf2(v949, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v951 = __riscv_vreinterpret_v_u8mf2_i8mf2(v950);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v952 = v21 + 47;
      const int8_t* v953 = (const int8_t*) v952;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v954 = *(const int8_t *)(v953);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v955 = __riscv_vwmacc_vx_i16m1(v895, v954, v951, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v956 = __riscv_vsrl_vx_u8mf2(v942, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v957 = __riscv_vand_vx_u8mf2(v956, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v958 = __riscv_vreinterpret_v_u8mf2_i8mf2(v957);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v959 = v21 + 79;
      const int8_t* v960 = (const int8_t*) v959;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v961 = *(const int8_t *)(v960);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v962 = __riscv_vwmacc_vx_i16m1(v902, v961, v958, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v963 = __riscv_vsrl_vx_u8mf2(v942, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v964 = __riscv_vand_vx_u8mf2(v963, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v965 = __riscv_vreinterpret_v_u8mf2_i8mf2(v964);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v966 = v21 + 111;
      const int8_t* v967 = (const int8_t*) v966;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v968 = *(const int8_t *)(v967);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v969 = __riscv_vwmacc_vx_i16m1(v909, v968, v965, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v970 = v19 + 512;
      const uint8_t* v971 = (const uint8_t*) v970;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v972 = __riscv_vle8_v_u8mf2(v971, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v973 = __riscv_vand_vx_u8mf2(v972, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v974 = __riscv_vreinterpret_v_u8mf2_i8mf2(v973);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v975 = v21 + 16;
      const int8_t* v976 = (const int8_t*) v975;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v977 = *(const int8_t *)(v976);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v978 = __riscv_vwmacc_vx_i16m1(v918, v977, v974, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v979 = __riscv_vsrl_vx_u8mf2(v972, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v980 = __riscv_vand_vx_u8mf2(v979, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v981 = __riscv_vreinterpret_v_u8mf2_i8mf2(v980);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v982 = v21 + 48;
      const int8_t* v983 = (const int8_t*) v982;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v984 = *(const int8_t *)(v983);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v985 = __riscv_vwmacc_vx_i16m1(v925, v984, v981, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v986 = __riscv_vsrl_vx_u8mf2(v972, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v987 = __riscv_vand_vx_u8mf2(v986, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v988 = __riscv_vreinterpret_v_u8mf2_i8mf2(v987);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v989 = v21 + 80;
      const int8_t* v990 = (const int8_t*) v989;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v991 = *(const int8_t *)(v990);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v992 = __riscv_vwmacc_vx_i16m1(v932, v991, v988, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v993 = __riscv_vsrl_vx_u8mf2(v972, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v994 = __riscv_vand_vx_u8mf2(v993, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v995 = __riscv_vreinterpret_v_u8mf2_i8mf2(v994);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v996 = v21 + 112;
      const int8_t* v997 = (const int8_t*) v996;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v998 = *(const int8_t *)(v997);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v999 = __riscv_vwmacc_vx_i16m1(v939, v998, v995, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1000 = v19 + 520;
      const uint8_t* v1001 = (const uint8_t*) v1000;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1002 = __riscv_vle8_v_u8mf2(v1001, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1003 = __riscv_vand_vx_u8mf2(v1002, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1004 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1003);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1005 = v21 + 16;
      const int8_t* v1006 = (const int8_t*) v1005;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1007 = *(const int8_t *)(v1006);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1008 = __riscv_vwmacc_vx_i16m1(v948, v1007, v1004, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1009 = __riscv_vsrl_vx_u8mf2(v1002, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1010 = __riscv_vand_vx_u8mf2(v1009, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1011 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1010);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1012 = v21 + 48;
      const int8_t* v1013 = (const int8_t*) v1012;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1014 = *(const int8_t *)(v1013);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1015 = __riscv_vwmacc_vx_i16m1(v955, v1014, v1011, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1016 = __riscv_vsrl_vx_u8mf2(v1002, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1017 = __riscv_vand_vx_u8mf2(v1016, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1018 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1017);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1019 = v21 + 80;
      const int8_t* v1020 = (const int8_t*) v1019;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1021 = *(const int8_t *)(v1020);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1022 = __riscv_vwmacc_vx_i16m1(v962, v1021, v1018, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1023 = __riscv_vsrl_vx_u8mf2(v1002, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1024 = __riscv_vand_vx_u8mf2(v1023, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1025 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1024);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1026 = v21 + 112;
      const int8_t* v1027 = (const int8_t*) v1026;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1028 = *(const int8_t *)(v1027);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1029 = __riscv_vwmacc_vx_i16m1(v969, v1028, v1025, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1030 = v19 + 528;
      const uint8_t* v1031 = (const uint8_t*) v1030;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1032 = __riscv_vle8_v_u8mf2(v1031, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1033 = __riscv_vand_vx_u8mf2(v1032, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1034 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1033);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1035 = v21 + 17;
      const int8_t* v1036 = (const int8_t*) v1035;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1037 = *(const int8_t *)(v1036);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1038 = __riscv_vwmacc_vx_i16m1(v978, v1037, v1034, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1039 = __riscv_vsrl_vx_u8mf2(v1032, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1040 = __riscv_vand_vx_u8mf2(v1039, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1041 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1040);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1042 = v21 + 49;
      const int8_t* v1043 = (const int8_t*) v1042;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1044 = *(const int8_t *)(v1043);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1045 = __riscv_vwmacc_vx_i16m1(v985, v1044, v1041, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1046 = __riscv_vsrl_vx_u8mf2(v1032, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1047 = __riscv_vand_vx_u8mf2(v1046, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1048 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1047);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1049 = v21 + 81;
      const int8_t* v1050 = (const int8_t*) v1049;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1051 = *(const int8_t *)(v1050);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1052 = __riscv_vwmacc_vx_i16m1(v992, v1051, v1048, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1053 = __riscv_vsrl_vx_u8mf2(v1032, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1054 = __riscv_vand_vx_u8mf2(v1053, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1055 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1054);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1056 = v21 + 113;
      const int8_t* v1057 = (const int8_t*) v1056;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1058 = *(const int8_t *)(v1057);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1059 = __riscv_vwmacc_vx_i16m1(v999, v1058, v1055, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1060 = v19 + 536;
      const uint8_t* v1061 = (const uint8_t*) v1060;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1062 = __riscv_vle8_v_u8mf2(v1061, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1063 = __riscv_vand_vx_u8mf2(v1062, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1064 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1063);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1065 = v21 + 17;
      const int8_t* v1066 = (const int8_t*) v1065;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1067 = *(const int8_t *)(v1066);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1068 = __riscv_vwmacc_vx_i16m1(v1008, v1067, v1064, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1069 = __riscv_vsrl_vx_u8mf2(v1062, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1070 = __riscv_vand_vx_u8mf2(v1069, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1071 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1070);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1072 = v21 + 49;
      const int8_t* v1073 = (const int8_t*) v1072;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1074 = *(const int8_t *)(v1073);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1075 = __riscv_vwmacc_vx_i16m1(v1015, v1074, v1071, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1076 = __riscv_vsrl_vx_u8mf2(v1062, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1077 = __riscv_vand_vx_u8mf2(v1076, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1078 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1077);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1079 = v21 + 81;
      const int8_t* v1080 = (const int8_t*) v1079;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1081 = *(const int8_t *)(v1080);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1082 = __riscv_vwmacc_vx_i16m1(v1022, v1081, v1078, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1083 = __riscv_vsrl_vx_u8mf2(v1062, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1084 = __riscv_vand_vx_u8mf2(v1083, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1085 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1084);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1086 = v21 + 113;
      const int8_t* v1087 = (const int8_t*) v1086;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1088 = *(const int8_t *)(v1087);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1089 = __riscv_vwmacc_vx_i16m1(v1029, v1088, v1085, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1090 = v19 + 544;
      const uint8_t* v1091 = (const uint8_t*) v1090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1092 = __riscv_vle8_v_u8mf2(v1091, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1093 = __riscv_vand_vx_u8mf2(v1092, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1094 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1093);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1095 = v21 + 18;
      const int8_t* v1096 = (const int8_t*) v1095;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1097 = *(const int8_t *)(v1096);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1098 = __riscv_vwmacc_vx_i16m1(v1038, v1097, v1094, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1099 = __riscv_vsrl_vx_u8mf2(v1092, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1100 = __riscv_vand_vx_u8mf2(v1099, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1101 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1100);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1102 = v21 + 50;
      const int8_t* v1103 = (const int8_t*) v1102;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1104 = *(const int8_t *)(v1103);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1105 = __riscv_vwmacc_vx_i16m1(v1045, v1104, v1101, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1106 = __riscv_vsrl_vx_u8mf2(v1092, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1107 = __riscv_vand_vx_u8mf2(v1106, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1108 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1107);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1109 = v21 + 82;
      const int8_t* v1110 = (const int8_t*) v1109;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1111 = *(const int8_t *)(v1110);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1112 = __riscv_vwmacc_vx_i16m1(v1052, v1111, v1108, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1113 = __riscv_vsrl_vx_u8mf2(v1092, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1114 = __riscv_vand_vx_u8mf2(v1113, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1115 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1114);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1116 = v21 + 114;
      const int8_t* v1117 = (const int8_t*) v1116;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1118 = *(const int8_t *)(v1117);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1119 = __riscv_vwmacc_vx_i16m1(v1059, v1118, v1115, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1120 = v19 + 552;
      const uint8_t* v1121 = (const uint8_t*) v1120;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1122 = __riscv_vle8_v_u8mf2(v1121, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1123 = __riscv_vand_vx_u8mf2(v1122, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1124 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1123);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1125 = v21 + 18;
      const int8_t* v1126 = (const int8_t*) v1125;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1127 = *(const int8_t *)(v1126);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1128 = __riscv_vwmacc_vx_i16m1(v1068, v1127, v1124, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1129 = __riscv_vsrl_vx_u8mf2(v1122, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1130 = __riscv_vand_vx_u8mf2(v1129, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1131 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1130);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1132 = v21 + 50;
      const int8_t* v1133 = (const int8_t*) v1132;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1134 = *(const int8_t *)(v1133);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1135 = __riscv_vwmacc_vx_i16m1(v1075, v1134, v1131, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1136 = __riscv_vsrl_vx_u8mf2(v1122, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1137 = __riscv_vand_vx_u8mf2(v1136, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1138 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1137);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1139 = v21 + 82;
      const int8_t* v1140 = (const int8_t*) v1139;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1141 = *(const int8_t *)(v1140);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1142 = __riscv_vwmacc_vx_i16m1(v1082, v1141, v1138, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1143 = __riscv_vsrl_vx_u8mf2(v1122, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1144 = __riscv_vand_vx_u8mf2(v1143, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1145 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1144);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1146 = v21 + 114;
      const int8_t* v1147 = (const int8_t*) v1146;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1148 = *(const int8_t *)(v1147);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1149 = __riscv_vwmacc_vx_i16m1(v1089, v1148, v1145, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1150 = v19 + 560;
      const uint8_t* v1151 = (const uint8_t*) v1150;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1152 = __riscv_vle8_v_u8mf2(v1151, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1153 = __riscv_vand_vx_u8mf2(v1152, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1154 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1153);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1155 = v21 + 19;
      const int8_t* v1156 = (const int8_t*) v1155;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1157 = *(const int8_t *)(v1156);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1158 = __riscv_vwmacc_vx_i16m1(v1098, v1157, v1154, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1159 = __riscv_vsrl_vx_u8mf2(v1152, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1160 = __riscv_vand_vx_u8mf2(v1159, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1161 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1160);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1162 = v21 + 51;
      const int8_t* v1163 = (const int8_t*) v1162;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1164 = *(const int8_t *)(v1163);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1165 = __riscv_vwmacc_vx_i16m1(v1105, v1164, v1161, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1166 = __riscv_vsrl_vx_u8mf2(v1152, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1167 = __riscv_vand_vx_u8mf2(v1166, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1168 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1167);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1169 = v21 + 83;
      const int8_t* v1170 = (const int8_t*) v1169;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1171 = *(const int8_t *)(v1170);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1172 = __riscv_vwmacc_vx_i16m1(v1112, v1171, v1168, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1173 = __riscv_vsrl_vx_u8mf2(v1152, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1174 = __riscv_vand_vx_u8mf2(v1173, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1175 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1174);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1176 = v21 + 115;
      const int8_t* v1177 = (const int8_t*) v1176;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1178 = *(const int8_t *)(v1177);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1179 = __riscv_vwmacc_vx_i16m1(v1119, v1178, v1175, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1180 = v19 + 568;
      const uint8_t* v1181 = (const uint8_t*) v1180;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1182 = __riscv_vle8_v_u8mf2(v1181, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1183 = __riscv_vand_vx_u8mf2(v1182, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1184 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1183);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1185 = v21 + 19;
      const int8_t* v1186 = (const int8_t*) v1185;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1187 = *(const int8_t *)(v1186);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1188 = __riscv_vwmacc_vx_i16m1(v1128, v1187, v1184, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1189 = __riscv_vsrl_vx_u8mf2(v1182, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1190 = __riscv_vand_vx_u8mf2(v1189, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1191 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1190);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1192 = v21 + 51;
      const int8_t* v1193 = (const int8_t*) v1192;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1194 = *(const int8_t *)(v1193);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1195 = __riscv_vwmacc_vx_i16m1(v1135, v1194, v1191, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1196 = __riscv_vsrl_vx_u8mf2(v1182, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1197 = __riscv_vand_vx_u8mf2(v1196, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1198 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1197);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1199 = v21 + 83;
      const int8_t* v1200 = (const int8_t*) v1199;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1201 = *(const int8_t *)(v1200);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1202 = __riscv_vwmacc_vx_i16m1(v1142, v1201, v1198, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1203 = __riscv_vsrl_vx_u8mf2(v1182, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1204 = __riscv_vand_vx_u8mf2(v1203, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1205 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1204);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1206 = v21 + 115;
      const int8_t* v1207 = (const int8_t*) v1206;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1208 = *(const int8_t *)(v1207);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1209 = __riscv_vwmacc_vx_i16m1(v1149, v1208, v1205, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v1210 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1211 = __riscv_vwmacc_vv_i32m2(v1210, v48, v1158, 8);
      v34 = v1211;
      vint32m2_t v1212 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1213 = __riscv_vwmacc_vv_i32m2(v1212, v66, v1165, 8);
      v34 = v1213;
      vint32m2_t v1214 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1215 = __riscv_vwmacc_vv_i32m2(v1214, v84, v1172, 8);
      v34 = v1215;
      vint32m2_t v1216 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1217 = __riscv_vwmacc_vv_i32m2(v1216, v102, v1179, 8);
      v34 = v1217;
      vint32m2_t v1218 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1219 = __riscv_vwmacc_vv_i32m2(v1218, v120, v1188, 8);
      v38 = v1219;
      vint32m2_t v1220 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1221 = __riscv_vwmacc_vv_i32m2(v1220, v138, v1195, 8);
      v38 = v1221;
      vint32m2_t v1222 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1223 = __riscv_vwmacc_vv_i32m2(v1222, v156, v1202, 8);
      v38 = v1223;
      vint32m2_t v1224 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1225 = __riscv_vwmacc_vv_i32m2(v1224, v174, v1209, 8);
      v38 = v1225;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1226 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1227 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1228 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1229 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1230 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1231 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1232 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1233 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1234 = v19 + 576;
      const uint8_t* v1235 = (const uint8_t*) v1234;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1236 = __riscv_vle8_v_u8mf2(v1235, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1237 = __riscv_vand_vx_u8mf2(v1236, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1238 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1237);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1239 = v21 + 20;
      const int8_t* v1240 = (const int8_t*) v1239;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1241 = *(const int8_t *)(v1240);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1242 = __riscv_vwmacc_vx_i16m1(v1226, v1241, v1238, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1243 = __riscv_vsrl_vx_u8mf2(v1236, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1244 = __riscv_vand_vx_u8mf2(v1243, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1245 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1244);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1246 = v21 + 52;
      const int8_t* v1247 = (const int8_t*) v1246;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1248 = *(const int8_t *)(v1247);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1249 = __riscv_vwmacc_vx_i16m1(v1227, v1248, v1245, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1250 = __riscv_vsrl_vx_u8mf2(v1236, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1251 = __riscv_vand_vx_u8mf2(v1250, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1252 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1251);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1253 = v21 + 84;
      const int8_t* v1254 = (const int8_t*) v1253;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1255 = *(const int8_t *)(v1254);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1256 = __riscv_vwmacc_vx_i16m1(v1228, v1255, v1252, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1257 = __riscv_vsrl_vx_u8mf2(v1236, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1258 = __riscv_vand_vx_u8mf2(v1257, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1259 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1258);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1260 = v21 + 116;
      const int8_t* v1261 = (const int8_t*) v1260;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1262 = *(const int8_t *)(v1261);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1263 = __riscv_vwmacc_vx_i16m1(v1229, v1262, v1259, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1264 = v19 + 584;
      const uint8_t* v1265 = (const uint8_t*) v1264;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1266 = __riscv_vle8_v_u8mf2(v1265, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1267 = __riscv_vand_vx_u8mf2(v1266, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1268 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1267);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1269 = v21 + 20;
      const int8_t* v1270 = (const int8_t*) v1269;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1271 = *(const int8_t *)(v1270);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1272 = __riscv_vwmacc_vx_i16m1(v1230, v1271, v1268, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1273 = __riscv_vsrl_vx_u8mf2(v1266, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1274 = __riscv_vand_vx_u8mf2(v1273, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1275 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1274);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1276 = v21 + 52;
      const int8_t* v1277 = (const int8_t*) v1276;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1278 = *(const int8_t *)(v1277);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1279 = __riscv_vwmacc_vx_i16m1(v1231, v1278, v1275, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1280 = __riscv_vsrl_vx_u8mf2(v1266, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1281 = __riscv_vand_vx_u8mf2(v1280, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1282 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1281);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1283 = v21 + 84;
      const int8_t* v1284 = (const int8_t*) v1283;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1285 = *(const int8_t *)(v1284);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1286 = __riscv_vwmacc_vx_i16m1(v1232, v1285, v1282, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1287 = __riscv_vsrl_vx_u8mf2(v1266, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1288 = __riscv_vand_vx_u8mf2(v1287, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1289 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1288);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1290 = v21 + 116;
      const int8_t* v1291 = (const int8_t*) v1290;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1292 = *(const int8_t *)(v1291);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1293 = __riscv_vwmacc_vx_i16m1(v1233, v1292, v1289, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1294 = v19 + 592;
      const uint8_t* v1295 = (const uint8_t*) v1294;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1296 = __riscv_vle8_v_u8mf2(v1295, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1297 = __riscv_vand_vx_u8mf2(v1296, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1298 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1297);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1299 = v21 + 21;
      const int8_t* v1300 = (const int8_t*) v1299;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1301 = *(const int8_t *)(v1300);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1302 = __riscv_vwmacc_vx_i16m1(v1242, v1301, v1298, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1303 = __riscv_vsrl_vx_u8mf2(v1296, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1304 = __riscv_vand_vx_u8mf2(v1303, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1305 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1304);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1306 = v21 + 53;
      const int8_t* v1307 = (const int8_t*) v1306;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1308 = *(const int8_t *)(v1307);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1309 = __riscv_vwmacc_vx_i16m1(v1249, v1308, v1305, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1310 = __riscv_vsrl_vx_u8mf2(v1296, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1311 = __riscv_vand_vx_u8mf2(v1310, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1312 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1311);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1313 = v21 + 85;
      const int8_t* v1314 = (const int8_t*) v1313;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1315 = *(const int8_t *)(v1314);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1316 = __riscv_vwmacc_vx_i16m1(v1256, v1315, v1312, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1317 = __riscv_vsrl_vx_u8mf2(v1296, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1318 = __riscv_vand_vx_u8mf2(v1317, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1319 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1318);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1320 = v21 + 117;
      const int8_t* v1321 = (const int8_t*) v1320;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1322 = *(const int8_t *)(v1321);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1323 = __riscv_vwmacc_vx_i16m1(v1263, v1322, v1319, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1324 = v19 + 600;
      const uint8_t* v1325 = (const uint8_t*) v1324;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1326 = __riscv_vle8_v_u8mf2(v1325, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1327 = __riscv_vand_vx_u8mf2(v1326, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1328 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1327);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1329 = v21 + 21;
      const int8_t* v1330 = (const int8_t*) v1329;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1331 = *(const int8_t *)(v1330);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1332 = __riscv_vwmacc_vx_i16m1(v1272, v1331, v1328, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1333 = __riscv_vsrl_vx_u8mf2(v1326, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1334 = __riscv_vand_vx_u8mf2(v1333, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1335 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1334);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1336 = v21 + 53;
      const int8_t* v1337 = (const int8_t*) v1336;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1338 = *(const int8_t *)(v1337);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1339 = __riscv_vwmacc_vx_i16m1(v1279, v1338, v1335, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1340 = __riscv_vsrl_vx_u8mf2(v1326, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1341 = __riscv_vand_vx_u8mf2(v1340, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1342 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1341);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1343 = v21 + 85;
      const int8_t* v1344 = (const int8_t*) v1343;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1345 = *(const int8_t *)(v1344);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1346 = __riscv_vwmacc_vx_i16m1(v1286, v1345, v1342, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1347 = __riscv_vsrl_vx_u8mf2(v1326, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1348 = __riscv_vand_vx_u8mf2(v1347, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1349 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1348);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1350 = v21 + 117;
      const int8_t* v1351 = (const int8_t*) v1350;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1352 = *(const int8_t *)(v1351);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1353 = __riscv_vwmacc_vx_i16m1(v1293, v1352, v1349, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1354 = v19 + 608;
      const uint8_t* v1355 = (const uint8_t*) v1354;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1356 = __riscv_vle8_v_u8mf2(v1355, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1357 = __riscv_vand_vx_u8mf2(v1356, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1358 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1357);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1359 = v21 + 22;
      const int8_t* v1360 = (const int8_t*) v1359;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1361 = *(const int8_t *)(v1360);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1362 = __riscv_vwmacc_vx_i16m1(v1302, v1361, v1358, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1363 = __riscv_vsrl_vx_u8mf2(v1356, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1364 = __riscv_vand_vx_u8mf2(v1363, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1365 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1364);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1366 = v21 + 54;
      const int8_t* v1367 = (const int8_t*) v1366;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1368 = *(const int8_t *)(v1367);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1369 = __riscv_vwmacc_vx_i16m1(v1309, v1368, v1365, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1370 = __riscv_vsrl_vx_u8mf2(v1356, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1371 = __riscv_vand_vx_u8mf2(v1370, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1372 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1371);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1373 = v21 + 86;
      const int8_t* v1374 = (const int8_t*) v1373;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1375 = *(const int8_t *)(v1374);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1376 = __riscv_vwmacc_vx_i16m1(v1316, v1375, v1372, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1377 = __riscv_vsrl_vx_u8mf2(v1356, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1378 = __riscv_vand_vx_u8mf2(v1377, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1379 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1378);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1380 = v21 + 118;
      const int8_t* v1381 = (const int8_t*) v1380;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1382 = *(const int8_t *)(v1381);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1383 = __riscv_vwmacc_vx_i16m1(v1323, v1382, v1379, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1384 = v19 + 616;
      const uint8_t* v1385 = (const uint8_t*) v1384;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1386 = __riscv_vle8_v_u8mf2(v1385, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1387 = __riscv_vand_vx_u8mf2(v1386, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1388 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1387);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1389 = v21 + 22;
      const int8_t* v1390 = (const int8_t*) v1389;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1391 = *(const int8_t *)(v1390);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1392 = __riscv_vwmacc_vx_i16m1(v1332, v1391, v1388, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1393 = __riscv_vsrl_vx_u8mf2(v1386, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1394 = __riscv_vand_vx_u8mf2(v1393, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1395 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1394);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1396 = v21 + 54;
      const int8_t* v1397 = (const int8_t*) v1396;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1398 = *(const int8_t *)(v1397);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1399 = __riscv_vwmacc_vx_i16m1(v1339, v1398, v1395, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1400 = __riscv_vsrl_vx_u8mf2(v1386, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1401 = __riscv_vand_vx_u8mf2(v1400, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1402 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1401);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1403 = v21 + 86;
      const int8_t* v1404 = (const int8_t*) v1403;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1405 = *(const int8_t *)(v1404);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1406 = __riscv_vwmacc_vx_i16m1(v1346, v1405, v1402, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1407 = __riscv_vsrl_vx_u8mf2(v1386, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1408 = __riscv_vand_vx_u8mf2(v1407, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1409 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1408);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1410 = v21 + 118;
      const int8_t* v1411 = (const int8_t*) v1410;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1412 = *(const int8_t *)(v1411);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1413 = __riscv_vwmacc_vx_i16m1(v1353, v1412, v1409, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1414 = v19 + 624;
      const uint8_t* v1415 = (const uint8_t*) v1414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1416 = __riscv_vle8_v_u8mf2(v1415, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1417 = __riscv_vand_vx_u8mf2(v1416, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1418 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1417);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1419 = v21 + 23;
      const int8_t* v1420 = (const int8_t*) v1419;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1421 = *(const int8_t *)(v1420);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1422 = __riscv_vwmacc_vx_i16m1(v1362, v1421, v1418, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1423 = __riscv_vsrl_vx_u8mf2(v1416, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1424 = __riscv_vand_vx_u8mf2(v1423, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1425 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1424);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1426 = v21 + 55;
      const int8_t* v1427 = (const int8_t*) v1426;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1428 = *(const int8_t *)(v1427);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1429 = __riscv_vwmacc_vx_i16m1(v1369, v1428, v1425, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1430 = __riscv_vsrl_vx_u8mf2(v1416, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1431 = __riscv_vand_vx_u8mf2(v1430, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1432 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1431);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1433 = v21 + 87;
      const int8_t* v1434 = (const int8_t*) v1433;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1435 = *(const int8_t *)(v1434);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1436 = __riscv_vwmacc_vx_i16m1(v1376, v1435, v1432, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1437 = __riscv_vsrl_vx_u8mf2(v1416, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1438 = __riscv_vand_vx_u8mf2(v1437, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1439 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1438);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1440 = v21 + 119;
      const int8_t* v1441 = (const int8_t*) v1440;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1442 = *(const int8_t *)(v1441);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1443 = __riscv_vwmacc_vx_i16m1(v1383, v1442, v1439, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1444 = v19 + 632;
      const uint8_t* v1445 = (const uint8_t*) v1444;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1446 = __riscv_vle8_v_u8mf2(v1445, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1447 = __riscv_vand_vx_u8mf2(v1446, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1448 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1447);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1449 = v21 + 23;
      const int8_t* v1450 = (const int8_t*) v1449;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1451 = *(const int8_t *)(v1450);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1452 = __riscv_vwmacc_vx_i16m1(v1392, v1451, v1448, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1453 = __riscv_vsrl_vx_u8mf2(v1446, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1454 = __riscv_vand_vx_u8mf2(v1453, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1455 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1454);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1456 = v21 + 55;
      const int8_t* v1457 = (const int8_t*) v1456;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1458 = *(const int8_t *)(v1457);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1459 = __riscv_vwmacc_vx_i16m1(v1399, v1458, v1455, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1460 = __riscv_vsrl_vx_u8mf2(v1446, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1461 = __riscv_vand_vx_u8mf2(v1460, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1462 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1461);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1463 = v21 + 87;
      const int8_t* v1464 = (const int8_t*) v1463;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1465 = *(const int8_t *)(v1464);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1466 = __riscv_vwmacc_vx_i16m1(v1406, v1465, v1462, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1467 = __riscv_vsrl_vx_u8mf2(v1446, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1468 = __riscv_vand_vx_u8mf2(v1467, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1469 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1468);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1470 = v21 + 119;
      const int8_t* v1471 = (const int8_t*) v1470;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1472 = *(const int8_t *)(v1471);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1473 = __riscv_vwmacc_vx_i16m1(v1413, v1472, v1469, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1474 = v19 + 640;
      const uint8_t* v1475 = (const uint8_t*) v1474;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1476 = __riscv_vle8_v_u8mf2(v1475, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1477 = __riscv_vand_vx_u8mf2(v1476, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1478 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1477);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1479 = v21 + 24;
      const int8_t* v1480 = (const int8_t*) v1479;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1481 = *(const int8_t *)(v1480);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1482 = __riscv_vwmacc_vx_i16m1(v1422, v1481, v1478, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1483 = __riscv_vsrl_vx_u8mf2(v1476, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1484 = __riscv_vand_vx_u8mf2(v1483, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1485 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1484);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1486 = v21 + 56;
      const int8_t* v1487 = (const int8_t*) v1486;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1488 = *(const int8_t *)(v1487);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1489 = __riscv_vwmacc_vx_i16m1(v1429, v1488, v1485, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1490 = __riscv_vsrl_vx_u8mf2(v1476, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1491 = __riscv_vand_vx_u8mf2(v1490, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1492 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1491);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1493 = v21 + 88;
      const int8_t* v1494 = (const int8_t*) v1493;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1495 = *(const int8_t *)(v1494);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1496 = __riscv_vwmacc_vx_i16m1(v1436, v1495, v1492, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1497 = __riscv_vsrl_vx_u8mf2(v1476, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1498 = __riscv_vand_vx_u8mf2(v1497, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1499 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1498);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1500 = v21 + 120;
      const int8_t* v1501 = (const int8_t*) v1500;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1502 = *(const int8_t *)(v1501);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1503 = __riscv_vwmacc_vx_i16m1(v1443, v1502, v1499, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1504 = v19 + 648;
      const uint8_t* v1505 = (const uint8_t*) v1504;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1506 = __riscv_vle8_v_u8mf2(v1505, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1507 = __riscv_vand_vx_u8mf2(v1506, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1508 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1507);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1509 = v21 + 24;
      const int8_t* v1510 = (const int8_t*) v1509;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1511 = *(const int8_t *)(v1510);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1512 = __riscv_vwmacc_vx_i16m1(v1452, v1511, v1508, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1513 = __riscv_vsrl_vx_u8mf2(v1506, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1514 = __riscv_vand_vx_u8mf2(v1513, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1515 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1514);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1516 = v21 + 56;
      const int8_t* v1517 = (const int8_t*) v1516;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1518 = *(const int8_t *)(v1517);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1519 = __riscv_vwmacc_vx_i16m1(v1459, v1518, v1515, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1520 = __riscv_vsrl_vx_u8mf2(v1506, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1521 = __riscv_vand_vx_u8mf2(v1520, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1522 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1521);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1523 = v21 + 88;
      const int8_t* v1524 = (const int8_t*) v1523;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1525 = *(const int8_t *)(v1524);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1526 = __riscv_vwmacc_vx_i16m1(v1466, v1525, v1522, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1527 = __riscv_vsrl_vx_u8mf2(v1506, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1528 = __riscv_vand_vx_u8mf2(v1527, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1529 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1528);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1530 = v21 + 120;
      const int8_t* v1531 = (const int8_t*) v1530;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1532 = *(const int8_t *)(v1531);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1533 = __riscv_vwmacc_vx_i16m1(v1473, v1532, v1529, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1534 = v19 + 656;
      const uint8_t* v1535 = (const uint8_t*) v1534;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1536 = __riscv_vle8_v_u8mf2(v1535, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1537 = __riscv_vand_vx_u8mf2(v1536, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1538 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1537);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1539 = v21 + 25;
      const int8_t* v1540 = (const int8_t*) v1539;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1541 = *(const int8_t *)(v1540);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1542 = __riscv_vwmacc_vx_i16m1(v1482, v1541, v1538, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1543 = __riscv_vsrl_vx_u8mf2(v1536, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1544 = __riscv_vand_vx_u8mf2(v1543, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1545 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1544);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1546 = v21 + 57;
      const int8_t* v1547 = (const int8_t*) v1546;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1548 = *(const int8_t *)(v1547);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1549 = __riscv_vwmacc_vx_i16m1(v1489, v1548, v1545, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1550 = __riscv_vsrl_vx_u8mf2(v1536, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1551 = __riscv_vand_vx_u8mf2(v1550, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1552 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1551);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1553 = v21 + 89;
      const int8_t* v1554 = (const int8_t*) v1553;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1555 = *(const int8_t *)(v1554);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1556 = __riscv_vwmacc_vx_i16m1(v1496, v1555, v1552, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1557 = __riscv_vsrl_vx_u8mf2(v1536, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1558 = __riscv_vand_vx_u8mf2(v1557, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1559 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1558);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1560 = v21 + 121;
      const int8_t* v1561 = (const int8_t*) v1560;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1562 = *(const int8_t *)(v1561);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1563 = __riscv_vwmacc_vx_i16m1(v1503, v1562, v1559, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1564 = v19 + 664;
      const uint8_t* v1565 = (const uint8_t*) v1564;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1566 = __riscv_vle8_v_u8mf2(v1565, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1567 = __riscv_vand_vx_u8mf2(v1566, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1568 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1567);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1569 = v21 + 25;
      const int8_t* v1570 = (const int8_t*) v1569;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1571 = *(const int8_t *)(v1570);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1572 = __riscv_vwmacc_vx_i16m1(v1512, v1571, v1568, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1573 = __riscv_vsrl_vx_u8mf2(v1566, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1574 = __riscv_vand_vx_u8mf2(v1573, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1575 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1574);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1576 = v21 + 57;
      const int8_t* v1577 = (const int8_t*) v1576;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1578 = *(const int8_t *)(v1577);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1579 = __riscv_vwmacc_vx_i16m1(v1519, v1578, v1575, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1580 = __riscv_vsrl_vx_u8mf2(v1566, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1581 = __riscv_vand_vx_u8mf2(v1580, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1582 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1581);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1583 = v21 + 89;
      const int8_t* v1584 = (const int8_t*) v1583;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1585 = *(const int8_t *)(v1584);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1586 = __riscv_vwmacc_vx_i16m1(v1526, v1585, v1582, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1587 = __riscv_vsrl_vx_u8mf2(v1566, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1588 = __riscv_vand_vx_u8mf2(v1587, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1589 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1588);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1590 = v21 + 121;
      const int8_t* v1591 = (const int8_t*) v1590;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1592 = *(const int8_t *)(v1591);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1593 = __riscv_vwmacc_vx_i16m1(v1533, v1592, v1589, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1594 = v19 + 672;
      const uint8_t* v1595 = (const uint8_t*) v1594;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1596 = __riscv_vle8_v_u8mf2(v1595, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1597 = __riscv_vand_vx_u8mf2(v1596, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1598 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1597);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1599 = v21 + 26;
      const int8_t* v1600 = (const int8_t*) v1599;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1601 = *(const int8_t *)(v1600);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1602 = __riscv_vwmacc_vx_i16m1(v1542, v1601, v1598, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1603 = __riscv_vsrl_vx_u8mf2(v1596, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1604 = __riscv_vand_vx_u8mf2(v1603, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1605 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1604);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1606 = v21 + 58;
      const int8_t* v1607 = (const int8_t*) v1606;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1608 = *(const int8_t *)(v1607);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1609 = __riscv_vwmacc_vx_i16m1(v1549, v1608, v1605, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1610 = __riscv_vsrl_vx_u8mf2(v1596, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1611 = __riscv_vand_vx_u8mf2(v1610, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1612 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1611);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1613 = v21 + 90;
      const int8_t* v1614 = (const int8_t*) v1613;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1615 = *(const int8_t *)(v1614);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1616 = __riscv_vwmacc_vx_i16m1(v1556, v1615, v1612, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1617 = __riscv_vsrl_vx_u8mf2(v1596, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1618 = __riscv_vand_vx_u8mf2(v1617, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1619 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1618);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1620 = v21 + 122;
      const int8_t* v1621 = (const int8_t*) v1620;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1622 = *(const int8_t *)(v1621);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1623 = __riscv_vwmacc_vx_i16m1(v1563, v1622, v1619, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1624 = v19 + 680;
      const uint8_t* v1625 = (const uint8_t*) v1624;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1626 = __riscv_vle8_v_u8mf2(v1625, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1627 = __riscv_vand_vx_u8mf2(v1626, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1628 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1627);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1629 = v21 + 26;
      const int8_t* v1630 = (const int8_t*) v1629;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1631 = *(const int8_t *)(v1630);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1632 = __riscv_vwmacc_vx_i16m1(v1572, v1631, v1628, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1633 = __riscv_vsrl_vx_u8mf2(v1626, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1634 = __riscv_vand_vx_u8mf2(v1633, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1635 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1634);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1636 = v21 + 58;
      const int8_t* v1637 = (const int8_t*) v1636;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1638 = *(const int8_t *)(v1637);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1639 = __riscv_vwmacc_vx_i16m1(v1579, v1638, v1635, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1640 = __riscv_vsrl_vx_u8mf2(v1626, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1641 = __riscv_vand_vx_u8mf2(v1640, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1642 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1641);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1643 = v21 + 90;
      const int8_t* v1644 = (const int8_t*) v1643;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1645 = *(const int8_t *)(v1644);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1646 = __riscv_vwmacc_vx_i16m1(v1586, v1645, v1642, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1647 = __riscv_vsrl_vx_u8mf2(v1626, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1648 = __riscv_vand_vx_u8mf2(v1647, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1649 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1648);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1650 = v21 + 122;
      const int8_t* v1651 = (const int8_t*) v1650;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1652 = *(const int8_t *)(v1651);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1653 = __riscv_vwmacc_vx_i16m1(v1593, v1652, v1649, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1654 = v19 + 688;
      const uint8_t* v1655 = (const uint8_t*) v1654;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1656 = __riscv_vle8_v_u8mf2(v1655, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1657 = __riscv_vand_vx_u8mf2(v1656, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1658 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1657);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1659 = v21 + 27;
      const int8_t* v1660 = (const int8_t*) v1659;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1661 = *(const int8_t *)(v1660);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1662 = __riscv_vwmacc_vx_i16m1(v1602, v1661, v1658, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1663 = __riscv_vsrl_vx_u8mf2(v1656, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1664 = __riscv_vand_vx_u8mf2(v1663, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1665 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1664);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1666 = v21 + 59;
      const int8_t* v1667 = (const int8_t*) v1666;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1668 = *(const int8_t *)(v1667);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1669 = __riscv_vwmacc_vx_i16m1(v1609, v1668, v1665, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1670 = __riscv_vsrl_vx_u8mf2(v1656, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1671 = __riscv_vand_vx_u8mf2(v1670, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1672 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1671);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1673 = v21 + 91;
      const int8_t* v1674 = (const int8_t*) v1673;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1675 = *(const int8_t *)(v1674);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1676 = __riscv_vwmacc_vx_i16m1(v1616, v1675, v1672, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1677 = __riscv_vsrl_vx_u8mf2(v1656, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1678 = __riscv_vand_vx_u8mf2(v1677, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1679 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1678);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1680 = v21 + 123;
      const int8_t* v1681 = (const int8_t*) v1680;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1682 = *(const int8_t *)(v1681);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1683 = __riscv_vwmacc_vx_i16m1(v1623, v1682, v1679, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1684 = v19 + 696;
      const uint8_t* v1685 = (const uint8_t*) v1684;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1686 = __riscv_vle8_v_u8mf2(v1685, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1687 = __riscv_vand_vx_u8mf2(v1686, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1688 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1687);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1689 = v21 + 27;
      const int8_t* v1690 = (const int8_t*) v1689;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1691 = *(const int8_t *)(v1690);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1692 = __riscv_vwmacc_vx_i16m1(v1632, v1691, v1688, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1693 = __riscv_vsrl_vx_u8mf2(v1686, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1694 = __riscv_vand_vx_u8mf2(v1693, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1695 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1694);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1696 = v21 + 59;
      const int8_t* v1697 = (const int8_t*) v1696;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1698 = *(const int8_t *)(v1697);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1699 = __riscv_vwmacc_vx_i16m1(v1639, v1698, v1695, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1700 = __riscv_vsrl_vx_u8mf2(v1686, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1701 = __riscv_vand_vx_u8mf2(v1700, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1702 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1701);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1703 = v21 + 91;
      const int8_t* v1704 = (const int8_t*) v1703;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1705 = *(const int8_t *)(v1704);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1706 = __riscv_vwmacc_vx_i16m1(v1646, v1705, v1702, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1707 = __riscv_vsrl_vx_u8mf2(v1686, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1708 = __riscv_vand_vx_u8mf2(v1707, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1709 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1708);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1710 = v21 + 123;
      const int8_t* v1711 = (const int8_t*) v1710;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1712 = *(const int8_t *)(v1711);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1713 = __riscv_vwmacc_vx_i16m1(v1653, v1712, v1709, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1714 = v19 + 704;
      const uint8_t* v1715 = (const uint8_t*) v1714;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1716 = __riscv_vle8_v_u8mf2(v1715, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1717 = __riscv_vand_vx_u8mf2(v1716, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1718 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1717);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1719 = v21 + 28;
      const int8_t* v1720 = (const int8_t*) v1719;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1721 = *(const int8_t *)(v1720);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1722 = __riscv_vwmacc_vx_i16m1(v1662, v1721, v1718, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1723 = __riscv_vsrl_vx_u8mf2(v1716, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1724 = __riscv_vand_vx_u8mf2(v1723, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1725 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1724);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1726 = v21 + 60;
      const int8_t* v1727 = (const int8_t*) v1726;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1728 = *(const int8_t *)(v1727);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1729 = __riscv_vwmacc_vx_i16m1(v1669, v1728, v1725, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1730 = __riscv_vsrl_vx_u8mf2(v1716, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1731 = __riscv_vand_vx_u8mf2(v1730, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1732 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1731);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1733 = v21 + 92;
      const int8_t* v1734 = (const int8_t*) v1733;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1735 = *(const int8_t *)(v1734);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1736 = __riscv_vwmacc_vx_i16m1(v1676, v1735, v1732, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1737 = __riscv_vsrl_vx_u8mf2(v1716, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1738 = __riscv_vand_vx_u8mf2(v1737, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1739 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1738);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1740 = v21 + 124;
      const int8_t* v1741 = (const int8_t*) v1740;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1742 = *(const int8_t *)(v1741);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1743 = __riscv_vwmacc_vx_i16m1(v1683, v1742, v1739, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1744 = v19 + 712;
      const uint8_t* v1745 = (const uint8_t*) v1744;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1746 = __riscv_vle8_v_u8mf2(v1745, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1747 = __riscv_vand_vx_u8mf2(v1746, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1748 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1747);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1749 = v21 + 28;
      const int8_t* v1750 = (const int8_t*) v1749;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1751 = *(const int8_t *)(v1750);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1752 = __riscv_vwmacc_vx_i16m1(v1692, v1751, v1748, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1753 = __riscv_vsrl_vx_u8mf2(v1746, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1754 = __riscv_vand_vx_u8mf2(v1753, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1755 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1754);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1756 = v21 + 60;
      const int8_t* v1757 = (const int8_t*) v1756;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1758 = *(const int8_t *)(v1757);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1759 = __riscv_vwmacc_vx_i16m1(v1699, v1758, v1755, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1760 = __riscv_vsrl_vx_u8mf2(v1746, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1761 = __riscv_vand_vx_u8mf2(v1760, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1762 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1761);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1763 = v21 + 92;
      const int8_t* v1764 = (const int8_t*) v1763;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1765 = *(const int8_t *)(v1764);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1766 = __riscv_vwmacc_vx_i16m1(v1706, v1765, v1762, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1767 = __riscv_vsrl_vx_u8mf2(v1746, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1768 = __riscv_vand_vx_u8mf2(v1767, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1769 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1768);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1770 = v21 + 124;
      const int8_t* v1771 = (const int8_t*) v1770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1772 = *(const int8_t *)(v1771);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1773 = __riscv_vwmacc_vx_i16m1(v1713, v1772, v1769, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1774 = v19 + 720;
      const uint8_t* v1775 = (const uint8_t*) v1774;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1776 = __riscv_vle8_v_u8mf2(v1775, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1777 = __riscv_vand_vx_u8mf2(v1776, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1778 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1777);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1779 = v21 + 29;
      const int8_t* v1780 = (const int8_t*) v1779;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1781 = *(const int8_t *)(v1780);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1782 = __riscv_vwmacc_vx_i16m1(v1722, v1781, v1778, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1783 = __riscv_vsrl_vx_u8mf2(v1776, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1784 = __riscv_vand_vx_u8mf2(v1783, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1785 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1784);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1786 = v21 + 61;
      const int8_t* v1787 = (const int8_t*) v1786;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1788 = *(const int8_t *)(v1787);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1789 = __riscv_vwmacc_vx_i16m1(v1729, v1788, v1785, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1790 = __riscv_vsrl_vx_u8mf2(v1776, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1791 = __riscv_vand_vx_u8mf2(v1790, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1792 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1791);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1793 = v21 + 93;
      const int8_t* v1794 = (const int8_t*) v1793;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1795 = *(const int8_t *)(v1794);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1796 = __riscv_vwmacc_vx_i16m1(v1736, v1795, v1792, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1797 = __riscv_vsrl_vx_u8mf2(v1776, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1798 = __riscv_vand_vx_u8mf2(v1797, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1799 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1798);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1800 = v21 + 125;
      const int8_t* v1801 = (const int8_t*) v1800;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1802 = *(const int8_t *)(v1801);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1803 = __riscv_vwmacc_vx_i16m1(v1743, v1802, v1799, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1804 = v19 + 728;
      const uint8_t* v1805 = (const uint8_t*) v1804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1806 = __riscv_vle8_v_u8mf2(v1805, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1807 = __riscv_vand_vx_u8mf2(v1806, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1808 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1807);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1809 = v21 + 29;
      const int8_t* v1810 = (const int8_t*) v1809;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1811 = *(const int8_t *)(v1810);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1812 = __riscv_vwmacc_vx_i16m1(v1752, v1811, v1808, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1813 = __riscv_vsrl_vx_u8mf2(v1806, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1814 = __riscv_vand_vx_u8mf2(v1813, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1815 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1814);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1816 = v21 + 61;
      const int8_t* v1817 = (const int8_t*) v1816;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1818 = *(const int8_t *)(v1817);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1819 = __riscv_vwmacc_vx_i16m1(v1759, v1818, v1815, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1820 = __riscv_vsrl_vx_u8mf2(v1806, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1821 = __riscv_vand_vx_u8mf2(v1820, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1822 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1821);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1823 = v21 + 93;
      const int8_t* v1824 = (const int8_t*) v1823;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1825 = *(const int8_t *)(v1824);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1826 = __riscv_vwmacc_vx_i16m1(v1766, v1825, v1822, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1827 = __riscv_vsrl_vx_u8mf2(v1806, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1828 = __riscv_vand_vx_u8mf2(v1827, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1829 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1828);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1830 = v21 + 125;
      const int8_t* v1831 = (const int8_t*) v1830;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1832 = *(const int8_t *)(v1831);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1833 = __riscv_vwmacc_vx_i16m1(v1773, v1832, v1829, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1834 = v19 + 736;
      const uint8_t* v1835 = (const uint8_t*) v1834;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1836 = __riscv_vle8_v_u8mf2(v1835, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1837 = __riscv_vand_vx_u8mf2(v1836, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1838 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1837);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1839 = v21 + 30;
      const int8_t* v1840 = (const int8_t*) v1839;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1841 = *(const int8_t *)(v1840);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1842 = __riscv_vwmacc_vx_i16m1(v1782, v1841, v1838, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1843 = __riscv_vsrl_vx_u8mf2(v1836, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1844 = __riscv_vand_vx_u8mf2(v1843, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1845 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1844);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1846 = v21 + 62;
      const int8_t* v1847 = (const int8_t*) v1846;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1848 = *(const int8_t *)(v1847);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1849 = __riscv_vwmacc_vx_i16m1(v1789, v1848, v1845, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1850 = __riscv_vsrl_vx_u8mf2(v1836, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1851 = __riscv_vand_vx_u8mf2(v1850, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1852 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1851);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1853 = v21 + 94;
      const int8_t* v1854 = (const int8_t*) v1853;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1855 = *(const int8_t *)(v1854);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1856 = __riscv_vwmacc_vx_i16m1(v1796, v1855, v1852, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1857 = __riscv_vsrl_vx_u8mf2(v1836, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1858 = __riscv_vand_vx_u8mf2(v1857, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1859 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1858);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1860 = v21 + 126;
      const int8_t* v1861 = (const int8_t*) v1860;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1862 = *(const int8_t *)(v1861);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1863 = __riscv_vwmacc_vx_i16m1(v1803, v1862, v1859, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1864 = v19 + 744;
      const uint8_t* v1865 = (const uint8_t*) v1864;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1866 = __riscv_vle8_v_u8mf2(v1865, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1867 = __riscv_vand_vx_u8mf2(v1866, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1868 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1867);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1869 = v21 + 30;
      const int8_t* v1870 = (const int8_t*) v1869;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1871 = *(const int8_t *)(v1870);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1872 = __riscv_vwmacc_vx_i16m1(v1812, v1871, v1868, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1873 = __riscv_vsrl_vx_u8mf2(v1866, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1874 = __riscv_vand_vx_u8mf2(v1873, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1875 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1874);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1876 = v21 + 62;
      const int8_t* v1877 = (const int8_t*) v1876;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1878 = *(const int8_t *)(v1877);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1879 = __riscv_vwmacc_vx_i16m1(v1819, v1878, v1875, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1880 = __riscv_vsrl_vx_u8mf2(v1866, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1881 = __riscv_vand_vx_u8mf2(v1880, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1882 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1881);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1883 = v21 + 94;
      const int8_t* v1884 = (const int8_t*) v1883;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1885 = *(const int8_t *)(v1884);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1886 = __riscv_vwmacc_vx_i16m1(v1826, v1885, v1882, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1887 = __riscv_vsrl_vx_u8mf2(v1866, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1888 = __riscv_vand_vx_u8mf2(v1887, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1889 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1888);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1890 = v21 + 126;
      const int8_t* v1891 = (const int8_t*) v1890;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1892 = *(const int8_t *)(v1891);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1893 = __riscv_vwmacc_vx_i16m1(v1833, v1892, v1889, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1894 = v19 + 752;
      const uint8_t* v1895 = (const uint8_t*) v1894;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1896 = __riscv_vle8_v_u8mf2(v1895, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1897 = __riscv_vand_vx_u8mf2(v1896, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1898 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1897);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1899 = v21 + 31;
      const int8_t* v1900 = (const int8_t*) v1899;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1901 = *(const int8_t *)(v1900);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1902 = __riscv_vwmacc_vx_i16m1(v1842, v1901, v1898, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1903 = __riscv_vsrl_vx_u8mf2(v1896, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1904 = __riscv_vand_vx_u8mf2(v1903, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1905 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1904);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1906 = v21 + 63;
      const int8_t* v1907 = (const int8_t*) v1906;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1908 = *(const int8_t *)(v1907);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1909 = __riscv_vwmacc_vx_i16m1(v1849, v1908, v1905, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1910 = __riscv_vsrl_vx_u8mf2(v1896, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1911 = __riscv_vand_vx_u8mf2(v1910, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1912 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1911);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1913 = v21 + 95;
      const int8_t* v1914 = (const int8_t*) v1913;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1915 = *(const int8_t *)(v1914);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1916 = __riscv_vwmacc_vx_i16m1(v1856, v1915, v1912, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1917 = __riscv_vsrl_vx_u8mf2(v1896, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1918 = __riscv_vand_vx_u8mf2(v1917, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1919 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1918);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1920 = v21 + 127;
      const int8_t* v1921 = (const int8_t*) v1920;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1922 = *(const int8_t *)(v1921);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1923 = __riscv_vwmacc_vx_i16m1(v1863, v1922, v1919, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1924 = v19 + 760;
      const uint8_t* v1925 = (const uint8_t*) v1924;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1926 = __riscv_vle8_v_u8mf2(v1925, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1927 = __riscv_vand_vx_u8mf2(v1926, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1928 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1927);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1929 = v21 + 31;
      const int8_t* v1930 = (const int8_t*) v1929;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1931 = *(const int8_t *)(v1930);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1932 = __riscv_vwmacc_vx_i16m1(v1872, v1931, v1928, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1933 = __riscv_vsrl_vx_u8mf2(v1926, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1934 = __riscv_vand_vx_u8mf2(v1933, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1935 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1934);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1936 = v21 + 63;
      const int8_t* v1937 = (const int8_t*) v1936;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1938 = *(const int8_t *)(v1937);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1939 = __riscv_vwmacc_vx_i16m1(v1879, v1938, v1935, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1940 = __riscv_vsrl_vx_u8mf2(v1926, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1941 = __riscv_vand_vx_u8mf2(v1940, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1942 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1941);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1943 = v21 + 95;
      const int8_t* v1944 = (const int8_t*) v1943;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1945 = *(const int8_t *)(v1944);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1946 = __riscv_vwmacc_vx_i16m1(v1886, v1945, v1942, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1947 = __riscv_vsrl_vx_u8mf2(v1926, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1948 = __riscv_vand_vx_u8mf2(v1947, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1949 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1948);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1950 = v21 + 127;
      const int8_t* v1951 = (const int8_t*) v1950;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1952 = *(const int8_t *)(v1951);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1953 = __riscv_vwmacc_vx_i16m1(v1893, v1952, v1949, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1954 = v19 + 768;
      const uint8_t* v1955 = (const uint8_t*) v1954;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1956 = __riscv_vle8_v_u8mf2(v1955, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1957 = __riscv_vand_vx_u8mf2(v1956, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1958 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1957);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1959 = v21 + 32;
      const int8_t* v1960 = (const int8_t*) v1959;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1961 = *(const int8_t *)(v1960);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1962 = __riscv_vwmacc_vx_i16m1(v1902, v1961, v1958, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1963 = __riscv_vsrl_vx_u8mf2(v1956, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1964 = __riscv_vand_vx_u8mf2(v1963, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1965 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1964);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1966 = v21 + 64;
      const int8_t* v1967 = (const int8_t*) v1966;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1968 = *(const int8_t *)(v1967);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1969 = __riscv_vwmacc_vx_i16m1(v1909, v1968, v1965, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1970 = __riscv_vsrl_vx_u8mf2(v1956, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1971 = __riscv_vand_vx_u8mf2(v1970, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1972 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1971);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1973 = v21 + 96;
      const int8_t* v1974 = (const int8_t*) v1973;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1975 = *(const int8_t *)(v1974);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1976 = __riscv_vwmacc_vx_i16m1(v1916, v1975, v1972, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1977 = __riscv_vsrl_vx_u8mf2(v1956, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1978 = __riscv_vand_vx_u8mf2(v1977, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1979 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1978);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1980 = v21 + 128;
      const int8_t* v1981 = (const int8_t*) v1980;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1982 = *(const int8_t *)(v1981);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1983 = __riscv_vwmacc_vx_i16m1(v1923, v1982, v1979, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v1984 = v19 + 776;
      const uint8_t* v1985 = (const uint8_t*) v1984;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1986 = __riscv_vle8_v_u8mf2(v1985, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1987 = __riscv_vand_vx_u8mf2(v1986, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1988 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1987);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1989 = v21 + 32;
      const int8_t* v1990 = (const int8_t*) v1989;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1991 = *(const int8_t *)(v1990);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1992 = __riscv_vwmacc_vx_i16m1(v1932, v1991, v1988, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1993 = __riscv_vsrl_vx_u8mf2(v1986, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1994 = __riscv_vand_vx_u8mf2(v1993, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1995 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1994);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1996 = v21 + 64;
      const int8_t* v1997 = (const int8_t*) v1996;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1998 = *(const int8_t *)(v1997);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1999 = __riscv_vwmacc_vx_i16m1(v1939, v1998, v1995, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2000 = __riscv_vsrl_vx_u8mf2(v1986, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2001 = __riscv_vand_vx_u8mf2(v2000, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2002 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2001);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2003 = v21 + 96;
      const int8_t* v2004 = (const int8_t*) v2003;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2005 = *(const int8_t *)(v2004);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2006 = __riscv_vwmacc_vx_i16m1(v1946, v2005, v2002, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2007 = __riscv_vsrl_vx_u8mf2(v1986, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2008 = __riscv_vand_vx_u8mf2(v2007, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2009 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2008);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2010 = v21 + 128;
      const int8_t* v2011 = (const int8_t*) v2010;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2012 = *(const int8_t *)(v2011);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2013 = __riscv_vwmacc_vx_i16m1(v1953, v2012, v2009, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2014 = v19 + 784;
      const uint8_t* v2015 = (const uint8_t*) v2014;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2016 = __riscv_vle8_v_u8mf2(v2015, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2017 = __riscv_vand_vx_u8mf2(v2016, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2018 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2017);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2019 = v21 + 33;
      const int8_t* v2020 = (const int8_t*) v2019;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2021 = *(const int8_t *)(v2020);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2022 = __riscv_vwmacc_vx_i16m1(v1962, v2021, v2018, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2023 = __riscv_vsrl_vx_u8mf2(v2016, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2024 = __riscv_vand_vx_u8mf2(v2023, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2025 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2024);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2026 = v21 + 65;
      const int8_t* v2027 = (const int8_t*) v2026;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2028 = *(const int8_t *)(v2027);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2029 = __riscv_vwmacc_vx_i16m1(v1969, v2028, v2025, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2030 = __riscv_vsrl_vx_u8mf2(v2016, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2031 = __riscv_vand_vx_u8mf2(v2030, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2032 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2031);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2033 = v21 + 97;
      const int8_t* v2034 = (const int8_t*) v2033;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2035 = *(const int8_t *)(v2034);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2036 = __riscv_vwmacc_vx_i16m1(v1976, v2035, v2032, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2037 = __riscv_vsrl_vx_u8mf2(v2016, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2038 = __riscv_vand_vx_u8mf2(v2037, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2039 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2038);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2040 = v21 + 129;
      const int8_t* v2041 = (const int8_t*) v2040;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2042 = *(const int8_t *)(v2041);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2043 = __riscv_vwmacc_vx_i16m1(v1983, v2042, v2039, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2044 = v19 + 792;
      const uint8_t* v2045 = (const uint8_t*) v2044;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2046 = __riscv_vle8_v_u8mf2(v2045, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2047 = __riscv_vand_vx_u8mf2(v2046, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2048 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2047);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2049 = v21 + 33;
      const int8_t* v2050 = (const int8_t*) v2049;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2051 = *(const int8_t *)(v2050);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2052 = __riscv_vwmacc_vx_i16m1(v1992, v2051, v2048, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2053 = __riscv_vsrl_vx_u8mf2(v2046, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2054 = __riscv_vand_vx_u8mf2(v2053, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2055 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2054);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2056 = v21 + 65;
      const int8_t* v2057 = (const int8_t*) v2056;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2058 = *(const int8_t *)(v2057);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2059 = __riscv_vwmacc_vx_i16m1(v1999, v2058, v2055, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2060 = __riscv_vsrl_vx_u8mf2(v2046, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2061 = __riscv_vand_vx_u8mf2(v2060, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2062 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2061);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2063 = v21 + 97;
      const int8_t* v2064 = (const int8_t*) v2063;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2065 = *(const int8_t *)(v2064);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2066 = __riscv_vwmacc_vx_i16m1(v2006, v2065, v2062, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2067 = __riscv_vsrl_vx_u8mf2(v2046, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2068 = __riscv_vand_vx_u8mf2(v2067, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2069 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2068);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2070 = v21 + 129;
      const int8_t* v2071 = (const int8_t*) v2070;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2072 = *(const int8_t *)(v2071);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2073 = __riscv_vwmacc_vx_i16m1(v2013, v2072, v2069, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2074 = v19 + 800;
      const uint8_t* v2075 = (const uint8_t*) v2074;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2076 = __riscv_vle8_v_u8mf2(v2075, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2077 = __riscv_vand_vx_u8mf2(v2076, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2078 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2077);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2079 = v21 + 34;
      const int8_t* v2080 = (const int8_t*) v2079;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2081 = *(const int8_t *)(v2080);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2082 = __riscv_vwmacc_vx_i16m1(v2022, v2081, v2078, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2083 = __riscv_vsrl_vx_u8mf2(v2076, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2084 = __riscv_vand_vx_u8mf2(v2083, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2085 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2084);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2086 = v21 + 66;
      const int8_t* v2087 = (const int8_t*) v2086;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2088 = *(const int8_t *)(v2087);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2089 = __riscv_vwmacc_vx_i16m1(v2029, v2088, v2085, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2090 = __riscv_vsrl_vx_u8mf2(v2076, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2091 = __riscv_vand_vx_u8mf2(v2090, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2092 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2091);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2093 = v21 + 98;
      const int8_t* v2094 = (const int8_t*) v2093;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2095 = *(const int8_t *)(v2094);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2096 = __riscv_vwmacc_vx_i16m1(v2036, v2095, v2092, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2097 = __riscv_vsrl_vx_u8mf2(v2076, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2098 = __riscv_vand_vx_u8mf2(v2097, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2099 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2098);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2100 = v21 + 130;
      const int8_t* v2101 = (const int8_t*) v2100;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2102 = *(const int8_t *)(v2101);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2103 = __riscv_vwmacc_vx_i16m1(v2043, v2102, v2099, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2104 = v19 + 808;
      const uint8_t* v2105 = (const uint8_t*) v2104;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2106 = __riscv_vle8_v_u8mf2(v2105, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2107 = __riscv_vand_vx_u8mf2(v2106, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2108 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2107);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2109 = v21 + 34;
      const int8_t* v2110 = (const int8_t*) v2109;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2111 = *(const int8_t *)(v2110);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2112 = __riscv_vwmacc_vx_i16m1(v2052, v2111, v2108, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2113 = __riscv_vsrl_vx_u8mf2(v2106, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2114 = __riscv_vand_vx_u8mf2(v2113, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2115 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2114);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2116 = v21 + 66;
      const int8_t* v2117 = (const int8_t*) v2116;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2118 = *(const int8_t *)(v2117);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2119 = __riscv_vwmacc_vx_i16m1(v2059, v2118, v2115, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2120 = __riscv_vsrl_vx_u8mf2(v2106, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2121 = __riscv_vand_vx_u8mf2(v2120, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2122 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2121);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2123 = v21 + 98;
      const int8_t* v2124 = (const int8_t*) v2123;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2125 = *(const int8_t *)(v2124);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2126 = __riscv_vwmacc_vx_i16m1(v2066, v2125, v2122, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2127 = __riscv_vsrl_vx_u8mf2(v2106, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2128 = __riscv_vand_vx_u8mf2(v2127, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2129 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2128);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2130 = v21 + 130;
      const int8_t* v2131 = (const int8_t*) v2130;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2132 = *(const int8_t *)(v2131);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2133 = __riscv_vwmacc_vx_i16m1(v2073, v2132, v2129, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2134 = v19 + 816;
      const uint8_t* v2135 = (const uint8_t*) v2134;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2136 = __riscv_vle8_v_u8mf2(v2135, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2137 = __riscv_vand_vx_u8mf2(v2136, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2138 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2137);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2139 = v21 + 35;
      const int8_t* v2140 = (const int8_t*) v2139;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2141 = *(const int8_t *)(v2140);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2142 = __riscv_vwmacc_vx_i16m1(v2082, v2141, v2138, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2143 = __riscv_vsrl_vx_u8mf2(v2136, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2144 = __riscv_vand_vx_u8mf2(v2143, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2145 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2144);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2146 = v21 + 67;
      const int8_t* v2147 = (const int8_t*) v2146;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2148 = *(const int8_t *)(v2147);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2149 = __riscv_vwmacc_vx_i16m1(v2089, v2148, v2145, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2150 = __riscv_vsrl_vx_u8mf2(v2136, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2151 = __riscv_vand_vx_u8mf2(v2150, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2152 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2151);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2153 = v21 + 99;
      const int8_t* v2154 = (const int8_t*) v2153;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2155 = *(const int8_t *)(v2154);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2156 = __riscv_vwmacc_vx_i16m1(v2096, v2155, v2152, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2157 = __riscv_vsrl_vx_u8mf2(v2136, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2158 = __riscv_vand_vx_u8mf2(v2157, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2159 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2158);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2160 = v21 + 131;
      const int8_t* v2161 = (const int8_t*) v2160;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2162 = *(const int8_t *)(v2161);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2163 = __riscv_vwmacc_vx_i16m1(v2103, v2162, v2159, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2164 = v19 + 824;
      const uint8_t* v2165 = (const uint8_t*) v2164;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2166 = __riscv_vle8_v_u8mf2(v2165, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2167 = __riscv_vand_vx_u8mf2(v2166, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2168 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2167);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2169 = v21 + 35;
      const int8_t* v2170 = (const int8_t*) v2169;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2171 = *(const int8_t *)(v2170);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2172 = __riscv_vwmacc_vx_i16m1(v2112, v2171, v2168, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2173 = __riscv_vsrl_vx_u8mf2(v2166, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2174 = __riscv_vand_vx_u8mf2(v2173, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2175 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2174);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2176 = v21 + 67;
      const int8_t* v2177 = (const int8_t*) v2176;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2178 = *(const int8_t *)(v2177);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2179 = __riscv_vwmacc_vx_i16m1(v2119, v2178, v2175, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2180 = __riscv_vsrl_vx_u8mf2(v2166, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2181 = __riscv_vand_vx_u8mf2(v2180, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2182 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2181);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2183 = v21 + 99;
      const int8_t* v2184 = (const int8_t*) v2183;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2185 = *(const int8_t *)(v2184);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2186 = __riscv_vwmacc_vx_i16m1(v2126, v2185, v2182, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2187 = __riscv_vsrl_vx_u8mf2(v2166, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2188 = __riscv_vand_vx_u8mf2(v2187, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2189 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2188);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2190 = v21 + 131;
      const int8_t* v2191 = (const int8_t*) v2190;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2192 = *(const int8_t *)(v2191);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2193 = __riscv_vwmacc_vx_i16m1(v2133, v2192, v2189, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v2194 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2195 = __riscv_vwmacc_vv_i32m2(v2194, v57, v2142, 8);
      v34 = v2195;
      vint32m2_t v2196 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2197 = __riscv_vwmacc_vv_i32m2(v2196, v75, v2149, 8);
      v34 = v2197;
      vint32m2_t v2198 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2199 = __riscv_vwmacc_vv_i32m2(v2198, v93, v2156, 8);
      v34 = v2199;
      vint32m2_t v2200 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2201 = __riscv_vwmacc_vv_i32m2(v2200, v111, v2163, 8);
      v34 = v2201;
      vint32m2_t v2202 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2203 = __riscv_vwmacc_vv_i32m2(v2202, v129, v2172, 8);
      v38 = v2203;
      vint32m2_t v2204 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2205 = __riscv_vwmacc_vv_i32m2(v2204, v147, v2179, 8);
      v38 = v2205;
      vint32m2_t v2206 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2207 = __riscv_vwmacc_vv_i32m2(v2206, v165, v2186, 8);
      v38 = v2207;
      vint32m2_t v2208 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2209 = __riscv_vwmacc_vv_i32m2(v2208, v183, v2193, 8);
      v38 = v2209;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
      const uint8_t* v2210 = v19 + 192;
      const uint8_t* v2211 = (const uint8_t*) v2210;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2212 = __riscv_vle8_v_u8mf2(v2211, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2213 = __riscv_vand_vx_u8mf2(v2212, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2214 = __riscv_vsrl_vx_u8mf2(v2212, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2215 = __riscv_vzext_vf2_u16m1(v2213, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2216 = __riscv_vreinterpret_v_u16m1_i16m1(v2215);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2217 = __riscv_vzext_vf2_u16m1(v2214, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2218 = __riscv_vreinterpret_v_u16m1_i16m1(v2217);
      const uint8_t* v2219 = v19 + 208;
      const uint8_t* v2220 = (const uint8_t*) v2219;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2221 = __riscv_vle8_v_u8mf2(v2220, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2222 = __riscv_vand_vx_u8mf2(v2221, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2223 = __riscv_vsrl_vx_u8mf2(v2221, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2224 = __riscv_vzext_vf2_u16m1(v2222, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2225 = __riscv_vreinterpret_v_u16m1_i16m1(v2224);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2226 = __riscv_vzext_vf2_u16m1(v2223, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2227 = __riscv_vreinterpret_v_u16m1_i16m1(v2226);
      const uint8_t* v2228 = v19 + 224;
      const uint8_t* v2229 = (const uint8_t*) v2228;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2230 = __riscv_vle8_v_u8mf2(v2229, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2231 = __riscv_vand_vx_u8mf2(v2230, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2232 = __riscv_vsrl_vx_u8mf2(v2230, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2233 = __riscv_vzext_vf2_u16m1(v2231, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2234 = __riscv_vreinterpret_v_u16m1_i16m1(v2233);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2235 = __riscv_vzext_vf2_u16m1(v2232, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2236 = __riscv_vreinterpret_v_u16m1_i16m1(v2235);
      const uint8_t* v2237 = v19 + 240;
      const uint8_t* v2238 = (const uint8_t*) v2237;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2239 = __riscv_vle8_v_u8mf2(v2238, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2240 = __riscv_vand_vx_u8mf2(v2239, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2241 = __riscv_vsrl_vx_u8mf2(v2239, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2242 = __riscv_vzext_vf2_u16m1(v2240, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2243 = __riscv_vreinterpret_v_u16m1_i16m1(v2242);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2244 = __riscv_vzext_vf2_u16m1(v2241, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2245 = __riscv_vreinterpret_v_u16m1_i16m1(v2244);
      const uint8_t* v2246 = v19 + 256;
      const uint8_t* v2247 = (const uint8_t*) v2246;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2248 = __riscv_vle8_v_u8mf2(v2247, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2249 = __riscv_vand_vx_u8mf2(v2248, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2250 = __riscv_vsrl_vx_u8mf2(v2248, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2251 = __riscv_vzext_vf2_u16m1(v2249, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2252 = __riscv_vreinterpret_v_u16m1_i16m1(v2251);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2253 = __riscv_vzext_vf2_u16m1(v2250, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2254 = __riscv_vreinterpret_v_u16m1_i16m1(v2253);
      const uint8_t* v2255 = v19 + 272;
      const uint8_t* v2256 = (const uint8_t*) v2255;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2257 = __riscv_vle8_v_u8mf2(v2256, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2258 = __riscv_vand_vx_u8mf2(v2257, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2259 = __riscv_vsrl_vx_u8mf2(v2257, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2260 = __riscv_vzext_vf2_u16m1(v2258, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2261 = __riscv_vreinterpret_v_u16m1_i16m1(v2260);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2262 = __riscv_vzext_vf2_u16m1(v2259, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2263 = __riscv_vreinterpret_v_u16m1_i16m1(v2262);
      const uint8_t* v2264 = v19 + 288;
      const uint8_t* v2265 = (const uint8_t*) v2264;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2266 = __riscv_vle8_v_u8mf2(v2265, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2267 = __riscv_vand_vx_u8mf2(v2266, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2268 = __riscv_vsrl_vx_u8mf2(v2266, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2269 = __riscv_vzext_vf2_u16m1(v2267, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2270 = __riscv_vreinterpret_v_u16m1_i16m1(v2269);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2271 = __riscv_vzext_vf2_u16m1(v2268, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2272 = __riscv_vreinterpret_v_u16m1_i16m1(v2271);
      const uint8_t* v2273 = v19 + 304;
      const uint8_t* v2274 = (const uint8_t*) v2273;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2275 = __riscv_vle8_v_u8mf2(v2274, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2276 = __riscv_vand_vx_u8mf2(v2275, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2277 = __riscv_vsrl_vx_u8mf2(v2275, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2278 = __riscv_vzext_vf2_u16m1(v2276, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2279 = __riscv_vreinterpret_v_u16m1_i16m1(v2278);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2280 = __riscv_vzext_vf2_u16m1(v2277, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2281 = __riscv_vreinterpret_v_u16m1_i16m1(v2280);
      const uint8_t* v2282 = v19 + 200;
      const uint8_t* v2283 = (const uint8_t*) v2282;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2284 = __riscv_vle8_v_u8mf2(v2283, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2285 = __riscv_vand_vx_u8mf2(v2284, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2286 = __riscv_vsrl_vx_u8mf2(v2284, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2287 = __riscv_vzext_vf2_u16m1(v2285, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2288 = __riscv_vreinterpret_v_u16m1_i16m1(v2287);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2289 = __riscv_vzext_vf2_u16m1(v2286, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2290 = __riscv_vreinterpret_v_u16m1_i16m1(v2289);
      const uint8_t* v2291 = v19 + 216;
      const uint8_t* v2292 = (const uint8_t*) v2291;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2293 = __riscv_vle8_v_u8mf2(v2292, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2294 = __riscv_vand_vx_u8mf2(v2293, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2295 = __riscv_vsrl_vx_u8mf2(v2293, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2296 = __riscv_vzext_vf2_u16m1(v2294, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2297 = __riscv_vreinterpret_v_u16m1_i16m1(v2296);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2298 = __riscv_vzext_vf2_u16m1(v2295, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2299 = __riscv_vreinterpret_v_u16m1_i16m1(v2298);
      const uint8_t* v2300 = v19 + 232;
      const uint8_t* v2301 = (const uint8_t*) v2300;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2302 = __riscv_vle8_v_u8mf2(v2301, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2303 = __riscv_vand_vx_u8mf2(v2302, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2304 = __riscv_vsrl_vx_u8mf2(v2302, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2305 = __riscv_vzext_vf2_u16m1(v2303, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2306 = __riscv_vreinterpret_v_u16m1_i16m1(v2305);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2307 = __riscv_vzext_vf2_u16m1(v2304, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2308 = __riscv_vreinterpret_v_u16m1_i16m1(v2307);
      const uint8_t* v2309 = v19 + 248;
      const uint8_t* v2310 = (const uint8_t*) v2309;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2311 = __riscv_vle8_v_u8mf2(v2310, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2312 = __riscv_vand_vx_u8mf2(v2311, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2313 = __riscv_vsrl_vx_u8mf2(v2311, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2314 = __riscv_vzext_vf2_u16m1(v2312, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2315 = __riscv_vreinterpret_v_u16m1_i16m1(v2314);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2316 = __riscv_vzext_vf2_u16m1(v2313, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2317 = __riscv_vreinterpret_v_u16m1_i16m1(v2316);
      const uint8_t* v2318 = v19 + 264;
      const uint8_t* v2319 = (const uint8_t*) v2318;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2320 = __riscv_vle8_v_u8mf2(v2319, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2321 = __riscv_vand_vx_u8mf2(v2320, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2322 = __riscv_vsrl_vx_u8mf2(v2320, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2323 = __riscv_vzext_vf2_u16m1(v2321, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2324 = __riscv_vreinterpret_v_u16m1_i16m1(v2323);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2325 = __riscv_vzext_vf2_u16m1(v2322, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2326 = __riscv_vreinterpret_v_u16m1_i16m1(v2325);
      const uint8_t* v2327 = v19 + 280;
      const uint8_t* v2328 = (const uint8_t*) v2327;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2329 = __riscv_vle8_v_u8mf2(v2328, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2330 = __riscv_vand_vx_u8mf2(v2329, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2331 = __riscv_vsrl_vx_u8mf2(v2329, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2332 = __riscv_vzext_vf2_u16m1(v2330, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2333 = __riscv_vreinterpret_v_u16m1_i16m1(v2332);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2334 = __riscv_vzext_vf2_u16m1(v2331, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2335 = __riscv_vreinterpret_v_u16m1_i16m1(v2334);
      const uint8_t* v2336 = v19 + 296;
      const uint8_t* v2337 = (const uint8_t*) v2336;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2338 = __riscv_vle8_v_u8mf2(v2337, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2339 = __riscv_vand_vx_u8mf2(v2338, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2340 = __riscv_vsrl_vx_u8mf2(v2338, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2341 = __riscv_vzext_vf2_u16m1(v2339, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2342 = __riscv_vreinterpret_v_u16m1_i16m1(v2341);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2343 = __riscv_vzext_vf2_u16m1(v2340, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2344 = __riscv_vreinterpret_v_u16m1_i16m1(v2343);
      const uint8_t* v2345 = v19 + 312;
      const uint8_t* v2346 = (const uint8_t*) v2345;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2347 = __riscv_vle8_v_u8mf2(v2346, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2348 = __riscv_vand_vx_u8mf2(v2347, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2349 = __riscv_vsrl_vx_u8mf2(v2347, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2350 = __riscv_vzext_vf2_u16m1(v2348, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2351 = __riscv_vreinterpret_v_u16m1_i16m1(v2350);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2352 = __riscv_vzext_vf2_u16m1(v2349, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2353 = __riscv_vreinterpret_v_u16m1_i16m1(v2352);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold
      const uint8_t* v2354 = v21 + 276;
      const int16_t* v2355 = (const int16_t*) v2354;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2356 = *(const int16_t *)(v2355);
      vint32m2_t v2357 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2358 = __riscv_vwmacc_vx_i32m2(v2357, v2356, v2218, 8);
      v36 = v2358;
      vint32m2_t v2359 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2360 = __riscv_vwmacc_vx_i32m2(v2359, v2356, v2290, 8);
      v40 = v2360;
      const uint8_t* v2361 = v21 + 278;
      const int16_t* v2362 = (const int16_t*) v2361;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2363 = *(const int16_t *)(v2362);
      vint32m2_t v2364 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2365 = __riscv_vwmacc_vx_i32m2(v2364, v2363, v2227, 8);
      v36 = v2365;
      vint32m2_t v2366 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2367 = __riscv_vwmacc_vx_i32m2(v2366, v2363, v2299, 8);
      v40 = v2367;
      const uint8_t* v2368 = v21 + 280;
      const int16_t* v2369 = (const int16_t*) v2368;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2370 = *(const int16_t *)(v2369);
      vint32m2_t v2371 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2372 = __riscv_vwmacc_vx_i32m2(v2371, v2370, v2236, 8);
      v36 = v2372;
      vint32m2_t v2373 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2374 = __riscv_vwmacc_vx_i32m2(v2373, v2370, v2308, 8);
      v40 = v2374;
      const uint8_t* v2375 = v21 + 282;
      const int16_t* v2376 = (const int16_t*) v2375;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2377 = *(const int16_t *)(v2376);
      vint32m2_t v2378 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2379 = __riscv_vwmacc_vx_i32m2(v2378, v2377, v2245, 8);
      v36 = v2379;
      vint32m2_t v2380 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2381 = __riscv_vwmacc_vx_i32m2(v2380, v2377, v2317, 8);
      v40 = v2381;
      const uint8_t* v2382 = v21 + 284;
      const int16_t* v2383 = (const int16_t*) v2382;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2384 = *(const int16_t *)(v2383);
      vint32m2_t v2385 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2386 = __riscv_vwmacc_vx_i32m2(v2385, v2384, v2254, 8);
      v36 = v2386;
      vint32m2_t v2387 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2388 = __riscv_vwmacc_vx_i32m2(v2387, v2384, v2326, 8);
      v40 = v2388;
      const uint8_t* v2389 = v21 + 286;
      const int16_t* v2390 = (const int16_t*) v2389;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2391 = *(const int16_t *)(v2390);
      vint32m2_t v2392 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2393 = __riscv_vwmacc_vx_i32m2(v2392, v2391, v2263, 8);
      v36 = v2393;
      vint32m2_t v2394 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2395 = __riscv_vwmacc_vx_i32m2(v2394, v2391, v2335, 8);
      v40 = v2395;
      const uint8_t* v2396 = v21 + 288;
      const int16_t* v2397 = (const int16_t*) v2396;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2398 = *(const int16_t *)(v2397);
      vint32m2_t v2399 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2400 = __riscv_vwmacc_vx_i32m2(v2399, v2398, v2272, 8);
      v36 = v2400;
      vint32m2_t v2401 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2402 = __riscv_vwmacc_vx_i32m2(v2401, v2398, v2344, 8);
      v40 = v2402;
      const uint8_t* v2403 = v21 + 290;
      const int16_t* v2404 = (const int16_t*) v2403;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2405 = *(const int16_t *)(v2404);
      vint32m2_t v2406 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2407 = __riscv_vwmacc_vx_i32m2(v2406, v2405, v2281, 8);
      v36 = v2407;
      vint32m2_t v2408 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2409 = __riscv_vwmacc_vx_i32m2(v2408, v2405, v2353, 8);
      v40 = v2409;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2410 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2411 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2412 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2413 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2414 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2415 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2416 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2417 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2418 = v19 + 832;
      const uint8_t* v2419 = (const uint8_t*) v2418;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2420 = __riscv_vle8_v_u8mf2(v2419, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2421 = __riscv_vand_vx_u8mf2(v2420, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2422 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2421);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2423 = v21 + 132;
      const int8_t* v2424 = (const int8_t*) v2423;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2425 = *(const int8_t *)(v2424);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2426 = __riscv_vwmacc_vx_i16m1(v2410, v2425, v2422, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2427 = __riscv_vsrl_vx_u8mf2(v2420, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2428 = __riscv_vand_vx_u8mf2(v2427, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2429 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2428);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2430 = v21 + 164;
      const int8_t* v2431 = (const int8_t*) v2430;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2432 = *(const int8_t *)(v2431);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2433 = __riscv_vwmacc_vx_i16m1(v2411, v2432, v2429, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2434 = __riscv_vsrl_vx_u8mf2(v2420, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2435 = __riscv_vand_vx_u8mf2(v2434, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2436 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2435);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2437 = v21 + 196;
      const int8_t* v2438 = (const int8_t*) v2437;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2439 = *(const int8_t *)(v2438);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2440 = __riscv_vwmacc_vx_i16m1(v2412, v2439, v2436, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2441 = __riscv_vsrl_vx_u8mf2(v2420, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2442 = __riscv_vand_vx_u8mf2(v2441, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2443 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2442);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2444 = v21 + 228;
      const int8_t* v2445 = (const int8_t*) v2444;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2446 = *(const int8_t *)(v2445);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2447 = __riscv_vwmacc_vx_i16m1(v2413, v2446, v2443, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2448 = v19 + 840;
      const uint8_t* v2449 = (const uint8_t*) v2448;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2450 = __riscv_vle8_v_u8mf2(v2449, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2451 = __riscv_vand_vx_u8mf2(v2450, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2452 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2451);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2453 = v21 + 132;
      const int8_t* v2454 = (const int8_t*) v2453;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2455 = *(const int8_t *)(v2454);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2456 = __riscv_vwmacc_vx_i16m1(v2414, v2455, v2452, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2457 = __riscv_vsrl_vx_u8mf2(v2450, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2458 = __riscv_vand_vx_u8mf2(v2457, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2459 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2458);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2460 = v21 + 164;
      const int8_t* v2461 = (const int8_t*) v2460;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2462 = *(const int8_t *)(v2461);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2463 = __riscv_vwmacc_vx_i16m1(v2415, v2462, v2459, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2464 = __riscv_vsrl_vx_u8mf2(v2450, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2465 = __riscv_vand_vx_u8mf2(v2464, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2466 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2465);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2467 = v21 + 196;
      const int8_t* v2468 = (const int8_t*) v2467;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2469 = *(const int8_t *)(v2468);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2470 = __riscv_vwmacc_vx_i16m1(v2416, v2469, v2466, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2471 = __riscv_vsrl_vx_u8mf2(v2450, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2472 = __riscv_vand_vx_u8mf2(v2471, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2473 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2472);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2474 = v21 + 228;
      const int8_t* v2475 = (const int8_t*) v2474;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2476 = *(const int8_t *)(v2475);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2477 = __riscv_vwmacc_vx_i16m1(v2417, v2476, v2473, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2478 = v19 + 848;
      const uint8_t* v2479 = (const uint8_t*) v2478;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2480 = __riscv_vle8_v_u8mf2(v2479, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2481 = __riscv_vand_vx_u8mf2(v2480, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2482 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2481);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2483 = v21 + 133;
      const int8_t* v2484 = (const int8_t*) v2483;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2485 = *(const int8_t *)(v2484);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2486 = __riscv_vwmacc_vx_i16m1(v2426, v2485, v2482, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2487 = __riscv_vsrl_vx_u8mf2(v2480, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2488 = __riscv_vand_vx_u8mf2(v2487, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2489 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2488);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2490 = v21 + 165;
      const int8_t* v2491 = (const int8_t*) v2490;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2492 = *(const int8_t *)(v2491);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2493 = __riscv_vwmacc_vx_i16m1(v2433, v2492, v2489, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2494 = __riscv_vsrl_vx_u8mf2(v2480, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2495 = __riscv_vand_vx_u8mf2(v2494, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2496 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2495);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2497 = v21 + 197;
      const int8_t* v2498 = (const int8_t*) v2497;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2499 = *(const int8_t *)(v2498);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2500 = __riscv_vwmacc_vx_i16m1(v2440, v2499, v2496, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2501 = __riscv_vsrl_vx_u8mf2(v2480, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2502 = __riscv_vand_vx_u8mf2(v2501, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2503 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2502);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2504 = v21 + 229;
      const int8_t* v2505 = (const int8_t*) v2504;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2506 = *(const int8_t *)(v2505);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2507 = __riscv_vwmacc_vx_i16m1(v2447, v2506, v2503, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2508 = v19 + 856;
      const uint8_t* v2509 = (const uint8_t*) v2508;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2510 = __riscv_vle8_v_u8mf2(v2509, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2511 = __riscv_vand_vx_u8mf2(v2510, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2512 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2511);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2513 = v21 + 133;
      const int8_t* v2514 = (const int8_t*) v2513;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2515 = *(const int8_t *)(v2514);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2516 = __riscv_vwmacc_vx_i16m1(v2456, v2515, v2512, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2517 = __riscv_vsrl_vx_u8mf2(v2510, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2518 = __riscv_vand_vx_u8mf2(v2517, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2519 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2518);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2520 = v21 + 165;
      const int8_t* v2521 = (const int8_t*) v2520;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2522 = *(const int8_t *)(v2521);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2523 = __riscv_vwmacc_vx_i16m1(v2463, v2522, v2519, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2524 = __riscv_vsrl_vx_u8mf2(v2510, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2525 = __riscv_vand_vx_u8mf2(v2524, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2526 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2525);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2527 = v21 + 197;
      const int8_t* v2528 = (const int8_t*) v2527;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2529 = *(const int8_t *)(v2528);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2530 = __riscv_vwmacc_vx_i16m1(v2470, v2529, v2526, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2531 = __riscv_vsrl_vx_u8mf2(v2510, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2532 = __riscv_vand_vx_u8mf2(v2531, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2533 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2532);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2534 = v21 + 229;
      const int8_t* v2535 = (const int8_t*) v2534;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2536 = *(const int8_t *)(v2535);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2537 = __riscv_vwmacc_vx_i16m1(v2477, v2536, v2533, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2538 = v19 + 864;
      const uint8_t* v2539 = (const uint8_t*) v2538;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2540 = __riscv_vle8_v_u8mf2(v2539, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2541 = __riscv_vand_vx_u8mf2(v2540, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2542 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2541);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2543 = v21 + 134;
      const int8_t* v2544 = (const int8_t*) v2543;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2545 = *(const int8_t *)(v2544);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2546 = __riscv_vwmacc_vx_i16m1(v2486, v2545, v2542, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2547 = __riscv_vsrl_vx_u8mf2(v2540, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2548 = __riscv_vand_vx_u8mf2(v2547, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2549 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2548);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2550 = v21 + 166;
      const int8_t* v2551 = (const int8_t*) v2550;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2552 = *(const int8_t *)(v2551);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2553 = __riscv_vwmacc_vx_i16m1(v2493, v2552, v2549, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2554 = __riscv_vsrl_vx_u8mf2(v2540, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2555 = __riscv_vand_vx_u8mf2(v2554, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2556 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2555);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2557 = v21 + 198;
      const int8_t* v2558 = (const int8_t*) v2557;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2559 = *(const int8_t *)(v2558);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2560 = __riscv_vwmacc_vx_i16m1(v2500, v2559, v2556, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2561 = __riscv_vsrl_vx_u8mf2(v2540, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2562 = __riscv_vand_vx_u8mf2(v2561, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2563 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2562);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2564 = v21 + 230;
      const int8_t* v2565 = (const int8_t*) v2564;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2566 = *(const int8_t *)(v2565);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2567 = __riscv_vwmacc_vx_i16m1(v2507, v2566, v2563, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2568 = v19 + 872;
      const uint8_t* v2569 = (const uint8_t*) v2568;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2570 = __riscv_vle8_v_u8mf2(v2569, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2571 = __riscv_vand_vx_u8mf2(v2570, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2572 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2571);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2573 = v21 + 134;
      const int8_t* v2574 = (const int8_t*) v2573;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2575 = *(const int8_t *)(v2574);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2576 = __riscv_vwmacc_vx_i16m1(v2516, v2575, v2572, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2577 = __riscv_vsrl_vx_u8mf2(v2570, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2578 = __riscv_vand_vx_u8mf2(v2577, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2579 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2578);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2580 = v21 + 166;
      const int8_t* v2581 = (const int8_t*) v2580;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2582 = *(const int8_t *)(v2581);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2583 = __riscv_vwmacc_vx_i16m1(v2523, v2582, v2579, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2584 = __riscv_vsrl_vx_u8mf2(v2570, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2585 = __riscv_vand_vx_u8mf2(v2584, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2586 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2585);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2587 = v21 + 198;
      const int8_t* v2588 = (const int8_t*) v2587;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2589 = *(const int8_t *)(v2588);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2590 = __riscv_vwmacc_vx_i16m1(v2530, v2589, v2586, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2591 = __riscv_vsrl_vx_u8mf2(v2570, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2592 = __riscv_vand_vx_u8mf2(v2591, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2593 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2592);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2594 = v21 + 230;
      const int8_t* v2595 = (const int8_t*) v2594;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2596 = *(const int8_t *)(v2595);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2597 = __riscv_vwmacc_vx_i16m1(v2537, v2596, v2593, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2598 = v19 + 880;
      const uint8_t* v2599 = (const uint8_t*) v2598;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2600 = __riscv_vle8_v_u8mf2(v2599, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2601 = __riscv_vand_vx_u8mf2(v2600, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2602 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2601);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2603 = v21 + 135;
      const int8_t* v2604 = (const int8_t*) v2603;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2605 = *(const int8_t *)(v2604);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2606 = __riscv_vwmacc_vx_i16m1(v2546, v2605, v2602, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2607 = __riscv_vsrl_vx_u8mf2(v2600, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2608 = __riscv_vand_vx_u8mf2(v2607, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2609 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2608);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2610 = v21 + 167;
      const int8_t* v2611 = (const int8_t*) v2610;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2612 = *(const int8_t *)(v2611);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2613 = __riscv_vwmacc_vx_i16m1(v2553, v2612, v2609, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2614 = __riscv_vsrl_vx_u8mf2(v2600, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2615 = __riscv_vand_vx_u8mf2(v2614, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2616 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2615);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2617 = v21 + 199;
      const int8_t* v2618 = (const int8_t*) v2617;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2619 = *(const int8_t *)(v2618);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2620 = __riscv_vwmacc_vx_i16m1(v2560, v2619, v2616, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2621 = __riscv_vsrl_vx_u8mf2(v2600, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2622 = __riscv_vand_vx_u8mf2(v2621, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2623 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2622);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2624 = v21 + 231;
      const int8_t* v2625 = (const int8_t*) v2624;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2626 = *(const int8_t *)(v2625);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2627 = __riscv_vwmacc_vx_i16m1(v2567, v2626, v2623, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2628 = v19 + 888;
      const uint8_t* v2629 = (const uint8_t*) v2628;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2630 = __riscv_vle8_v_u8mf2(v2629, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2631 = __riscv_vand_vx_u8mf2(v2630, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2632 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2631);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2633 = v21 + 135;
      const int8_t* v2634 = (const int8_t*) v2633;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2635 = *(const int8_t *)(v2634);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2636 = __riscv_vwmacc_vx_i16m1(v2576, v2635, v2632, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2637 = __riscv_vsrl_vx_u8mf2(v2630, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2638 = __riscv_vand_vx_u8mf2(v2637, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2639 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2638);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2640 = v21 + 167;
      const int8_t* v2641 = (const int8_t*) v2640;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2642 = *(const int8_t *)(v2641);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2643 = __riscv_vwmacc_vx_i16m1(v2583, v2642, v2639, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2644 = __riscv_vsrl_vx_u8mf2(v2630, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2645 = __riscv_vand_vx_u8mf2(v2644, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2646 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2645);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2647 = v21 + 199;
      const int8_t* v2648 = (const int8_t*) v2647;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2649 = *(const int8_t *)(v2648);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2650 = __riscv_vwmacc_vx_i16m1(v2590, v2649, v2646, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2651 = __riscv_vsrl_vx_u8mf2(v2630, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2652 = __riscv_vand_vx_u8mf2(v2651, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2653 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2652);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2654 = v21 + 231;
      const int8_t* v2655 = (const int8_t*) v2654;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2656 = *(const int8_t *)(v2655);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2657 = __riscv_vwmacc_vx_i16m1(v2597, v2656, v2653, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2658 = v19 + 896;
      const uint8_t* v2659 = (const uint8_t*) v2658;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2660 = __riscv_vle8_v_u8mf2(v2659, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2661 = __riscv_vand_vx_u8mf2(v2660, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2662 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2661);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2663 = v21 + 136;
      const int8_t* v2664 = (const int8_t*) v2663;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2665 = *(const int8_t *)(v2664);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2666 = __riscv_vwmacc_vx_i16m1(v2606, v2665, v2662, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2667 = __riscv_vsrl_vx_u8mf2(v2660, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2668 = __riscv_vand_vx_u8mf2(v2667, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2669 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2668);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2670 = v21 + 168;
      const int8_t* v2671 = (const int8_t*) v2670;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2672 = *(const int8_t *)(v2671);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2673 = __riscv_vwmacc_vx_i16m1(v2613, v2672, v2669, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2674 = __riscv_vsrl_vx_u8mf2(v2660, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2675 = __riscv_vand_vx_u8mf2(v2674, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2676 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2675);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2677 = v21 + 200;
      const int8_t* v2678 = (const int8_t*) v2677;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2679 = *(const int8_t *)(v2678);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2680 = __riscv_vwmacc_vx_i16m1(v2620, v2679, v2676, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2681 = __riscv_vsrl_vx_u8mf2(v2660, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2682 = __riscv_vand_vx_u8mf2(v2681, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2683 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2682);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2684 = v21 + 232;
      const int8_t* v2685 = (const int8_t*) v2684;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2686 = *(const int8_t *)(v2685);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2687 = __riscv_vwmacc_vx_i16m1(v2627, v2686, v2683, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2688 = v19 + 904;
      const uint8_t* v2689 = (const uint8_t*) v2688;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2690 = __riscv_vle8_v_u8mf2(v2689, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2691 = __riscv_vand_vx_u8mf2(v2690, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2692 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2691);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2693 = v21 + 136;
      const int8_t* v2694 = (const int8_t*) v2693;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2695 = *(const int8_t *)(v2694);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2696 = __riscv_vwmacc_vx_i16m1(v2636, v2695, v2692, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2697 = __riscv_vsrl_vx_u8mf2(v2690, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2698 = __riscv_vand_vx_u8mf2(v2697, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2699 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2698);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2700 = v21 + 168;
      const int8_t* v2701 = (const int8_t*) v2700;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2702 = *(const int8_t *)(v2701);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2703 = __riscv_vwmacc_vx_i16m1(v2643, v2702, v2699, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2704 = __riscv_vsrl_vx_u8mf2(v2690, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2705 = __riscv_vand_vx_u8mf2(v2704, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2706 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2705);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2707 = v21 + 200;
      const int8_t* v2708 = (const int8_t*) v2707;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2709 = *(const int8_t *)(v2708);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2710 = __riscv_vwmacc_vx_i16m1(v2650, v2709, v2706, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2711 = __riscv_vsrl_vx_u8mf2(v2690, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2712 = __riscv_vand_vx_u8mf2(v2711, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2713 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2712);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2714 = v21 + 232;
      const int8_t* v2715 = (const int8_t*) v2714;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2716 = *(const int8_t *)(v2715);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2717 = __riscv_vwmacc_vx_i16m1(v2657, v2716, v2713, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2718 = v19 + 912;
      const uint8_t* v2719 = (const uint8_t*) v2718;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2720 = __riscv_vle8_v_u8mf2(v2719, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2721 = __riscv_vand_vx_u8mf2(v2720, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2722 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2721);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2723 = v21 + 137;
      const int8_t* v2724 = (const int8_t*) v2723;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2725 = *(const int8_t *)(v2724);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2726 = __riscv_vwmacc_vx_i16m1(v2666, v2725, v2722, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2727 = __riscv_vsrl_vx_u8mf2(v2720, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2728 = __riscv_vand_vx_u8mf2(v2727, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2729 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2728);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2730 = v21 + 169;
      const int8_t* v2731 = (const int8_t*) v2730;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2732 = *(const int8_t *)(v2731);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2733 = __riscv_vwmacc_vx_i16m1(v2673, v2732, v2729, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2734 = __riscv_vsrl_vx_u8mf2(v2720, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2735 = __riscv_vand_vx_u8mf2(v2734, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2736 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2735);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2737 = v21 + 201;
      const int8_t* v2738 = (const int8_t*) v2737;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2739 = *(const int8_t *)(v2738);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2740 = __riscv_vwmacc_vx_i16m1(v2680, v2739, v2736, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2741 = __riscv_vsrl_vx_u8mf2(v2720, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2742 = __riscv_vand_vx_u8mf2(v2741, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2743 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2742);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2744 = v21 + 233;
      const int8_t* v2745 = (const int8_t*) v2744;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2746 = *(const int8_t *)(v2745);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2747 = __riscv_vwmacc_vx_i16m1(v2687, v2746, v2743, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2748 = v19 + 920;
      const uint8_t* v2749 = (const uint8_t*) v2748;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2750 = __riscv_vle8_v_u8mf2(v2749, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2751 = __riscv_vand_vx_u8mf2(v2750, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2752 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2751);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2753 = v21 + 137;
      const int8_t* v2754 = (const int8_t*) v2753;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2755 = *(const int8_t *)(v2754);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2756 = __riscv_vwmacc_vx_i16m1(v2696, v2755, v2752, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2757 = __riscv_vsrl_vx_u8mf2(v2750, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2758 = __riscv_vand_vx_u8mf2(v2757, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2759 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2758);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2760 = v21 + 169;
      const int8_t* v2761 = (const int8_t*) v2760;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2762 = *(const int8_t *)(v2761);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2763 = __riscv_vwmacc_vx_i16m1(v2703, v2762, v2759, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2764 = __riscv_vsrl_vx_u8mf2(v2750, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2765 = __riscv_vand_vx_u8mf2(v2764, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2766 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2765);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2767 = v21 + 201;
      const int8_t* v2768 = (const int8_t*) v2767;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2769 = *(const int8_t *)(v2768);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2770 = __riscv_vwmacc_vx_i16m1(v2710, v2769, v2766, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2771 = __riscv_vsrl_vx_u8mf2(v2750, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2772 = __riscv_vand_vx_u8mf2(v2771, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2773 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2772);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2774 = v21 + 233;
      const int8_t* v2775 = (const int8_t*) v2774;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2776 = *(const int8_t *)(v2775);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2777 = __riscv_vwmacc_vx_i16m1(v2717, v2776, v2773, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2778 = v19 + 928;
      const uint8_t* v2779 = (const uint8_t*) v2778;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2780 = __riscv_vle8_v_u8mf2(v2779, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2781 = __riscv_vand_vx_u8mf2(v2780, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2782 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2781);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2783 = v21 + 138;
      const int8_t* v2784 = (const int8_t*) v2783;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2785 = *(const int8_t *)(v2784);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2786 = __riscv_vwmacc_vx_i16m1(v2726, v2785, v2782, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2787 = __riscv_vsrl_vx_u8mf2(v2780, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2788 = __riscv_vand_vx_u8mf2(v2787, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2789 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2788);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2790 = v21 + 170;
      const int8_t* v2791 = (const int8_t*) v2790;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2792 = *(const int8_t *)(v2791);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2793 = __riscv_vwmacc_vx_i16m1(v2733, v2792, v2789, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2794 = __riscv_vsrl_vx_u8mf2(v2780, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2795 = __riscv_vand_vx_u8mf2(v2794, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2796 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2795);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2797 = v21 + 202;
      const int8_t* v2798 = (const int8_t*) v2797;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2799 = *(const int8_t *)(v2798);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2800 = __riscv_vwmacc_vx_i16m1(v2740, v2799, v2796, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2801 = __riscv_vsrl_vx_u8mf2(v2780, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2802 = __riscv_vand_vx_u8mf2(v2801, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2803 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2802);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2804 = v21 + 234;
      const int8_t* v2805 = (const int8_t*) v2804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2806 = *(const int8_t *)(v2805);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2807 = __riscv_vwmacc_vx_i16m1(v2747, v2806, v2803, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2808 = v19 + 936;
      const uint8_t* v2809 = (const uint8_t*) v2808;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2810 = __riscv_vle8_v_u8mf2(v2809, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2811 = __riscv_vand_vx_u8mf2(v2810, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2812 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2811);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2813 = v21 + 138;
      const int8_t* v2814 = (const int8_t*) v2813;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2815 = *(const int8_t *)(v2814);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2816 = __riscv_vwmacc_vx_i16m1(v2756, v2815, v2812, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2817 = __riscv_vsrl_vx_u8mf2(v2810, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2818 = __riscv_vand_vx_u8mf2(v2817, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2819 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2818);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2820 = v21 + 170;
      const int8_t* v2821 = (const int8_t*) v2820;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2822 = *(const int8_t *)(v2821);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2823 = __riscv_vwmacc_vx_i16m1(v2763, v2822, v2819, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2824 = __riscv_vsrl_vx_u8mf2(v2810, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2825 = __riscv_vand_vx_u8mf2(v2824, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2826 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2825);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2827 = v21 + 202;
      const int8_t* v2828 = (const int8_t*) v2827;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2829 = *(const int8_t *)(v2828);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2830 = __riscv_vwmacc_vx_i16m1(v2770, v2829, v2826, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2831 = __riscv_vsrl_vx_u8mf2(v2810, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2832 = __riscv_vand_vx_u8mf2(v2831, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2833 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2832);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2834 = v21 + 234;
      const int8_t* v2835 = (const int8_t*) v2834;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2836 = *(const int8_t *)(v2835);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2837 = __riscv_vwmacc_vx_i16m1(v2777, v2836, v2833, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2838 = v19 + 944;
      const uint8_t* v2839 = (const uint8_t*) v2838;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2840 = __riscv_vle8_v_u8mf2(v2839, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2841 = __riscv_vand_vx_u8mf2(v2840, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2842 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2841);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2843 = v21 + 139;
      const int8_t* v2844 = (const int8_t*) v2843;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2845 = *(const int8_t *)(v2844);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2846 = __riscv_vwmacc_vx_i16m1(v2786, v2845, v2842, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2847 = __riscv_vsrl_vx_u8mf2(v2840, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2848 = __riscv_vand_vx_u8mf2(v2847, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2849 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2848);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2850 = v21 + 171;
      const int8_t* v2851 = (const int8_t*) v2850;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2852 = *(const int8_t *)(v2851);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2853 = __riscv_vwmacc_vx_i16m1(v2793, v2852, v2849, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2854 = __riscv_vsrl_vx_u8mf2(v2840, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2855 = __riscv_vand_vx_u8mf2(v2854, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2856 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2855);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2857 = v21 + 203;
      const int8_t* v2858 = (const int8_t*) v2857;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2859 = *(const int8_t *)(v2858);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2860 = __riscv_vwmacc_vx_i16m1(v2800, v2859, v2856, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2861 = __riscv_vsrl_vx_u8mf2(v2840, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2862 = __riscv_vand_vx_u8mf2(v2861, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2863 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2862);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2864 = v21 + 235;
      const int8_t* v2865 = (const int8_t*) v2864;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2866 = *(const int8_t *)(v2865);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2867 = __riscv_vwmacc_vx_i16m1(v2807, v2866, v2863, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2868 = v19 + 952;
      const uint8_t* v2869 = (const uint8_t*) v2868;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2870 = __riscv_vle8_v_u8mf2(v2869, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2871 = __riscv_vand_vx_u8mf2(v2870, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2872 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2871);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2873 = v21 + 139;
      const int8_t* v2874 = (const int8_t*) v2873;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2875 = *(const int8_t *)(v2874);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2876 = __riscv_vwmacc_vx_i16m1(v2816, v2875, v2872, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2877 = __riscv_vsrl_vx_u8mf2(v2870, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2878 = __riscv_vand_vx_u8mf2(v2877, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2879 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2878);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2880 = v21 + 171;
      const int8_t* v2881 = (const int8_t*) v2880;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2882 = *(const int8_t *)(v2881);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2883 = __riscv_vwmacc_vx_i16m1(v2823, v2882, v2879, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2884 = __riscv_vsrl_vx_u8mf2(v2870, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2885 = __riscv_vand_vx_u8mf2(v2884, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2886 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2885);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2887 = v21 + 203;
      const int8_t* v2888 = (const int8_t*) v2887;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2889 = *(const int8_t *)(v2888);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2890 = __riscv_vwmacc_vx_i16m1(v2830, v2889, v2886, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2891 = __riscv_vsrl_vx_u8mf2(v2870, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2892 = __riscv_vand_vx_u8mf2(v2891, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2893 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2892);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2894 = v21 + 235;
      const int8_t* v2895 = (const int8_t*) v2894;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2896 = *(const int8_t *)(v2895);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2897 = __riscv_vwmacc_vx_i16m1(v2837, v2896, v2893, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2898 = v19 + 960;
      const uint8_t* v2899 = (const uint8_t*) v2898;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2900 = __riscv_vle8_v_u8mf2(v2899, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2901 = __riscv_vand_vx_u8mf2(v2900, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2902 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2901);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2903 = v21 + 140;
      const int8_t* v2904 = (const int8_t*) v2903;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2905 = *(const int8_t *)(v2904);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2906 = __riscv_vwmacc_vx_i16m1(v2846, v2905, v2902, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2907 = __riscv_vsrl_vx_u8mf2(v2900, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2908 = __riscv_vand_vx_u8mf2(v2907, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2909 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2908);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2910 = v21 + 172;
      const int8_t* v2911 = (const int8_t*) v2910;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2912 = *(const int8_t *)(v2911);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2913 = __riscv_vwmacc_vx_i16m1(v2853, v2912, v2909, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2914 = __riscv_vsrl_vx_u8mf2(v2900, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2915 = __riscv_vand_vx_u8mf2(v2914, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2916 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2915);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2917 = v21 + 204;
      const int8_t* v2918 = (const int8_t*) v2917;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2919 = *(const int8_t *)(v2918);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2920 = __riscv_vwmacc_vx_i16m1(v2860, v2919, v2916, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2921 = __riscv_vsrl_vx_u8mf2(v2900, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2922 = __riscv_vand_vx_u8mf2(v2921, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2923 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2922);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2924 = v21 + 236;
      const int8_t* v2925 = (const int8_t*) v2924;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2926 = *(const int8_t *)(v2925);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2927 = __riscv_vwmacc_vx_i16m1(v2867, v2926, v2923, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2928 = v19 + 968;
      const uint8_t* v2929 = (const uint8_t*) v2928;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2930 = __riscv_vle8_v_u8mf2(v2929, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2931 = __riscv_vand_vx_u8mf2(v2930, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2932 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2931);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2933 = v21 + 140;
      const int8_t* v2934 = (const int8_t*) v2933;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2935 = *(const int8_t *)(v2934);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2936 = __riscv_vwmacc_vx_i16m1(v2876, v2935, v2932, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2937 = __riscv_vsrl_vx_u8mf2(v2930, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2938 = __riscv_vand_vx_u8mf2(v2937, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2939 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2938);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2940 = v21 + 172;
      const int8_t* v2941 = (const int8_t*) v2940;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2942 = *(const int8_t *)(v2941);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2943 = __riscv_vwmacc_vx_i16m1(v2883, v2942, v2939, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2944 = __riscv_vsrl_vx_u8mf2(v2930, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2945 = __riscv_vand_vx_u8mf2(v2944, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2946 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2945);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2947 = v21 + 204;
      const int8_t* v2948 = (const int8_t*) v2947;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2949 = *(const int8_t *)(v2948);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2950 = __riscv_vwmacc_vx_i16m1(v2890, v2949, v2946, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2951 = __riscv_vsrl_vx_u8mf2(v2930, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2952 = __riscv_vand_vx_u8mf2(v2951, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2953 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2952);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2954 = v21 + 236;
      const int8_t* v2955 = (const int8_t*) v2954;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2956 = *(const int8_t *)(v2955);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2957 = __riscv_vwmacc_vx_i16m1(v2897, v2956, v2953, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2958 = v19 + 976;
      const uint8_t* v2959 = (const uint8_t*) v2958;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2960 = __riscv_vle8_v_u8mf2(v2959, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2961 = __riscv_vand_vx_u8mf2(v2960, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2962 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2961);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2963 = v21 + 141;
      const int8_t* v2964 = (const int8_t*) v2963;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2965 = *(const int8_t *)(v2964);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2966 = __riscv_vwmacc_vx_i16m1(v2906, v2965, v2962, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2967 = __riscv_vsrl_vx_u8mf2(v2960, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2968 = __riscv_vand_vx_u8mf2(v2967, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2969 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2968);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2970 = v21 + 173;
      const int8_t* v2971 = (const int8_t*) v2970;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2972 = *(const int8_t *)(v2971);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2973 = __riscv_vwmacc_vx_i16m1(v2913, v2972, v2969, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2974 = __riscv_vsrl_vx_u8mf2(v2960, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2975 = __riscv_vand_vx_u8mf2(v2974, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2976 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2975);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2977 = v21 + 205;
      const int8_t* v2978 = (const int8_t*) v2977;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2979 = *(const int8_t *)(v2978);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2980 = __riscv_vwmacc_vx_i16m1(v2920, v2979, v2976, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2981 = __riscv_vsrl_vx_u8mf2(v2960, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2982 = __riscv_vand_vx_u8mf2(v2981, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2983 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2982);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2984 = v21 + 237;
      const int8_t* v2985 = (const int8_t*) v2984;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2986 = *(const int8_t *)(v2985);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2987 = __riscv_vwmacc_vx_i16m1(v2927, v2986, v2983, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v2988 = v19 + 984;
      const uint8_t* v2989 = (const uint8_t*) v2988;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2990 = __riscv_vle8_v_u8mf2(v2989, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2991 = __riscv_vand_vx_u8mf2(v2990, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2992 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2991);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2993 = v21 + 141;
      const int8_t* v2994 = (const int8_t*) v2993;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2995 = *(const int8_t *)(v2994);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2996 = __riscv_vwmacc_vx_i16m1(v2936, v2995, v2992, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2997 = __riscv_vsrl_vx_u8mf2(v2990, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2998 = __riscv_vand_vx_u8mf2(v2997, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2999 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2998);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3000 = v21 + 173;
      const int8_t* v3001 = (const int8_t*) v3000;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3002 = *(const int8_t *)(v3001);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3003 = __riscv_vwmacc_vx_i16m1(v2943, v3002, v2999, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3004 = __riscv_vsrl_vx_u8mf2(v2990, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3005 = __riscv_vand_vx_u8mf2(v3004, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3006 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3005);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3007 = v21 + 205;
      const int8_t* v3008 = (const int8_t*) v3007;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3009 = *(const int8_t *)(v3008);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3010 = __riscv_vwmacc_vx_i16m1(v2950, v3009, v3006, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3011 = __riscv_vsrl_vx_u8mf2(v2990, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3012 = __riscv_vand_vx_u8mf2(v3011, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3013 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3012);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3014 = v21 + 237;
      const int8_t* v3015 = (const int8_t*) v3014;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3016 = *(const int8_t *)(v3015);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3017 = __riscv_vwmacc_vx_i16m1(v2957, v3016, v3013, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3018 = v19 + 992;
      const uint8_t* v3019 = (const uint8_t*) v3018;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3020 = __riscv_vle8_v_u8mf2(v3019, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3021 = __riscv_vand_vx_u8mf2(v3020, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3022 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3021);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3023 = v21 + 142;
      const int8_t* v3024 = (const int8_t*) v3023;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3025 = *(const int8_t *)(v3024);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3026 = __riscv_vwmacc_vx_i16m1(v2966, v3025, v3022, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3027 = __riscv_vsrl_vx_u8mf2(v3020, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3028 = __riscv_vand_vx_u8mf2(v3027, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3029 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3028);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3030 = v21 + 174;
      const int8_t* v3031 = (const int8_t*) v3030;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3032 = *(const int8_t *)(v3031);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3033 = __riscv_vwmacc_vx_i16m1(v2973, v3032, v3029, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3034 = __riscv_vsrl_vx_u8mf2(v3020, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3035 = __riscv_vand_vx_u8mf2(v3034, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3036 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3035);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3037 = v21 + 206;
      const int8_t* v3038 = (const int8_t*) v3037;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3039 = *(const int8_t *)(v3038);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3040 = __riscv_vwmacc_vx_i16m1(v2980, v3039, v3036, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3041 = __riscv_vsrl_vx_u8mf2(v3020, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3042 = __riscv_vand_vx_u8mf2(v3041, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3043 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3042);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3044 = v21 + 238;
      const int8_t* v3045 = (const int8_t*) v3044;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3046 = *(const int8_t *)(v3045);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3047 = __riscv_vwmacc_vx_i16m1(v2987, v3046, v3043, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3048 = v19 + 1000;
      const uint8_t* v3049 = (const uint8_t*) v3048;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3050 = __riscv_vle8_v_u8mf2(v3049, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3051 = __riscv_vand_vx_u8mf2(v3050, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3052 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3051);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3053 = v21 + 142;
      const int8_t* v3054 = (const int8_t*) v3053;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3055 = *(const int8_t *)(v3054);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3056 = __riscv_vwmacc_vx_i16m1(v2996, v3055, v3052, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3057 = __riscv_vsrl_vx_u8mf2(v3050, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3058 = __riscv_vand_vx_u8mf2(v3057, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3059 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3058);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3060 = v21 + 174;
      const int8_t* v3061 = (const int8_t*) v3060;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3062 = *(const int8_t *)(v3061);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3063 = __riscv_vwmacc_vx_i16m1(v3003, v3062, v3059, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3064 = __riscv_vsrl_vx_u8mf2(v3050, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3065 = __riscv_vand_vx_u8mf2(v3064, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3066 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3065);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3067 = v21 + 206;
      const int8_t* v3068 = (const int8_t*) v3067;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3069 = *(const int8_t *)(v3068);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3070 = __riscv_vwmacc_vx_i16m1(v3010, v3069, v3066, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3071 = __riscv_vsrl_vx_u8mf2(v3050, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3072 = __riscv_vand_vx_u8mf2(v3071, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3073 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3072);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3074 = v21 + 238;
      const int8_t* v3075 = (const int8_t*) v3074;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3076 = *(const int8_t *)(v3075);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3077 = __riscv_vwmacc_vx_i16m1(v3017, v3076, v3073, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3078 = v19 + 1008;
      const uint8_t* v3079 = (const uint8_t*) v3078;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3080 = __riscv_vle8_v_u8mf2(v3079, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3081 = __riscv_vand_vx_u8mf2(v3080, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3082 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3081);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3083 = v21 + 143;
      const int8_t* v3084 = (const int8_t*) v3083;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3085 = *(const int8_t *)(v3084);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3086 = __riscv_vwmacc_vx_i16m1(v3026, v3085, v3082, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3087 = __riscv_vsrl_vx_u8mf2(v3080, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3088 = __riscv_vand_vx_u8mf2(v3087, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3089 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3088);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3090 = v21 + 175;
      const int8_t* v3091 = (const int8_t*) v3090;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3092 = *(const int8_t *)(v3091);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3093 = __riscv_vwmacc_vx_i16m1(v3033, v3092, v3089, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3094 = __riscv_vsrl_vx_u8mf2(v3080, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3095 = __riscv_vand_vx_u8mf2(v3094, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3096 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3095);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3097 = v21 + 207;
      const int8_t* v3098 = (const int8_t*) v3097;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3099 = *(const int8_t *)(v3098);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3100 = __riscv_vwmacc_vx_i16m1(v3040, v3099, v3096, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3101 = __riscv_vsrl_vx_u8mf2(v3080, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3102 = __riscv_vand_vx_u8mf2(v3101, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3103 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3102);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3104 = v21 + 239;
      const int8_t* v3105 = (const int8_t*) v3104;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3106 = *(const int8_t *)(v3105);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3107 = __riscv_vwmacc_vx_i16m1(v3047, v3106, v3103, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3108 = v19 + 1016;
      const uint8_t* v3109 = (const uint8_t*) v3108;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3110 = __riscv_vle8_v_u8mf2(v3109, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3111 = __riscv_vand_vx_u8mf2(v3110, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3112 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3111);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3113 = v21 + 143;
      const int8_t* v3114 = (const int8_t*) v3113;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3115 = *(const int8_t *)(v3114);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3116 = __riscv_vwmacc_vx_i16m1(v3056, v3115, v3112, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3117 = __riscv_vsrl_vx_u8mf2(v3110, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3118 = __riscv_vand_vx_u8mf2(v3117, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3119 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3118);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3120 = v21 + 175;
      const int8_t* v3121 = (const int8_t*) v3120;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3122 = *(const int8_t *)(v3121);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3123 = __riscv_vwmacc_vx_i16m1(v3063, v3122, v3119, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3124 = __riscv_vsrl_vx_u8mf2(v3110, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3125 = __riscv_vand_vx_u8mf2(v3124, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3126 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3125);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3127 = v21 + 207;
      const int8_t* v3128 = (const int8_t*) v3127;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3129 = *(const int8_t *)(v3128);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3130 = __riscv_vwmacc_vx_i16m1(v3070, v3129, v3126, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3131 = __riscv_vsrl_vx_u8mf2(v3110, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3132 = __riscv_vand_vx_u8mf2(v3131, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3133 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3132);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3134 = v21 + 239;
      const int8_t* v3135 = (const int8_t*) v3134;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3136 = *(const int8_t *)(v3135);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3137 = __riscv_vwmacc_vx_i16m1(v3077, v3136, v3133, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3138 = v19 + 1024;
      const uint8_t* v3139 = (const uint8_t*) v3138;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3140 = __riscv_vle8_v_u8mf2(v3139, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3141 = __riscv_vand_vx_u8mf2(v3140, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3142 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3141);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3143 = v21 + 144;
      const int8_t* v3144 = (const int8_t*) v3143;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3145 = *(const int8_t *)(v3144);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3146 = __riscv_vwmacc_vx_i16m1(v3086, v3145, v3142, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3147 = __riscv_vsrl_vx_u8mf2(v3140, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3148 = __riscv_vand_vx_u8mf2(v3147, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3149 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3148);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3150 = v21 + 176;
      const int8_t* v3151 = (const int8_t*) v3150;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3152 = *(const int8_t *)(v3151);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3153 = __riscv_vwmacc_vx_i16m1(v3093, v3152, v3149, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3154 = __riscv_vsrl_vx_u8mf2(v3140, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3155 = __riscv_vand_vx_u8mf2(v3154, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3156 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3155);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3157 = v21 + 208;
      const int8_t* v3158 = (const int8_t*) v3157;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3159 = *(const int8_t *)(v3158);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3160 = __riscv_vwmacc_vx_i16m1(v3100, v3159, v3156, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3161 = __riscv_vsrl_vx_u8mf2(v3140, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3162 = __riscv_vand_vx_u8mf2(v3161, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3163 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3162);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3164 = v21 + 240;
      const int8_t* v3165 = (const int8_t*) v3164;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3166 = *(const int8_t *)(v3165);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3167 = __riscv_vwmacc_vx_i16m1(v3107, v3166, v3163, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3168 = v19 + 1032;
      const uint8_t* v3169 = (const uint8_t*) v3168;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3170 = __riscv_vle8_v_u8mf2(v3169, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3171 = __riscv_vand_vx_u8mf2(v3170, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3172 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3171);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3173 = v21 + 144;
      const int8_t* v3174 = (const int8_t*) v3173;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3175 = *(const int8_t *)(v3174);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3176 = __riscv_vwmacc_vx_i16m1(v3116, v3175, v3172, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3177 = __riscv_vsrl_vx_u8mf2(v3170, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3178 = __riscv_vand_vx_u8mf2(v3177, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3179 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3178);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3180 = v21 + 176;
      const int8_t* v3181 = (const int8_t*) v3180;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3182 = *(const int8_t *)(v3181);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3183 = __riscv_vwmacc_vx_i16m1(v3123, v3182, v3179, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3184 = __riscv_vsrl_vx_u8mf2(v3170, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3185 = __riscv_vand_vx_u8mf2(v3184, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3186 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3185);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3187 = v21 + 208;
      const int8_t* v3188 = (const int8_t*) v3187;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3189 = *(const int8_t *)(v3188);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3190 = __riscv_vwmacc_vx_i16m1(v3130, v3189, v3186, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3191 = __riscv_vsrl_vx_u8mf2(v3170, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3192 = __riscv_vand_vx_u8mf2(v3191, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3193 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3192);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3194 = v21 + 240;
      const int8_t* v3195 = (const int8_t*) v3194;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3196 = *(const int8_t *)(v3195);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3197 = __riscv_vwmacc_vx_i16m1(v3137, v3196, v3193, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3198 = v19 + 1040;
      const uint8_t* v3199 = (const uint8_t*) v3198;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3200 = __riscv_vle8_v_u8mf2(v3199, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3201 = __riscv_vand_vx_u8mf2(v3200, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3202 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3201);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3203 = v21 + 145;
      const int8_t* v3204 = (const int8_t*) v3203;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3205 = *(const int8_t *)(v3204);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3206 = __riscv_vwmacc_vx_i16m1(v3146, v3205, v3202, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3207 = __riscv_vsrl_vx_u8mf2(v3200, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3208 = __riscv_vand_vx_u8mf2(v3207, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3209 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3208);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3210 = v21 + 177;
      const int8_t* v3211 = (const int8_t*) v3210;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3212 = *(const int8_t *)(v3211);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3213 = __riscv_vwmacc_vx_i16m1(v3153, v3212, v3209, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3214 = __riscv_vsrl_vx_u8mf2(v3200, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3215 = __riscv_vand_vx_u8mf2(v3214, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3216 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3215);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3217 = v21 + 209;
      const int8_t* v3218 = (const int8_t*) v3217;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3219 = *(const int8_t *)(v3218);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3220 = __riscv_vwmacc_vx_i16m1(v3160, v3219, v3216, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3221 = __riscv_vsrl_vx_u8mf2(v3200, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3222 = __riscv_vand_vx_u8mf2(v3221, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3223 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3222);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3224 = v21 + 241;
      const int8_t* v3225 = (const int8_t*) v3224;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3226 = *(const int8_t *)(v3225);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3227 = __riscv_vwmacc_vx_i16m1(v3167, v3226, v3223, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3228 = v19 + 1048;
      const uint8_t* v3229 = (const uint8_t*) v3228;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3230 = __riscv_vle8_v_u8mf2(v3229, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3231 = __riscv_vand_vx_u8mf2(v3230, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3232 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3231);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3233 = v21 + 145;
      const int8_t* v3234 = (const int8_t*) v3233;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3235 = *(const int8_t *)(v3234);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3236 = __riscv_vwmacc_vx_i16m1(v3176, v3235, v3232, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3237 = __riscv_vsrl_vx_u8mf2(v3230, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3238 = __riscv_vand_vx_u8mf2(v3237, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3239 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3238);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3240 = v21 + 177;
      const int8_t* v3241 = (const int8_t*) v3240;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3242 = *(const int8_t *)(v3241);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3243 = __riscv_vwmacc_vx_i16m1(v3183, v3242, v3239, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3244 = __riscv_vsrl_vx_u8mf2(v3230, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3245 = __riscv_vand_vx_u8mf2(v3244, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3246 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3245);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3247 = v21 + 209;
      const int8_t* v3248 = (const int8_t*) v3247;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3249 = *(const int8_t *)(v3248);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3250 = __riscv_vwmacc_vx_i16m1(v3190, v3249, v3246, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3251 = __riscv_vsrl_vx_u8mf2(v3230, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3252 = __riscv_vand_vx_u8mf2(v3251, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3253 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3252);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3254 = v21 + 241;
      const int8_t* v3255 = (const int8_t*) v3254;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3256 = *(const int8_t *)(v3255);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3257 = __riscv_vwmacc_vx_i16m1(v3197, v3256, v3253, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3258 = v19 + 1056;
      const uint8_t* v3259 = (const uint8_t*) v3258;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3260 = __riscv_vle8_v_u8mf2(v3259, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3261 = __riscv_vand_vx_u8mf2(v3260, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3262 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3261);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3263 = v21 + 146;
      const int8_t* v3264 = (const int8_t*) v3263;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3265 = *(const int8_t *)(v3264);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3266 = __riscv_vwmacc_vx_i16m1(v3206, v3265, v3262, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3267 = __riscv_vsrl_vx_u8mf2(v3260, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3268 = __riscv_vand_vx_u8mf2(v3267, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3269 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3268);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3270 = v21 + 178;
      const int8_t* v3271 = (const int8_t*) v3270;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3272 = *(const int8_t *)(v3271);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3273 = __riscv_vwmacc_vx_i16m1(v3213, v3272, v3269, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3274 = __riscv_vsrl_vx_u8mf2(v3260, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3275 = __riscv_vand_vx_u8mf2(v3274, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3276 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3275);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3277 = v21 + 210;
      const int8_t* v3278 = (const int8_t*) v3277;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3279 = *(const int8_t *)(v3278);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3280 = __riscv_vwmacc_vx_i16m1(v3220, v3279, v3276, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3281 = __riscv_vsrl_vx_u8mf2(v3260, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3282 = __riscv_vand_vx_u8mf2(v3281, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3283 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3282);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3284 = v21 + 242;
      const int8_t* v3285 = (const int8_t*) v3284;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3286 = *(const int8_t *)(v3285);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3287 = __riscv_vwmacc_vx_i16m1(v3227, v3286, v3283, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3288 = v19 + 1064;
      const uint8_t* v3289 = (const uint8_t*) v3288;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3290 = __riscv_vle8_v_u8mf2(v3289, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3291 = __riscv_vand_vx_u8mf2(v3290, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3292 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3291);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3293 = v21 + 146;
      const int8_t* v3294 = (const int8_t*) v3293;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3295 = *(const int8_t *)(v3294);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3296 = __riscv_vwmacc_vx_i16m1(v3236, v3295, v3292, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3297 = __riscv_vsrl_vx_u8mf2(v3290, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3298 = __riscv_vand_vx_u8mf2(v3297, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3299 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3298);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3300 = v21 + 178;
      const int8_t* v3301 = (const int8_t*) v3300;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3302 = *(const int8_t *)(v3301);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3303 = __riscv_vwmacc_vx_i16m1(v3243, v3302, v3299, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3304 = __riscv_vsrl_vx_u8mf2(v3290, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3305 = __riscv_vand_vx_u8mf2(v3304, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3306 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3305);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3307 = v21 + 210;
      const int8_t* v3308 = (const int8_t*) v3307;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3309 = *(const int8_t *)(v3308);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3310 = __riscv_vwmacc_vx_i16m1(v3250, v3309, v3306, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3311 = __riscv_vsrl_vx_u8mf2(v3290, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3312 = __riscv_vand_vx_u8mf2(v3311, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3313 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3312);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3314 = v21 + 242;
      const int8_t* v3315 = (const int8_t*) v3314;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3316 = *(const int8_t *)(v3315);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3317 = __riscv_vwmacc_vx_i16m1(v3257, v3316, v3313, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3318 = v19 + 1072;
      const uint8_t* v3319 = (const uint8_t*) v3318;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3320 = __riscv_vle8_v_u8mf2(v3319, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3321 = __riscv_vand_vx_u8mf2(v3320, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3322 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3321);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3323 = v21 + 147;
      const int8_t* v3324 = (const int8_t*) v3323;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3325 = *(const int8_t *)(v3324);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3326 = __riscv_vwmacc_vx_i16m1(v3266, v3325, v3322, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3327 = __riscv_vsrl_vx_u8mf2(v3320, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3328 = __riscv_vand_vx_u8mf2(v3327, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3329 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3328);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3330 = v21 + 179;
      const int8_t* v3331 = (const int8_t*) v3330;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3332 = *(const int8_t *)(v3331);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3333 = __riscv_vwmacc_vx_i16m1(v3273, v3332, v3329, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3334 = __riscv_vsrl_vx_u8mf2(v3320, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3335 = __riscv_vand_vx_u8mf2(v3334, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3336 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3335);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3337 = v21 + 211;
      const int8_t* v3338 = (const int8_t*) v3337;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3339 = *(const int8_t *)(v3338);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3340 = __riscv_vwmacc_vx_i16m1(v3280, v3339, v3336, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3341 = __riscv_vsrl_vx_u8mf2(v3320, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3342 = __riscv_vand_vx_u8mf2(v3341, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3343 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3342);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3344 = v21 + 243;
      const int8_t* v3345 = (const int8_t*) v3344;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3346 = *(const int8_t *)(v3345);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3347 = __riscv_vwmacc_vx_i16m1(v3287, v3346, v3343, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3348 = v19 + 1080;
      const uint8_t* v3349 = (const uint8_t*) v3348;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3350 = __riscv_vle8_v_u8mf2(v3349, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3351 = __riscv_vand_vx_u8mf2(v3350, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3352 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3351);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3353 = v21 + 147;
      const int8_t* v3354 = (const int8_t*) v3353;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3355 = *(const int8_t *)(v3354);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3356 = __riscv_vwmacc_vx_i16m1(v3296, v3355, v3352, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3357 = __riscv_vsrl_vx_u8mf2(v3350, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3358 = __riscv_vand_vx_u8mf2(v3357, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3359 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3358);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3360 = v21 + 179;
      const int8_t* v3361 = (const int8_t*) v3360;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3362 = *(const int8_t *)(v3361);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3363 = __riscv_vwmacc_vx_i16m1(v3303, v3362, v3359, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3364 = __riscv_vsrl_vx_u8mf2(v3350, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3365 = __riscv_vand_vx_u8mf2(v3364, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3366 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3365);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3367 = v21 + 211;
      const int8_t* v3368 = (const int8_t*) v3367;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3369 = *(const int8_t *)(v3368);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3370 = __riscv_vwmacc_vx_i16m1(v3310, v3369, v3366, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3371 = __riscv_vsrl_vx_u8mf2(v3350, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3372 = __riscv_vand_vx_u8mf2(v3371, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3373 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3372);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3374 = v21 + 243;
      const int8_t* v3375 = (const int8_t*) v3374;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3376 = *(const int8_t *)(v3375);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3377 = __riscv_vwmacc_vx_i16m1(v3317, v3376, v3373, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v3378 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3379 = __riscv_vwmacc_vv_i32m2(v3378, v2216, v3326, 8);
      v34 = v3379;
      vint32m2_t v3380 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3381 = __riscv_vwmacc_vv_i32m2(v3380, v2234, v3333, 8);
      v34 = v3381;
      vint32m2_t v3382 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3383 = __riscv_vwmacc_vv_i32m2(v3382, v2252, v3340, 8);
      v34 = v3383;
      vint32m2_t v3384 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3385 = __riscv_vwmacc_vv_i32m2(v3384, v2270, v3347, 8);
      v34 = v3385;
      vint32m2_t v3386 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3387 = __riscv_vwmacc_vv_i32m2(v3386, v2288, v3356, 8);
      v38 = v3387;
      vint32m2_t v3388 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3389 = __riscv_vwmacc_vv_i32m2(v3388, v2306, v3363, 8);
      v38 = v3389;
      vint32m2_t v3390 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3391 = __riscv_vwmacc_vv_i32m2(v3390, v2324, v3370, 8);
      v38 = v3391;
      vint32m2_t v3392 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3393 = __riscv_vwmacc_vv_i32m2(v3392, v2342, v3377, 8);
      v38 = v3393;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3394 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3395 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3396 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3397 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3398 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3399 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3400 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3401 = __riscv_vmv_v_x_i16m1(0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3402 = v19 + 1088;
      const uint8_t* v3403 = (const uint8_t*) v3402;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3404 = __riscv_vle8_v_u8mf2(v3403, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3405 = __riscv_vand_vx_u8mf2(v3404, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3406 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3405);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3407 = v21 + 148;
      const int8_t* v3408 = (const int8_t*) v3407;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3409 = *(const int8_t *)(v3408);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3410 = __riscv_vwmacc_vx_i16m1(v3394, v3409, v3406, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3411 = __riscv_vsrl_vx_u8mf2(v3404, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3412 = __riscv_vand_vx_u8mf2(v3411, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3413 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3412);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3414 = v21 + 180;
      const int8_t* v3415 = (const int8_t*) v3414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3416 = *(const int8_t *)(v3415);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3417 = __riscv_vwmacc_vx_i16m1(v3395, v3416, v3413, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3418 = __riscv_vsrl_vx_u8mf2(v3404, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3419 = __riscv_vand_vx_u8mf2(v3418, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3420 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3419);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3421 = v21 + 212;
      const int8_t* v3422 = (const int8_t*) v3421;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3423 = *(const int8_t *)(v3422);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3424 = __riscv_vwmacc_vx_i16m1(v3396, v3423, v3420, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3425 = __riscv_vsrl_vx_u8mf2(v3404, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3426 = __riscv_vand_vx_u8mf2(v3425, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3427 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3426);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3428 = v21 + 244;
      const int8_t* v3429 = (const int8_t*) v3428;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3430 = *(const int8_t *)(v3429);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3431 = __riscv_vwmacc_vx_i16m1(v3397, v3430, v3427, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3432 = v19 + 1096;
      const uint8_t* v3433 = (const uint8_t*) v3432;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3434 = __riscv_vle8_v_u8mf2(v3433, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3435 = __riscv_vand_vx_u8mf2(v3434, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3436 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3435);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3437 = v21 + 148;
      const int8_t* v3438 = (const int8_t*) v3437;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3439 = *(const int8_t *)(v3438);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3440 = __riscv_vwmacc_vx_i16m1(v3398, v3439, v3436, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3441 = __riscv_vsrl_vx_u8mf2(v3434, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3442 = __riscv_vand_vx_u8mf2(v3441, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3443 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3442);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3444 = v21 + 180;
      const int8_t* v3445 = (const int8_t*) v3444;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3446 = *(const int8_t *)(v3445);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3447 = __riscv_vwmacc_vx_i16m1(v3399, v3446, v3443, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3448 = __riscv_vsrl_vx_u8mf2(v3434, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3449 = __riscv_vand_vx_u8mf2(v3448, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3450 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3449);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3451 = v21 + 212;
      const int8_t* v3452 = (const int8_t*) v3451;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3453 = *(const int8_t *)(v3452);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3454 = __riscv_vwmacc_vx_i16m1(v3400, v3453, v3450, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3455 = __riscv_vsrl_vx_u8mf2(v3434, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3456 = __riscv_vand_vx_u8mf2(v3455, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3457 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3456);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3458 = v21 + 244;
      const int8_t* v3459 = (const int8_t*) v3458;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3460 = *(const int8_t *)(v3459);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3461 = __riscv_vwmacc_vx_i16m1(v3401, v3460, v3457, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3462 = v19 + 1104;
      const uint8_t* v3463 = (const uint8_t*) v3462;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3464 = __riscv_vle8_v_u8mf2(v3463, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3465 = __riscv_vand_vx_u8mf2(v3464, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3466 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3465);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3467 = v21 + 149;
      const int8_t* v3468 = (const int8_t*) v3467;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3469 = *(const int8_t *)(v3468);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3470 = __riscv_vwmacc_vx_i16m1(v3410, v3469, v3466, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3471 = __riscv_vsrl_vx_u8mf2(v3464, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3472 = __riscv_vand_vx_u8mf2(v3471, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3473 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3472);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3474 = v21 + 181;
      const int8_t* v3475 = (const int8_t*) v3474;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3476 = *(const int8_t *)(v3475);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3477 = __riscv_vwmacc_vx_i16m1(v3417, v3476, v3473, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3478 = __riscv_vsrl_vx_u8mf2(v3464, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3479 = __riscv_vand_vx_u8mf2(v3478, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3480 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3479);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3481 = v21 + 213;
      const int8_t* v3482 = (const int8_t*) v3481;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3483 = *(const int8_t *)(v3482);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3484 = __riscv_vwmacc_vx_i16m1(v3424, v3483, v3480, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3485 = __riscv_vsrl_vx_u8mf2(v3464, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3486 = __riscv_vand_vx_u8mf2(v3485, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3487 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3486);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3488 = v21 + 245;
      const int8_t* v3489 = (const int8_t*) v3488;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3490 = *(const int8_t *)(v3489);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3491 = __riscv_vwmacc_vx_i16m1(v3431, v3490, v3487, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3492 = v19 + 1112;
      const uint8_t* v3493 = (const uint8_t*) v3492;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3494 = __riscv_vle8_v_u8mf2(v3493, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3495 = __riscv_vand_vx_u8mf2(v3494, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3496 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3495);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3497 = v21 + 149;
      const int8_t* v3498 = (const int8_t*) v3497;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3499 = *(const int8_t *)(v3498);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3500 = __riscv_vwmacc_vx_i16m1(v3440, v3499, v3496, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3501 = __riscv_vsrl_vx_u8mf2(v3494, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3502 = __riscv_vand_vx_u8mf2(v3501, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3503 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3502);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3504 = v21 + 181;
      const int8_t* v3505 = (const int8_t*) v3504;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3506 = *(const int8_t *)(v3505);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3507 = __riscv_vwmacc_vx_i16m1(v3447, v3506, v3503, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3508 = __riscv_vsrl_vx_u8mf2(v3494, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3509 = __riscv_vand_vx_u8mf2(v3508, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3510 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3509);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3511 = v21 + 213;
      const int8_t* v3512 = (const int8_t*) v3511;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3513 = *(const int8_t *)(v3512);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3514 = __riscv_vwmacc_vx_i16m1(v3454, v3513, v3510, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3515 = __riscv_vsrl_vx_u8mf2(v3494, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3516 = __riscv_vand_vx_u8mf2(v3515, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3517 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3516);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3518 = v21 + 245;
      const int8_t* v3519 = (const int8_t*) v3518;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3520 = *(const int8_t *)(v3519);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3521 = __riscv_vwmacc_vx_i16m1(v3461, v3520, v3517, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3522 = v19 + 1120;
      const uint8_t* v3523 = (const uint8_t*) v3522;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3524 = __riscv_vle8_v_u8mf2(v3523, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3525 = __riscv_vand_vx_u8mf2(v3524, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3526 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3525);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3527 = v21 + 150;
      const int8_t* v3528 = (const int8_t*) v3527;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3529 = *(const int8_t *)(v3528);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3530 = __riscv_vwmacc_vx_i16m1(v3470, v3529, v3526, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3531 = __riscv_vsrl_vx_u8mf2(v3524, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3532 = __riscv_vand_vx_u8mf2(v3531, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3533 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3532);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3534 = v21 + 182;
      const int8_t* v3535 = (const int8_t*) v3534;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3536 = *(const int8_t *)(v3535);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3537 = __riscv_vwmacc_vx_i16m1(v3477, v3536, v3533, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3538 = __riscv_vsrl_vx_u8mf2(v3524, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3539 = __riscv_vand_vx_u8mf2(v3538, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3540 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3539);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3541 = v21 + 214;
      const int8_t* v3542 = (const int8_t*) v3541;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3543 = *(const int8_t *)(v3542);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3544 = __riscv_vwmacc_vx_i16m1(v3484, v3543, v3540, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3545 = __riscv_vsrl_vx_u8mf2(v3524, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3546 = __riscv_vand_vx_u8mf2(v3545, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3547 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3546);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3548 = v21 + 246;
      const int8_t* v3549 = (const int8_t*) v3548;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3550 = *(const int8_t *)(v3549);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3551 = __riscv_vwmacc_vx_i16m1(v3491, v3550, v3547, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3552 = v19 + 1128;
      const uint8_t* v3553 = (const uint8_t*) v3552;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3554 = __riscv_vle8_v_u8mf2(v3553, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3555 = __riscv_vand_vx_u8mf2(v3554, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3556 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3555);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3557 = v21 + 150;
      const int8_t* v3558 = (const int8_t*) v3557;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3559 = *(const int8_t *)(v3558);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3560 = __riscv_vwmacc_vx_i16m1(v3500, v3559, v3556, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3561 = __riscv_vsrl_vx_u8mf2(v3554, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3562 = __riscv_vand_vx_u8mf2(v3561, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3563 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3562);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3564 = v21 + 182;
      const int8_t* v3565 = (const int8_t*) v3564;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3566 = *(const int8_t *)(v3565);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3567 = __riscv_vwmacc_vx_i16m1(v3507, v3566, v3563, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3568 = __riscv_vsrl_vx_u8mf2(v3554, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3569 = __riscv_vand_vx_u8mf2(v3568, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3570 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3569);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3571 = v21 + 214;
      const int8_t* v3572 = (const int8_t*) v3571;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3573 = *(const int8_t *)(v3572);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3574 = __riscv_vwmacc_vx_i16m1(v3514, v3573, v3570, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3575 = __riscv_vsrl_vx_u8mf2(v3554, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3576 = __riscv_vand_vx_u8mf2(v3575, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3577 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3576);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3578 = v21 + 246;
      const int8_t* v3579 = (const int8_t*) v3578;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3580 = *(const int8_t *)(v3579);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3581 = __riscv_vwmacc_vx_i16m1(v3521, v3580, v3577, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3582 = v19 + 1136;
      const uint8_t* v3583 = (const uint8_t*) v3582;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3584 = __riscv_vle8_v_u8mf2(v3583, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3585 = __riscv_vand_vx_u8mf2(v3584, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3586 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3585);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3587 = v21 + 151;
      const int8_t* v3588 = (const int8_t*) v3587;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3589 = *(const int8_t *)(v3588);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3590 = __riscv_vwmacc_vx_i16m1(v3530, v3589, v3586, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3591 = __riscv_vsrl_vx_u8mf2(v3584, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3592 = __riscv_vand_vx_u8mf2(v3591, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3593 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3592);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3594 = v21 + 183;
      const int8_t* v3595 = (const int8_t*) v3594;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3596 = *(const int8_t *)(v3595);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3597 = __riscv_vwmacc_vx_i16m1(v3537, v3596, v3593, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3598 = __riscv_vsrl_vx_u8mf2(v3584, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3599 = __riscv_vand_vx_u8mf2(v3598, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3600 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3599);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3601 = v21 + 215;
      const int8_t* v3602 = (const int8_t*) v3601;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3603 = *(const int8_t *)(v3602);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3604 = __riscv_vwmacc_vx_i16m1(v3544, v3603, v3600, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3605 = __riscv_vsrl_vx_u8mf2(v3584, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3606 = __riscv_vand_vx_u8mf2(v3605, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3607 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3606);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3608 = v21 + 247;
      const int8_t* v3609 = (const int8_t*) v3608;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3610 = *(const int8_t *)(v3609);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3611 = __riscv_vwmacc_vx_i16m1(v3551, v3610, v3607, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3612 = v19 + 1144;
      const uint8_t* v3613 = (const uint8_t*) v3612;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3614 = __riscv_vle8_v_u8mf2(v3613, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3615 = __riscv_vand_vx_u8mf2(v3614, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3616 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3615);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3617 = v21 + 151;
      const int8_t* v3618 = (const int8_t*) v3617;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3619 = *(const int8_t *)(v3618);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3620 = __riscv_vwmacc_vx_i16m1(v3560, v3619, v3616, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3621 = __riscv_vsrl_vx_u8mf2(v3614, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3622 = __riscv_vand_vx_u8mf2(v3621, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3623 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3622);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3624 = v21 + 183;
      const int8_t* v3625 = (const int8_t*) v3624;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3626 = *(const int8_t *)(v3625);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3627 = __riscv_vwmacc_vx_i16m1(v3567, v3626, v3623, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3628 = __riscv_vsrl_vx_u8mf2(v3614, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3629 = __riscv_vand_vx_u8mf2(v3628, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3630 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3629);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3631 = v21 + 215;
      const int8_t* v3632 = (const int8_t*) v3631;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3633 = *(const int8_t *)(v3632);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3634 = __riscv_vwmacc_vx_i16m1(v3574, v3633, v3630, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3635 = __riscv_vsrl_vx_u8mf2(v3614, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3636 = __riscv_vand_vx_u8mf2(v3635, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3637 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3636);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3638 = v21 + 247;
      const int8_t* v3639 = (const int8_t*) v3638;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3640 = *(const int8_t *)(v3639);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3641 = __riscv_vwmacc_vx_i16m1(v3581, v3640, v3637, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3642 = v19 + 1152;
      const uint8_t* v3643 = (const uint8_t*) v3642;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3644 = __riscv_vle8_v_u8mf2(v3643, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3645 = __riscv_vand_vx_u8mf2(v3644, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3646 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3645);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3647 = v21 + 152;
      const int8_t* v3648 = (const int8_t*) v3647;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3649 = *(const int8_t *)(v3648);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3650 = __riscv_vwmacc_vx_i16m1(v3590, v3649, v3646, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3651 = __riscv_vsrl_vx_u8mf2(v3644, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3652 = __riscv_vand_vx_u8mf2(v3651, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3653 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3652);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3654 = v21 + 184;
      const int8_t* v3655 = (const int8_t*) v3654;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3656 = *(const int8_t *)(v3655);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3657 = __riscv_vwmacc_vx_i16m1(v3597, v3656, v3653, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3658 = __riscv_vsrl_vx_u8mf2(v3644, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3659 = __riscv_vand_vx_u8mf2(v3658, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3660 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3659);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3661 = v21 + 216;
      const int8_t* v3662 = (const int8_t*) v3661;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3663 = *(const int8_t *)(v3662);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3664 = __riscv_vwmacc_vx_i16m1(v3604, v3663, v3660, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3665 = __riscv_vsrl_vx_u8mf2(v3644, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3666 = __riscv_vand_vx_u8mf2(v3665, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3667 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3666);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3668 = v21 + 248;
      const int8_t* v3669 = (const int8_t*) v3668;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3670 = *(const int8_t *)(v3669);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3671 = __riscv_vwmacc_vx_i16m1(v3611, v3670, v3667, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3672 = v19 + 1160;
      const uint8_t* v3673 = (const uint8_t*) v3672;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3674 = __riscv_vle8_v_u8mf2(v3673, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3675 = __riscv_vand_vx_u8mf2(v3674, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3676 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3675);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3677 = v21 + 152;
      const int8_t* v3678 = (const int8_t*) v3677;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3679 = *(const int8_t *)(v3678);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3680 = __riscv_vwmacc_vx_i16m1(v3620, v3679, v3676, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3681 = __riscv_vsrl_vx_u8mf2(v3674, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3682 = __riscv_vand_vx_u8mf2(v3681, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3683 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3682);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3684 = v21 + 184;
      const int8_t* v3685 = (const int8_t*) v3684;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3686 = *(const int8_t *)(v3685);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3687 = __riscv_vwmacc_vx_i16m1(v3627, v3686, v3683, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3688 = __riscv_vsrl_vx_u8mf2(v3674, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3689 = __riscv_vand_vx_u8mf2(v3688, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3690 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3689);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3691 = v21 + 216;
      const int8_t* v3692 = (const int8_t*) v3691;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3693 = *(const int8_t *)(v3692);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3694 = __riscv_vwmacc_vx_i16m1(v3634, v3693, v3690, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3695 = __riscv_vsrl_vx_u8mf2(v3674, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3696 = __riscv_vand_vx_u8mf2(v3695, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3697 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3696);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3698 = v21 + 248;
      const int8_t* v3699 = (const int8_t*) v3698;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3700 = *(const int8_t *)(v3699);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3701 = __riscv_vwmacc_vx_i16m1(v3641, v3700, v3697, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3702 = v19 + 1168;
      const uint8_t* v3703 = (const uint8_t*) v3702;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3704 = __riscv_vle8_v_u8mf2(v3703, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3705 = __riscv_vand_vx_u8mf2(v3704, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3706 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3705);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3707 = v21 + 153;
      const int8_t* v3708 = (const int8_t*) v3707;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3709 = *(const int8_t *)(v3708);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3710 = __riscv_vwmacc_vx_i16m1(v3650, v3709, v3706, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3711 = __riscv_vsrl_vx_u8mf2(v3704, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3712 = __riscv_vand_vx_u8mf2(v3711, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3713 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3712);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3714 = v21 + 185;
      const int8_t* v3715 = (const int8_t*) v3714;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3716 = *(const int8_t *)(v3715);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3717 = __riscv_vwmacc_vx_i16m1(v3657, v3716, v3713, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3718 = __riscv_vsrl_vx_u8mf2(v3704, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3719 = __riscv_vand_vx_u8mf2(v3718, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3720 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3719);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3721 = v21 + 217;
      const int8_t* v3722 = (const int8_t*) v3721;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3723 = *(const int8_t *)(v3722);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3724 = __riscv_vwmacc_vx_i16m1(v3664, v3723, v3720, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3725 = __riscv_vsrl_vx_u8mf2(v3704, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3726 = __riscv_vand_vx_u8mf2(v3725, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3727 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3726);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3728 = v21 + 249;
      const int8_t* v3729 = (const int8_t*) v3728;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3730 = *(const int8_t *)(v3729);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3731 = __riscv_vwmacc_vx_i16m1(v3671, v3730, v3727, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3732 = v19 + 1176;
      const uint8_t* v3733 = (const uint8_t*) v3732;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3734 = __riscv_vle8_v_u8mf2(v3733, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3735 = __riscv_vand_vx_u8mf2(v3734, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3736 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3735);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3737 = v21 + 153;
      const int8_t* v3738 = (const int8_t*) v3737;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3739 = *(const int8_t *)(v3738);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3740 = __riscv_vwmacc_vx_i16m1(v3680, v3739, v3736, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3741 = __riscv_vsrl_vx_u8mf2(v3734, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3742 = __riscv_vand_vx_u8mf2(v3741, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3743 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3742);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3744 = v21 + 185;
      const int8_t* v3745 = (const int8_t*) v3744;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3746 = *(const int8_t *)(v3745);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3747 = __riscv_vwmacc_vx_i16m1(v3687, v3746, v3743, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3748 = __riscv_vsrl_vx_u8mf2(v3734, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3749 = __riscv_vand_vx_u8mf2(v3748, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3750 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3749);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3751 = v21 + 217;
      const int8_t* v3752 = (const int8_t*) v3751;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3753 = *(const int8_t *)(v3752);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3754 = __riscv_vwmacc_vx_i16m1(v3694, v3753, v3750, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3755 = __riscv_vsrl_vx_u8mf2(v3734, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3756 = __riscv_vand_vx_u8mf2(v3755, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3757 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3756);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3758 = v21 + 249;
      const int8_t* v3759 = (const int8_t*) v3758;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3760 = *(const int8_t *)(v3759);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3761 = __riscv_vwmacc_vx_i16m1(v3701, v3760, v3757, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3762 = v19 + 1184;
      const uint8_t* v3763 = (const uint8_t*) v3762;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3764 = __riscv_vle8_v_u8mf2(v3763, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3765 = __riscv_vand_vx_u8mf2(v3764, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3766 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3765);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3767 = v21 + 154;
      const int8_t* v3768 = (const int8_t*) v3767;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3769 = *(const int8_t *)(v3768);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3770 = __riscv_vwmacc_vx_i16m1(v3710, v3769, v3766, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3771 = __riscv_vsrl_vx_u8mf2(v3764, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3772 = __riscv_vand_vx_u8mf2(v3771, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3773 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3772);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3774 = v21 + 186;
      const int8_t* v3775 = (const int8_t*) v3774;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3776 = *(const int8_t *)(v3775);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3777 = __riscv_vwmacc_vx_i16m1(v3717, v3776, v3773, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3778 = __riscv_vsrl_vx_u8mf2(v3764, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3779 = __riscv_vand_vx_u8mf2(v3778, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3780 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3779);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3781 = v21 + 218;
      const int8_t* v3782 = (const int8_t*) v3781;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3783 = *(const int8_t *)(v3782);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3784 = __riscv_vwmacc_vx_i16m1(v3724, v3783, v3780, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3785 = __riscv_vsrl_vx_u8mf2(v3764, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3786 = __riscv_vand_vx_u8mf2(v3785, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3787 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3786);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3788 = v21 + 250;
      const int8_t* v3789 = (const int8_t*) v3788;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3790 = *(const int8_t *)(v3789);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3791 = __riscv_vwmacc_vx_i16m1(v3731, v3790, v3787, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3792 = v19 + 1192;
      const uint8_t* v3793 = (const uint8_t*) v3792;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3794 = __riscv_vle8_v_u8mf2(v3793, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3795 = __riscv_vand_vx_u8mf2(v3794, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3796 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3795);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3797 = v21 + 154;
      const int8_t* v3798 = (const int8_t*) v3797;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3799 = *(const int8_t *)(v3798);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3800 = __riscv_vwmacc_vx_i16m1(v3740, v3799, v3796, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3801 = __riscv_vsrl_vx_u8mf2(v3794, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3802 = __riscv_vand_vx_u8mf2(v3801, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3803 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3802);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3804 = v21 + 186;
      const int8_t* v3805 = (const int8_t*) v3804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3806 = *(const int8_t *)(v3805);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3807 = __riscv_vwmacc_vx_i16m1(v3747, v3806, v3803, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3808 = __riscv_vsrl_vx_u8mf2(v3794, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3809 = __riscv_vand_vx_u8mf2(v3808, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3810 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3809);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3811 = v21 + 218;
      const int8_t* v3812 = (const int8_t*) v3811;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3813 = *(const int8_t *)(v3812);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3814 = __riscv_vwmacc_vx_i16m1(v3754, v3813, v3810, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3815 = __riscv_vsrl_vx_u8mf2(v3794, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3816 = __riscv_vand_vx_u8mf2(v3815, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3817 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3816);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3818 = v21 + 250;
      const int8_t* v3819 = (const int8_t*) v3818;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3820 = *(const int8_t *)(v3819);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3821 = __riscv_vwmacc_vx_i16m1(v3761, v3820, v3817, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3822 = v19 + 1200;
      const uint8_t* v3823 = (const uint8_t*) v3822;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3824 = __riscv_vle8_v_u8mf2(v3823, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3825 = __riscv_vand_vx_u8mf2(v3824, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3826 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3825);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3827 = v21 + 155;
      const int8_t* v3828 = (const int8_t*) v3827;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3829 = *(const int8_t *)(v3828);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3830 = __riscv_vwmacc_vx_i16m1(v3770, v3829, v3826, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3831 = __riscv_vsrl_vx_u8mf2(v3824, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3832 = __riscv_vand_vx_u8mf2(v3831, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3833 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3832);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3834 = v21 + 187;
      const int8_t* v3835 = (const int8_t*) v3834;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3836 = *(const int8_t *)(v3835);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3837 = __riscv_vwmacc_vx_i16m1(v3777, v3836, v3833, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3838 = __riscv_vsrl_vx_u8mf2(v3824, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3839 = __riscv_vand_vx_u8mf2(v3838, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3840 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3839);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3841 = v21 + 219;
      const int8_t* v3842 = (const int8_t*) v3841;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3843 = *(const int8_t *)(v3842);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3844 = __riscv_vwmacc_vx_i16m1(v3784, v3843, v3840, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3845 = __riscv_vsrl_vx_u8mf2(v3824, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3846 = __riscv_vand_vx_u8mf2(v3845, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3847 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3846);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3848 = v21 + 251;
      const int8_t* v3849 = (const int8_t*) v3848;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3850 = *(const int8_t *)(v3849);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3851 = __riscv_vwmacc_vx_i16m1(v3791, v3850, v3847, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3852 = v19 + 1208;
      const uint8_t* v3853 = (const uint8_t*) v3852;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3854 = __riscv_vle8_v_u8mf2(v3853, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3855 = __riscv_vand_vx_u8mf2(v3854, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3856 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3855);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3857 = v21 + 155;
      const int8_t* v3858 = (const int8_t*) v3857;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3859 = *(const int8_t *)(v3858);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3860 = __riscv_vwmacc_vx_i16m1(v3800, v3859, v3856, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3861 = __riscv_vsrl_vx_u8mf2(v3854, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3862 = __riscv_vand_vx_u8mf2(v3861, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3863 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3862);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3864 = v21 + 187;
      const int8_t* v3865 = (const int8_t*) v3864;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3866 = *(const int8_t *)(v3865);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3867 = __riscv_vwmacc_vx_i16m1(v3807, v3866, v3863, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3868 = __riscv_vsrl_vx_u8mf2(v3854, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3869 = __riscv_vand_vx_u8mf2(v3868, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3870 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3869);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3871 = v21 + 219;
      const int8_t* v3872 = (const int8_t*) v3871;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3873 = *(const int8_t *)(v3872);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3874 = __riscv_vwmacc_vx_i16m1(v3814, v3873, v3870, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3875 = __riscv_vsrl_vx_u8mf2(v3854, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3876 = __riscv_vand_vx_u8mf2(v3875, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3877 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3876);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3878 = v21 + 251;
      const int8_t* v3879 = (const int8_t*) v3878;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3880 = *(const int8_t *)(v3879);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3881 = __riscv_vwmacc_vx_i16m1(v3821, v3880, v3877, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3882 = v19 + 1216;
      const uint8_t* v3883 = (const uint8_t*) v3882;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3884 = __riscv_vle8_v_u8mf2(v3883, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3885 = __riscv_vand_vx_u8mf2(v3884, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3886 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3885);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3887 = v21 + 156;
      const int8_t* v3888 = (const int8_t*) v3887;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3889 = *(const int8_t *)(v3888);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3890 = __riscv_vwmacc_vx_i16m1(v3830, v3889, v3886, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3891 = __riscv_vsrl_vx_u8mf2(v3884, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3892 = __riscv_vand_vx_u8mf2(v3891, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3893 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3892);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3894 = v21 + 188;
      const int8_t* v3895 = (const int8_t*) v3894;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3896 = *(const int8_t *)(v3895);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3897 = __riscv_vwmacc_vx_i16m1(v3837, v3896, v3893, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3898 = __riscv_vsrl_vx_u8mf2(v3884, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3899 = __riscv_vand_vx_u8mf2(v3898, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3900 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3899);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3901 = v21 + 220;
      const int8_t* v3902 = (const int8_t*) v3901;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3903 = *(const int8_t *)(v3902);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3904 = __riscv_vwmacc_vx_i16m1(v3844, v3903, v3900, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3905 = __riscv_vsrl_vx_u8mf2(v3884, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3906 = __riscv_vand_vx_u8mf2(v3905, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3907 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3906);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3908 = v21 + 252;
      const int8_t* v3909 = (const int8_t*) v3908;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3910 = *(const int8_t *)(v3909);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3911 = __riscv_vwmacc_vx_i16m1(v3851, v3910, v3907, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3912 = v19 + 1224;
      const uint8_t* v3913 = (const uint8_t*) v3912;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3914 = __riscv_vle8_v_u8mf2(v3913, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3915 = __riscv_vand_vx_u8mf2(v3914, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3916 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3915);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3917 = v21 + 156;
      const int8_t* v3918 = (const int8_t*) v3917;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3919 = *(const int8_t *)(v3918);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3920 = __riscv_vwmacc_vx_i16m1(v3860, v3919, v3916, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3921 = __riscv_vsrl_vx_u8mf2(v3914, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3922 = __riscv_vand_vx_u8mf2(v3921, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3923 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3922);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3924 = v21 + 188;
      const int8_t* v3925 = (const int8_t*) v3924;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3926 = *(const int8_t *)(v3925);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3927 = __riscv_vwmacc_vx_i16m1(v3867, v3926, v3923, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3928 = __riscv_vsrl_vx_u8mf2(v3914, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3929 = __riscv_vand_vx_u8mf2(v3928, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3930 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3929);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3931 = v21 + 220;
      const int8_t* v3932 = (const int8_t*) v3931;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3933 = *(const int8_t *)(v3932);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3934 = __riscv_vwmacc_vx_i16m1(v3874, v3933, v3930, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3935 = __riscv_vsrl_vx_u8mf2(v3914, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3936 = __riscv_vand_vx_u8mf2(v3935, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3937 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3936);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3938 = v21 + 252;
      const int8_t* v3939 = (const int8_t*) v3938;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3940 = *(const int8_t *)(v3939);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3941 = __riscv_vwmacc_vx_i16m1(v3881, v3940, v3937, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3942 = v19 + 1232;
      const uint8_t* v3943 = (const uint8_t*) v3942;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3944 = __riscv_vle8_v_u8mf2(v3943, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3945 = __riscv_vand_vx_u8mf2(v3944, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3946 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3945);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3947 = v21 + 157;
      const int8_t* v3948 = (const int8_t*) v3947;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3949 = *(const int8_t *)(v3948);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3950 = __riscv_vwmacc_vx_i16m1(v3890, v3949, v3946, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3951 = __riscv_vsrl_vx_u8mf2(v3944, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3952 = __riscv_vand_vx_u8mf2(v3951, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3953 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3952);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3954 = v21 + 189;
      const int8_t* v3955 = (const int8_t*) v3954;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3956 = *(const int8_t *)(v3955);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3957 = __riscv_vwmacc_vx_i16m1(v3897, v3956, v3953, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3958 = __riscv_vsrl_vx_u8mf2(v3944, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3959 = __riscv_vand_vx_u8mf2(v3958, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3960 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3959);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3961 = v21 + 221;
      const int8_t* v3962 = (const int8_t*) v3961;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3963 = *(const int8_t *)(v3962);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3964 = __riscv_vwmacc_vx_i16m1(v3904, v3963, v3960, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3965 = __riscv_vsrl_vx_u8mf2(v3944, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3966 = __riscv_vand_vx_u8mf2(v3965, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3967 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3966);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3968 = v21 + 253;
      const int8_t* v3969 = (const int8_t*) v3968;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3970 = *(const int8_t *)(v3969);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3971 = __riscv_vwmacc_vx_i16m1(v3911, v3970, v3967, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v3972 = v19 + 1240;
      const uint8_t* v3973 = (const uint8_t*) v3972;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3974 = __riscv_vle8_v_u8mf2(v3973, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3975 = __riscv_vand_vx_u8mf2(v3974, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3976 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3975);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3977 = v21 + 157;
      const int8_t* v3978 = (const int8_t*) v3977;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3979 = *(const int8_t *)(v3978);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3980 = __riscv_vwmacc_vx_i16m1(v3920, v3979, v3976, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3981 = __riscv_vsrl_vx_u8mf2(v3974, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3982 = __riscv_vand_vx_u8mf2(v3981, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3983 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3982);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3984 = v21 + 189;
      const int8_t* v3985 = (const int8_t*) v3984;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3986 = *(const int8_t *)(v3985);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3987 = __riscv_vwmacc_vx_i16m1(v3927, v3986, v3983, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3988 = __riscv_vsrl_vx_u8mf2(v3974, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3989 = __riscv_vand_vx_u8mf2(v3988, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3990 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3989);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3991 = v21 + 221;
      const int8_t* v3992 = (const int8_t*) v3991;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3993 = *(const int8_t *)(v3992);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3994 = __riscv_vwmacc_vx_i16m1(v3934, v3993, v3990, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3995 = __riscv_vsrl_vx_u8mf2(v3974, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3996 = __riscv_vand_vx_u8mf2(v3995, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3997 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3996);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3998 = v21 + 253;
      const int8_t* v3999 = (const int8_t*) v3998;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4000 = *(const int8_t *)(v3999);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4001 = __riscv_vwmacc_vx_i16m1(v3941, v4000, v3997, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v4002 = v19 + 1248;
      const uint8_t* v4003 = (const uint8_t*) v4002;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4004 = __riscv_vle8_v_u8mf2(v4003, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4005 = __riscv_vand_vx_u8mf2(v4004, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4006 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4005);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4007 = v21 + 158;
      const int8_t* v4008 = (const int8_t*) v4007;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4009 = *(const int8_t *)(v4008);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4010 = __riscv_vwmacc_vx_i16m1(v3950, v4009, v4006, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4011 = __riscv_vsrl_vx_u8mf2(v4004, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4012 = __riscv_vand_vx_u8mf2(v4011, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4013 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4012);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4014 = v21 + 190;
      const int8_t* v4015 = (const int8_t*) v4014;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4016 = *(const int8_t *)(v4015);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4017 = __riscv_vwmacc_vx_i16m1(v3957, v4016, v4013, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4018 = __riscv_vsrl_vx_u8mf2(v4004, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4019 = __riscv_vand_vx_u8mf2(v4018, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4020 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4019);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4021 = v21 + 222;
      const int8_t* v4022 = (const int8_t*) v4021;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4023 = *(const int8_t *)(v4022);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4024 = __riscv_vwmacc_vx_i16m1(v3964, v4023, v4020, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4025 = __riscv_vsrl_vx_u8mf2(v4004, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4026 = __riscv_vand_vx_u8mf2(v4025, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4027 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4026);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4028 = v21 + 254;
      const int8_t* v4029 = (const int8_t*) v4028;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4030 = *(const int8_t *)(v4029);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4031 = __riscv_vwmacc_vx_i16m1(v3971, v4030, v4027, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v4032 = v19 + 1256;
      const uint8_t* v4033 = (const uint8_t*) v4032;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4034 = __riscv_vle8_v_u8mf2(v4033, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4035 = __riscv_vand_vx_u8mf2(v4034, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4036 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4035);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4037 = v21 + 158;
      const int8_t* v4038 = (const int8_t*) v4037;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4039 = *(const int8_t *)(v4038);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4040 = __riscv_vwmacc_vx_i16m1(v3980, v4039, v4036, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4041 = __riscv_vsrl_vx_u8mf2(v4034, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4042 = __riscv_vand_vx_u8mf2(v4041, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4043 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4042);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4044 = v21 + 190;
      const int8_t* v4045 = (const int8_t*) v4044;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4046 = *(const int8_t *)(v4045);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4047 = __riscv_vwmacc_vx_i16m1(v3987, v4046, v4043, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4048 = __riscv_vsrl_vx_u8mf2(v4034, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4049 = __riscv_vand_vx_u8mf2(v4048, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4050 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4049);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4051 = v21 + 222;
      const int8_t* v4052 = (const int8_t*) v4051;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4053 = *(const int8_t *)(v4052);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4054 = __riscv_vwmacc_vx_i16m1(v3994, v4053, v4050, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4055 = __riscv_vsrl_vx_u8mf2(v4034, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4056 = __riscv_vand_vx_u8mf2(v4055, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4057 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4056);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4058 = v21 + 254;
      const int8_t* v4059 = (const int8_t*) v4058;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4060 = *(const int8_t *)(v4059);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4061 = __riscv_vwmacc_vx_i16m1(v4001, v4060, v4057, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v4062 = v19 + 1264;
      const uint8_t* v4063 = (const uint8_t*) v4062;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4064 = __riscv_vle8_v_u8mf2(v4063, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4065 = __riscv_vand_vx_u8mf2(v4064, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4066 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4065);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4067 = v21 + 159;
      const int8_t* v4068 = (const int8_t*) v4067;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4069 = *(const int8_t *)(v4068);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4070 = __riscv_vwmacc_vx_i16m1(v4010, v4069, v4066, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4071 = __riscv_vsrl_vx_u8mf2(v4064, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4072 = __riscv_vand_vx_u8mf2(v4071, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4073 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4072);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4074 = v21 + 191;
      const int8_t* v4075 = (const int8_t*) v4074;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4076 = *(const int8_t *)(v4075);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4077 = __riscv_vwmacc_vx_i16m1(v4017, v4076, v4073, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4078 = __riscv_vsrl_vx_u8mf2(v4064, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4079 = __riscv_vand_vx_u8mf2(v4078, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4080 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4079);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4081 = v21 + 223;
      const int8_t* v4082 = (const int8_t*) v4081;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4083 = *(const int8_t *)(v4082);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4084 = __riscv_vwmacc_vx_i16m1(v4024, v4083, v4080, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4085 = __riscv_vsrl_vx_u8mf2(v4064, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4086 = __riscv_vand_vx_u8mf2(v4085, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4087 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4086);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4088 = v21 + 255;
      const int8_t* v4089 = (const int8_t*) v4088;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4090 = *(const int8_t *)(v4089);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4091 = __riscv_vwmacc_vx_i16m1(v4031, v4090, v4087, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v4092 = v19 + 1272;
      const uint8_t* v4093 = (const uint8_t*) v4092;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4094 = __riscv_vle8_v_u8mf2(v4093, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4095 = __riscv_vand_vx_u8mf2(v4094, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4096 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4095);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4097 = v21 + 159;
      const int8_t* v4098 = (const int8_t*) v4097;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4099 = *(const int8_t *)(v4098);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4100 = __riscv_vwmacc_vx_i16m1(v4040, v4099, v4096, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4101 = __riscv_vsrl_vx_u8mf2(v4094, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4102 = __riscv_vand_vx_u8mf2(v4101, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4103 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4102);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4104 = v21 + 191;
      const int8_t* v4105 = (const int8_t*) v4104;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4106 = *(const int8_t *)(v4105);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4107 = __riscv_vwmacc_vx_i16m1(v4047, v4106, v4103, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4108 = __riscv_vsrl_vx_u8mf2(v4094, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4109 = __riscv_vand_vx_u8mf2(v4108, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4110 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4109);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4111 = v21 + 223;
      const int8_t* v4112 = (const int8_t*) v4111;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4113 = *(const int8_t *)(v4112);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4114 = __riscv_vwmacc_vx_i16m1(v4054, v4113, v4110, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4115 = __riscv_vsrl_vx_u8mf2(v4094, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4116 = __riscv_vand_vx_u8mf2(v4115, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4117 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4116);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4118 = v21 + 255;
      const int8_t* v4119 = (const int8_t*) v4118;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4120 = *(const int8_t *)(v4119);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4121 = __riscv_vwmacc_vx_i16m1(v4061, v4120, v4117, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v4122 = v19 + 1280;
      const uint8_t* v4123 = (const uint8_t*) v4122;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4124 = __riscv_vle8_v_u8mf2(v4123, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4125 = __riscv_vand_vx_u8mf2(v4124, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4126 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4125);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4127 = v21 + 160;
      const int8_t* v4128 = (const int8_t*) v4127;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4129 = *(const int8_t *)(v4128);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4130 = __riscv_vwmacc_vx_i16m1(v4070, v4129, v4126, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4131 = __riscv_vsrl_vx_u8mf2(v4124, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4132 = __riscv_vand_vx_u8mf2(v4131, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4133 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4132);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4134 = v21 + 192;
      const int8_t* v4135 = (const int8_t*) v4134;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4136 = *(const int8_t *)(v4135);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4137 = __riscv_vwmacc_vx_i16m1(v4077, v4136, v4133, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4138 = __riscv_vsrl_vx_u8mf2(v4124, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4139 = __riscv_vand_vx_u8mf2(v4138, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4140 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4139);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4141 = v21 + 224;
      const int8_t* v4142 = (const int8_t*) v4141;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4143 = *(const int8_t *)(v4142);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4144 = __riscv_vwmacc_vx_i16m1(v4084, v4143, v4140, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4145 = __riscv_vsrl_vx_u8mf2(v4124, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4146 = __riscv_vand_vx_u8mf2(v4145, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4147 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4146);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4148 = v21 + 256;
      const int8_t* v4149 = (const int8_t*) v4148;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4150 = *(const int8_t *)(v4149);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4151 = __riscv_vwmacc_vx_i16m1(v4091, v4150, v4147, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v4152 = v19 + 1288;
      const uint8_t* v4153 = (const uint8_t*) v4152;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4154 = __riscv_vle8_v_u8mf2(v4153, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4155 = __riscv_vand_vx_u8mf2(v4154, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4156 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4155);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4157 = v21 + 160;
      const int8_t* v4158 = (const int8_t*) v4157;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4159 = *(const int8_t *)(v4158);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4160 = __riscv_vwmacc_vx_i16m1(v4100, v4159, v4156, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4161 = __riscv_vsrl_vx_u8mf2(v4154, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4162 = __riscv_vand_vx_u8mf2(v4161, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4163 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4162);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4164 = v21 + 192;
      const int8_t* v4165 = (const int8_t*) v4164;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4166 = *(const int8_t *)(v4165);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4167 = __riscv_vwmacc_vx_i16m1(v4107, v4166, v4163, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4168 = __riscv_vsrl_vx_u8mf2(v4154, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4169 = __riscv_vand_vx_u8mf2(v4168, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4170 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4169);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4171 = v21 + 224;
      const int8_t* v4172 = (const int8_t*) v4171;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4173 = *(const int8_t *)(v4172);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4174 = __riscv_vwmacc_vx_i16m1(v4114, v4173, v4170, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4175 = __riscv_vsrl_vx_u8mf2(v4154, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4176 = __riscv_vand_vx_u8mf2(v4175, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4177 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4176);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4178 = v21 + 256;
      const int8_t* v4179 = (const int8_t*) v4178;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4180 = *(const int8_t *)(v4179);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4181 = __riscv_vwmacc_vx_i16m1(v4121, v4180, v4177, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v4182 = v19 + 1296;
      const uint8_t* v4183 = (const uint8_t*) v4182;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4184 = __riscv_vle8_v_u8mf2(v4183, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4185 = __riscv_vand_vx_u8mf2(v4184, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4186 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4185);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4187 = v21 + 161;
      const int8_t* v4188 = (const int8_t*) v4187;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4189 = *(const int8_t *)(v4188);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4190 = __riscv_vwmacc_vx_i16m1(v4130, v4189, v4186, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4191 = __riscv_vsrl_vx_u8mf2(v4184, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4192 = __riscv_vand_vx_u8mf2(v4191, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4193 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4192);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4194 = v21 + 193;
      const int8_t* v4195 = (const int8_t*) v4194;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4196 = *(const int8_t *)(v4195);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4197 = __riscv_vwmacc_vx_i16m1(v4137, v4196, v4193, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4198 = __riscv_vsrl_vx_u8mf2(v4184, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4199 = __riscv_vand_vx_u8mf2(v4198, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4200 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4199);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4201 = v21 + 225;
      const int8_t* v4202 = (const int8_t*) v4201;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4203 = *(const int8_t *)(v4202);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4204 = __riscv_vwmacc_vx_i16m1(v4144, v4203, v4200, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4205 = __riscv_vsrl_vx_u8mf2(v4184, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4206 = __riscv_vand_vx_u8mf2(v4205, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4207 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4206);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4208 = v21 + 257;
      const int8_t* v4209 = (const int8_t*) v4208;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4210 = *(const int8_t *)(v4209);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4211 = __riscv_vwmacc_vx_i16m1(v4151, v4210, v4207, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v4212 = v19 + 1304;
      const uint8_t* v4213 = (const uint8_t*) v4212;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4214 = __riscv_vle8_v_u8mf2(v4213, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4215 = __riscv_vand_vx_u8mf2(v4214, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4216 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4215);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4217 = v21 + 161;
      const int8_t* v4218 = (const int8_t*) v4217;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4219 = *(const int8_t *)(v4218);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4220 = __riscv_vwmacc_vx_i16m1(v4160, v4219, v4216, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4221 = __riscv_vsrl_vx_u8mf2(v4214, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4222 = __riscv_vand_vx_u8mf2(v4221, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4223 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4222);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4224 = v21 + 193;
      const int8_t* v4225 = (const int8_t*) v4224;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4226 = *(const int8_t *)(v4225);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4227 = __riscv_vwmacc_vx_i16m1(v4167, v4226, v4223, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4228 = __riscv_vsrl_vx_u8mf2(v4214, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4229 = __riscv_vand_vx_u8mf2(v4228, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4230 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4229);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4231 = v21 + 225;
      const int8_t* v4232 = (const int8_t*) v4231;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4233 = *(const int8_t *)(v4232);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4234 = __riscv_vwmacc_vx_i16m1(v4174, v4233, v4230, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4235 = __riscv_vsrl_vx_u8mf2(v4214, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4236 = __riscv_vand_vx_u8mf2(v4235, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4237 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4236);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4238 = v21 + 257;
      const int8_t* v4239 = (const int8_t*) v4238;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4240 = *(const int8_t *)(v4239);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4241 = __riscv_vwmacc_vx_i16m1(v4181, v4240, v4237, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v4242 = v19 + 1312;
      const uint8_t* v4243 = (const uint8_t*) v4242;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4244 = __riscv_vle8_v_u8mf2(v4243, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4245 = __riscv_vand_vx_u8mf2(v4244, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4246 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4245);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4247 = v21 + 162;
      const int8_t* v4248 = (const int8_t*) v4247;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4249 = *(const int8_t *)(v4248);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4250 = __riscv_vwmacc_vx_i16m1(v4190, v4249, v4246, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4251 = __riscv_vsrl_vx_u8mf2(v4244, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4252 = __riscv_vand_vx_u8mf2(v4251, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4253 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4252);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4254 = v21 + 194;
      const int8_t* v4255 = (const int8_t*) v4254;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4256 = *(const int8_t *)(v4255);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4257 = __riscv_vwmacc_vx_i16m1(v4197, v4256, v4253, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4258 = __riscv_vsrl_vx_u8mf2(v4244, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4259 = __riscv_vand_vx_u8mf2(v4258, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4260 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4259);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4261 = v21 + 226;
      const int8_t* v4262 = (const int8_t*) v4261;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4263 = *(const int8_t *)(v4262);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4264 = __riscv_vwmacc_vx_i16m1(v4204, v4263, v4260, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4265 = __riscv_vsrl_vx_u8mf2(v4244, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4266 = __riscv_vand_vx_u8mf2(v4265, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4267 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4266);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4268 = v21 + 258;
      const int8_t* v4269 = (const int8_t*) v4268;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4270 = *(const int8_t *)(v4269);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4271 = __riscv_vwmacc_vx_i16m1(v4211, v4270, v4267, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v4272 = v19 + 1320;
      const uint8_t* v4273 = (const uint8_t*) v4272;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4274 = __riscv_vle8_v_u8mf2(v4273, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4275 = __riscv_vand_vx_u8mf2(v4274, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4276 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4275);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4277 = v21 + 162;
      const int8_t* v4278 = (const int8_t*) v4277;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4279 = *(const int8_t *)(v4278);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4280 = __riscv_vwmacc_vx_i16m1(v4220, v4279, v4276, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4281 = __riscv_vsrl_vx_u8mf2(v4274, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4282 = __riscv_vand_vx_u8mf2(v4281, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4283 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4282);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4284 = v21 + 194;
      const int8_t* v4285 = (const int8_t*) v4284;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4286 = *(const int8_t *)(v4285);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4287 = __riscv_vwmacc_vx_i16m1(v4227, v4286, v4283, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4288 = __riscv_vsrl_vx_u8mf2(v4274, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4289 = __riscv_vand_vx_u8mf2(v4288, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4290 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4289);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4291 = v21 + 226;
      const int8_t* v4292 = (const int8_t*) v4291;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4293 = *(const int8_t *)(v4292);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4294 = __riscv_vwmacc_vx_i16m1(v4234, v4293, v4290, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4295 = __riscv_vsrl_vx_u8mf2(v4274, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4296 = __riscv_vand_vx_u8mf2(v4295, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4297 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4296);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4298 = v21 + 258;
      const int8_t* v4299 = (const int8_t*) v4298;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4300 = *(const int8_t *)(v4299);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4301 = __riscv_vwmacc_vx_i16m1(v4241, v4300, v4297, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v4302 = v19 + 1328;
      const uint8_t* v4303 = (const uint8_t*) v4302;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4304 = __riscv_vle8_v_u8mf2(v4303, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4305 = __riscv_vand_vx_u8mf2(v4304, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4306 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4305);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4307 = v21 + 163;
      const int8_t* v4308 = (const int8_t*) v4307;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4309 = *(const int8_t *)(v4308);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4310 = __riscv_vwmacc_vx_i16m1(v4250, v4309, v4306, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4311 = __riscv_vsrl_vx_u8mf2(v4304, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4312 = __riscv_vand_vx_u8mf2(v4311, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4313 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4312);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4314 = v21 + 195;
      const int8_t* v4315 = (const int8_t*) v4314;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4316 = *(const int8_t *)(v4315);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4317 = __riscv_vwmacc_vx_i16m1(v4257, v4316, v4313, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4318 = __riscv_vsrl_vx_u8mf2(v4304, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4319 = __riscv_vand_vx_u8mf2(v4318, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4320 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4319);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4321 = v21 + 227;
      const int8_t* v4322 = (const int8_t*) v4321;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4323 = *(const int8_t *)(v4322);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4324 = __riscv_vwmacc_vx_i16m1(v4264, v4323, v4320, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4325 = __riscv_vsrl_vx_u8mf2(v4304, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4326 = __riscv_vand_vx_u8mf2(v4325, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4327 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4326);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4328 = v21 + 259;
      const int8_t* v4329 = (const int8_t*) v4328;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4330 = *(const int8_t *)(v4329);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4331 = __riscv_vwmacc_vx_i16m1(v4271, v4330, v4327, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
      const uint8_t* v4332 = v19 + 1336;
      const uint8_t* v4333 = (const uint8_t*) v4332;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4334 = __riscv_vle8_v_u8mf2(v4333, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4335 = __riscv_vand_vx_u8mf2(v4334, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4336 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4335);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4337 = v21 + 163;
      const int8_t* v4338 = (const int8_t*) v4337;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4339 = *(const int8_t *)(v4338);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4340 = __riscv_vwmacc_vx_i16m1(v4280, v4339, v4336, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4341 = __riscv_vsrl_vx_u8mf2(v4334, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4342 = __riscv_vand_vx_u8mf2(v4341, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4343 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4342);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4344 = v21 + 195;
      const int8_t* v4345 = (const int8_t*) v4344;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4346 = *(const int8_t *)(v4345);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4347 = __riscv_vwmacc_vx_i16m1(v4287, v4346, v4343, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4348 = __riscv_vsrl_vx_u8mf2(v4334, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4349 = __riscv_vand_vx_u8mf2(v4348, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4350 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4349);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4351 = v21 + 227;
      const int8_t* v4352 = (const int8_t*) v4351;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4353 = *(const int8_t *)(v4352);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4354 = __riscv_vwmacc_vx_i16m1(v4294, v4353, v4350, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4355 = __riscv_vsrl_vx_u8mf2(v4334, 6, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4356 = __riscv_vand_vx_u8mf2(v4355, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4357 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4356);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4358 = v21 + 259;
      const int8_t* v4359 = (const int8_t*) v4358;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4360 = *(const int8_t *)(v4359);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4361 = __riscv_vwmacc_vx_i16m1(v4301, v4360, v4357, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v4362 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v4363 = __riscv_vwmacc_vv_i32m2(v4362, v2225, v4310, 8);
      v34 = v4363;
      vint32m2_t v4364 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v4365 = __riscv_vwmacc_vv_i32m2(v4364, v2243, v4317, 8);
      v34 = v4365;
      vint32m2_t v4366 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v4367 = __riscv_vwmacc_vv_i32m2(v4366, v2261, v4324, 8);
      v34 = v4367;
      vint32m2_t v4368 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v4369 = __riscv_vwmacc_vv_i32m2(v4368, v2279, v4331, 8);
      v34 = v4369;
      vint32m2_t v4370 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v4371 = __riscv_vwmacc_vv_i32m2(v4370, v2297, v4340, 8);
      v38 = v4371;
      vint32m2_t v4372 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v4373 = __riscv_vwmacc_vv_i32m2(v4372, v2315, v4347, 8);
      v38 = v4373;
      vint32m2_t v4374 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v4375 = __riscv_vwmacc_vv_i32m2(v4374, v2333, v4354, 8);
      v38 = v4375;
      vint32m2_t v4376 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v4377 = __riscv_vwmacc_vv_i32m2(v4376, v2351, v4361, 8);
      v38 = v4377;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v4378 = (const _Float16*) v19;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v4379 = __riscv_vle16_v_f16m1(v4378, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v4380 = __riscv_vfwcvt_f_f_v_f32m2(v4379, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v4381 = __riscv_vfmul_vf_f32m2(v4380, v23, 8);
      vint32m2_t v4382 = v34;
      vfloat32m2_t v4383 = v13;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v4384 = __riscv_vfcvt_f_x_v_f32m2(v4382, 8);
      vfloat32m2_t v4385 = __riscv_vfmacc_vv_f32m2(v4383, v4384, v4381, 8);
      vint32m2_t v4386 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v4387 = __riscv_vfcvt_f_x_v_f32m2(v4386, 8);
      vfloat32m2_t v4388 = __riscv_vfnmsac_vv_f32m2(v4385, v28, v4387, 8);
      v13 = v4388;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v4389 = v19 + 16;
      const _Float16* v4390 = (const _Float16*) v4389;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v4391 = __riscv_vle16_v_f16m1(v4390, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v4392 = __riscv_vfwcvt_f_f_v_f32m2(v4391, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v4393 = __riscv_vfmul_vf_f32m2(v4392, v23, 8);
      vint32m2_t v4394 = v38;
      vfloat32m2_t v4395 = v15;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v4396 = __riscv_vfcvt_f_x_v_f32m2(v4394, 8);
      vfloat32m2_t v4397 = __riscv_vfmacc_vv_f32m2(v4395, v4396, v4393, 8);
      vint32m2_t v4398 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v4399 = __riscv_vfcvt_f_x_v_f32m2(v4398, 8);
      vfloat32m2_t v4400 = __riscv_vfnmsac_vv_f32m2(v4397, v33, v4399, 8);
      v15 = v4400;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v4401 = v9 * 16;
    float* v4402 = v2 + v4401;
    vfloat32m2_t v4403 = v13;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v4402, v4403, 8);
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v4404 = v9 * 16;
    size_t v4405 = v4404 + 8;
    float* v4406 = v2 + v4405;
    vfloat32m2_t v4407 = v15;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v4406, v4407, 8);
  }
  return;
}


