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
      vint16m1_t v242;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v243 = __riscv_vmv_v_x_i16m1(0, 8);
      v242 = v243;
      vint16m1_t v244;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v245 = __riscv_vmv_v_x_i16m1(0, 8);
      v244 = v245;
      vint16m1_t v246;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v247 = __riscv_vmv_v_x_i16m1(0, 8);
      v246 = v247;
      vint16m1_t v248;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v249 = __riscv_vmv_v_x_i16m1(0, 8);
      v248 = v249;
      vint16m1_t v250;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v251 = __riscv_vmv_v_x_i16m1(0, 8);
      v250 = v251;
      vint16m1_t v252;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v253 = __riscv_vmv_v_x_i16m1(0, 8);
      v252 = v253;
      vint16m1_t v254;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v255 = __riscv_vmv_v_x_i16m1(0, 8);
      v254 = v255;
      vint16m1_t v256;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v257 = __riscv_vmv_v_x_i16m1(0, 8);
      v256 = v257;
      for (size_t v258 = 0; v258 < 16; v258 += 1) {
        size_t v259 = v258 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
        size_t v260 = 320 + v259;
        const uint8_t* v261 = v19 + v260;
        const uint8_t* v262 = (const uint8_t*) v261;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v263 = __riscv_vle8_v_u8mf2(v262, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v264 = __riscv_vand_vx_u8mf2(v263, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v265 = __riscv_vreinterpret_v_u8mf2_i8mf2(v264);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v266 = 4 + v258;
        const uint8_t* v267 = v21 + v266;
        const int8_t* v268 = (const int8_t*) v267;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v269 = *(const int8_t *)(v268);
        vint16m1_t v270 = v242;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v271 = __riscv_vwmacc_vx_i16m1(v270, v269, v265, 8);
        v242 = v271;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v272 = __riscv_vsrl_vx_u8mf2(v263, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v273 = __riscv_vand_vx_u8mf2(v272, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v274 = __riscv_vreinterpret_v_u8mf2_i8mf2(v273);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v275 = 36 + v258;
        const uint8_t* v276 = v21 + v275;
        const int8_t* v277 = (const int8_t*) v276;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v278 = *(const int8_t *)(v277);
        vint16m1_t v279 = v244;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v280 = __riscv_vwmacc_vx_i16m1(v279, v278, v274, 8);
        v244 = v280;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v281 = __riscv_vsrl_vx_u8mf2(v263, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v282 = __riscv_vand_vx_u8mf2(v281, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v283 = __riscv_vreinterpret_v_u8mf2_i8mf2(v282);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v284 = 68 + v258;
        const uint8_t* v285 = v21 + v284;
        const int8_t* v286 = (const int8_t*) v285;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v287 = *(const int8_t *)(v286);
        vint16m1_t v288 = v246;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v289 = __riscv_vwmacc_vx_i16m1(v288, v287, v283, 8);
        v246 = v289;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v290 = __riscv_vsrl_vx_u8mf2(v263, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v291 = __riscv_vand_vx_u8mf2(v290, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v292 = __riscv_vreinterpret_v_u8mf2_i8mf2(v291);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v293 = 100 + v258;
        const uint8_t* v294 = v21 + v293;
        const int8_t* v295 = (const int8_t*) v294;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v296 = *(const int8_t *)(v295);
        vint16m1_t v297 = v248;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v298 = __riscv_vwmacc_vx_i16m1(v297, v296, v292, 8);
        v248 = v298;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
        size_t v299 = 328 + v259;
        const uint8_t* v300 = v19 + v299;
        const uint8_t* v301 = (const uint8_t*) v300;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v302 = __riscv_vle8_v_u8mf2(v301, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v303 = __riscv_vand_vx_u8mf2(v302, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v304 = __riscv_vreinterpret_v_u8mf2_i8mf2(v303);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v305 = 4 + v258;
        const uint8_t* v306 = v21 + v305;
        const int8_t* v307 = (const int8_t*) v306;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v308 = *(const int8_t *)(v307);
        vint16m1_t v309 = v250;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v310 = __riscv_vwmacc_vx_i16m1(v309, v308, v304, 8);
        v250 = v310;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v311 = __riscv_vsrl_vx_u8mf2(v302, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v312 = __riscv_vand_vx_u8mf2(v311, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v313 = __riscv_vreinterpret_v_u8mf2_i8mf2(v312);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v314 = 36 + v258;
        const uint8_t* v315 = v21 + v314;
        const int8_t* v316 = (const int8_t*) v315;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v317 = *(const int8_t *)(v316);
        vint16m1_t v318 = v252;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v319 = __riscv_vwmacc_vx_i16m1(v318, v317, v313, 8);
        v252 = v319;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v320 = __riscv_vsrl_vx_u8mf2(v302, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v321 = __riscv_vand_vx_u8mf2(v320, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v322 = __riscv_vreinterpret_v_u8mf2_i8mf2(v321);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v323 = 68 + v258;
        const uint8_t* v324 = v21 + v323;
        const int8_t* v325 = (const int8_t*) v324;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v326 = *(const int8_t *)(v325);
        vint16m1_t v327 = v254;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v328 = __riscv_vwmacc_vx_i16m1(v327, v326, v322, 8);
        v254 = v328;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v329 = __riscv_vsrl_vx_u8mf2(v302, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v330 = __riscv_vand_vx_u8mf2(v329, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v331 = __riscv_vreinterpret_v_u8mf2_i8mf2(v330);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v332 = 100 + v258;
        const uint8_t* v333 = v21 + v332;
        const int8_t* v334 = (const int8_t*) v333;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v335 = *(const int8_t *)(v334);
        vint16m1_t v336 = v256;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v337 = __riscv_vwmacc_vx_i16m1(v336, v335, v331, 8);
        v256 = v337;
      }
      vint16m1_t v338 = v242;
      vint16m1_t v339 = v244;
      vint16m1_t v340 = v246;
      vint16m1_t v341 = v248;
      vint16m1_t v342 = v250;
      vint16m1_t v343 = v252;
      vint16m1_t v344 = v254;
      vint16m1_t v345 = v256;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v346 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v347 = __riscv_vwmacc_vv_i32m2(v346, v48, v338, 8);
      v34 = v347;
      vint32m2_t v348 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v349 = __riscv_vwmacc_vv_i32m2(v348, v66, v339, 8);
      v34 = v349;
      vint32m2_t v350 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v351 = __riscv_vwmacc_vv_i32m2(v350, v84, v340, 8);
      v34 = v351;
      vint32m2_t v352 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v353 = __riscv_vwmacc_vv_i32m2(v352, v102, v341, 8);
      v34 = v353;
      vint32m2_t v354 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v355 = __riscv_vwmacc_vv_i32m2(v354, v120, v342, 8);
      v38 = v355;
      vint32m2_t v356 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v357 = __riscv_vwmacc_vv_i32m2(v356, v138, v343, 8);
      v38 = v357;
      vint32m2_t v358 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v359 = __riscv_vwmacc_vv_i32m2(v358, v156, v344, 8);
      v38 = v359;
      vint32m2_t v360 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v361 = __riscv_vwmacc_vv_i32m2(v360, v174, v345, 8);
      v38 = v361;
      vint16m1_t v362;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v363 = __riscv_vmv_v_x_i16m1(0, 8);
      v362 = v363;
      vint16m1_t v364;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v365 = __riscv_vmv_v_x_i16m1(0, 8);
      v364 = v365;
      vint16m1_t v366;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v367 = __riscv_vmv_v_x_i16m1(0, 8);
      v366 = v367;
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
      for (size_t v378 = 0; v378 < 16; v378 += 1) {
        size_t v379 = v378 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
        size_t v380 = 576 + v379;
        const uint8_t* v381 = v19 + v380;
        const uint8_t* v382 = (const uint8_t*) v381;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v383 = __riscv_vle8_v_u8mf2(v382, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v384 = __riscv_vand_vx_u8mf2(v383, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v385 = __riscv_vreinterpret_v_u8mf2_i8mf2(v384);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v386 = 20 + v378;
        const uint8_t* v387 = v21 + v386;
        const int8_t* v388 = (const int8_t*) v387;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v389 = *(const int8_t *)(v388);
        vint16m1_t v390 = v362;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v391 = __riscv_vwmacc_vx_i16m1(v390, v389, v385, 8);
        v362 = v391;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v392 = __riscv_vsrl_vx_u8mf2(v383, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v393 = __riscv_vand_vx_u8mf2(v392, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v394 = __riscv_vreinterpret_v_u8mf2_i8mf2(v393);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v395 = 52 + v378;
        const uint8_t* v396 = v21 + v395;
        const int8_t* v397 = (const int8_t*) v396;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v398 = *(const int8_t *)(v397);
        vint16m1_t v399 = v364;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v400 = __riscv_vwmacc_vx_i16m1(v399, v398, v394, 8);
        v364 = v400;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v401 = __riscv_vsrl_vx_u8mf2(v383, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v402 = __riscv_vand_vx_u8mf2(v401, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v403 = __riscv_vreinterpret_v_u8mf2_i8mf2(v402);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v404 = 84 + v378;
        const uint8_t* v405 = v21 + v404;
        const int8_t* v406 = (const int8_t*) v405;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v407 = *(const int8_t *)(v406);
        vint16m1_t v408 = v366;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v409 = __riscv_vwmacc_vx_i16m1(v408, v407, v403, 8);
        v366 = v409;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v410 = __riscv_vsrl_vx_u8mf2(v383, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v411 = __riscv_vand_vx_u8mf2(v410, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v412 = __riscv_vreinterpret_v_u8mf2_i8mf2(v411);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v413 = 116 + v378;
        const uint8_t* v414 = v21 + v413;
        const int8_t* v415 = (const int8_t*) v414;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v416 = *(const int8_t *)(v415);
        vint16m1_t v417 = v368;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v418 = __riscv_vwmacc_vx_i16m1(v417, v416, v412, 8);
        v368 = v418;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
        size_t v419 = 584 + v379;
        const uint8_t* v420 = v19 + v419;
        const uint8_t* v421 = (const uint8_t*) v420;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v422 = __riscv_vle8_v_u8mf2(v421, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v423 = __riscv_vand_vx_u8mf2(v422, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v424 = __riscv_vreinterpret_v_u8mf2_i8mf2(v423);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v425 = 20 + v378;
        const uint8_t* v426 = v21 + v425;
        const int8_t* v427 = (const int8_t*) v426;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v428 = *(const int8_t *)(v427);
        vint16m1_t v429 = v370;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v430 = __riscv_vwmacc_vx_i16m1(v429, v428, v424, 8);
        v370 = v430;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v431 = __riscv_vsrl_vx_u8mf2(v422, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v432 = __riscv_vand_vx_u8mf2(v431, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v433 = __riscv_vreinterpret_v_u8mf2_i8mf2(v432);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v434 = 52 + v378;
        const uint8_t* v435 = v21 + v434;
        const int8_t* v436 = (const int8_t*) v435;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v437 = *(const int8_t *)(v436);
        vint16m1_t v438 = v372;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v439 = __riscv_vwmacc_vx_i16m1(v438, v437, v433, 8);
        v372 = v439;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v440 = __riscv_vsrl_vx_u8mf2(v422, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v441 = __riscv_vand_vx_u8mf2(v440, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v442 = __riscv_vreinterpret_v_u8mf2_i8mf2(v441);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v443 = 84 + v378;
        const uint8_t* v444 = v21 + v443;
        const int8_t* v445 = (const int8_t*) v444;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v446 = *(const int8_t *)(v445);
        vint16m1_t v447 = v374;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v448 = __riscv_vwmacc_vx_i16m1(v447, v446, v442, 8);
        v374 = v448;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v449 = __riscv_vsrl_vx_u8mf2(v422, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v450 = __riscv_vand_vx_u8mf2(v449, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v451 = __riscv_vreinterpret_v_u8mf2_i8mf2(v450);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v452 = 116 + v378;
        const uint8_t* v453 = v21 + v452;
        const int8_t* v454 = (const int8_t*) v453;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v455 = *(const int8_t *)(v454);
        vint16m1_t v456 = v376;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v457 = __riscv_vwmacc_vx_i16m1(v456, v455, v451, 8);
        v376 = v457;
      }
      vint16m1_t v458 = v362;
      vint16m1_t v459 = v364;
      vint16m1_t v460 = v366;
      vint16m1_t v461 = v368;
      vint16m1_t v462 = v370;
      vint16m1_t v463 = v372;
      vint16m1_t v464 = v374;
      vint16m1_t v465 = v376;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v466 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v467 = __riscv_vwmacc_vv_i32m2(v466, v57, v458, 8);
      v34 = v467;
      vint32m2_t v468 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v469 = __riscv_vwmacc_vv_i32m2(v468, v75, v459, 8);
      v34 = v469;
      vint32m2_t v470 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v471 = __riscv_vwmacc_vv_i32m2(v470, v93, v460, 8);
      v34 = v471;
      vint32m2_t v472 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v473 = __riscv_vwmacc_vv_i32m2(v472, v111, v461, 8);
      v34 = v473;
      vint32m2_t v474 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v475 = __riscv_vwmacc_vv_i32m2(v474, v129, v462, 8);
      v38 = v475;
      vint32m2_t v476 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v477 = __riscv_vwmacc_vv_i32m2(v476, v147, v463, 8);
      v38 = v477;
      vint32m2_t v478 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v479 = __riscv_vwmacc_vv_i32m2(v478, v165, v464, 8);
      v38 = v479;
      vint32m2_t v480 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v481 = __riscv_vwmacc_vv_i32m2(v480, v183, v465, 8);
      v38 = v481;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
      const uint8_t* v482 = v19 + 192;
      const uint8_t* v483 = (const uint8_t*) v482;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v484 = __riscv_vle8_v_u8mf2(v483, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v485 = __riscv_vand_vx_u8mf2(v484, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v486 = __riscv_vsrl_vx_u8mf2(v484, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v487 = __riscv_vzext_vf2_u16m1(v485, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v488 = __riscv_vreinterpret_v_u16m1_i16m1(v487);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v489 = __riscv_vzext_vf2_u16m1(v486, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v490 = __riscv_vreinterpret_v_u16m1_i16m1(v489);
      const uint8_t* v491 = v19 + 208;
      const uint8_t* v492 = (const uint8_t*) v491;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v493 = __riscv_vle8_v_u8mf2(v492, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v494 = __riscv_vand_vx_u8mf2(v493, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v495 = __riscv_vsrl_vx_u8mf2(v493, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v496 = __riscv_vzext_vf2_u16m1(v494, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v497 = __riscv_vreinterpret_v_u16m1_i16m1(v496);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v498 = __riscv_vzext_vf2_u16m1(v495, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v499 = __riscv_vreinterpret_v_u16m1_i16m1(v498);
      const uint8_t* v500 = v19 + 224;
      const uint8_t* v501 = (const uint8_t*) v500;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v502 = __riscv_vle8_v_u8mf2(v501, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v503 = __riscv_vand_vx_u8mf2(v502, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v504 = __riscv_vsrl_vx_u8mf2(v502, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v505 = __riscv_vzext_vf2_u16m1(v503, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v506 = __riscv_vreinterpret_v_u16m1_i16m1(v505);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v507 = __riscv_vzext_vf2_u16m1(v504, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v508 = __riscv_vreinterpret_v_u16m1_i16m1(v507);
      const uint8_t* v509 = v19 + 240;
      const uint8_t* v510 = (const uint8_t*) v509;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v511 = __riscv_vle8_v_u8mf2(v510, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v512 = __riscv_vand_vx_u8mf2(v511, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v513 = __riscv_vsrl_vx_u8mf2(v511, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v514 = __riscv_vzext_vf2_u16m1(v512, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v515 = __riscv_vreinterpret_v_u16m1_i16m1(v514);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v516 = __riscv_vzext_vf2_u16m1(v513, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v517 = __riscv_vreinterpret_v_u16m1_i16m1(v516);
      const uint8_t* v518 = v19 + 256;
      const uint8_t* v519 = (const uint8_t*) v518;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v520 = __riscv_vle8_v_u8mf2(v519, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v521 = __riscv_vand_vx_u8mf2(v520, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v522 = __riscv_vsrl_vx_u8mf2(v520, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v523 = __riscv_vzext_vf2_u16m1(v521, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v524 = __riscv_vreinterpret_v_u16m1_i16m1(v523);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v525 = __riscv_vzext_vf2_u16m1(v522, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v526 = __riscv_vreinterpret_v_u16m1_i16m1(v525);
      const uint8_t* v527 = v19 + 272;
      const uint8_t* v528 = (const uint8_t*) v527;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v529 = __riscv_vle8_v_u8mf2(v528, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v530 = __riscv_vand_vx_u8mf2(v529, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v531 = __riscv_vsrl_vx_u8mf2(v529, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v532 = __riscv_vzext_vf2_u16m1(v530, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v533 = __riscv_vreinterpret_v_u16m1_i16m1(v532);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v534 = __riscv_vzext_vf2_u16m1(v531, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v535 = __riscv_vreinterpret_v_u16m1_i16m1(v534);
      const uint8_t* v536 = v19 + 288;
      const uint8_t* v537 = (const uint8_t*) v536;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v538 = __riscv_vle8_v_u8mf2(v537, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v539 = __riscv_vand_vx_u8mf2(v538, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v540 = __riscv_vsrl_vx_u8mf2(v538, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v541 = __riscv_vzext_vf2_u16m1(v539, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v542 = __riscv_vreinterpret_v_u16m1_i16m1(v541);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v543 = __riscv_vzext_vf2_u16m1(v540, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v544 = __riscv_vreinterpret_v_u16m1_i16m1(v543);
      const uint8_t* v545 = v19 + 304;
      const uint8_t* v546 = (const uint8_t*) v545;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v547 = __riscv_vle8_v_u8mf2(v546, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v548 = __riscv_vand_vx_u8mf2(v547, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v549 = __riscv_vsrl_vx_u8mf2(v547, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v550 = __riscv_vzext_vf2_u16m1(v548, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v551 = __riscv_vreinterpret_v_u16m1_i16m1(v550);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v552 = __riscv_vzext_vf2_u16m1(v549, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v553 = __riscv_vreinterpret_v_u16m1_i16m1(v552);
      const uint8_t* v554 = v19 + 200;
      const uint8_t* v555 = (const uint8_t*) v554;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v556 = __riscv_vle8_v_u8mf2(v555, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v557 = __riscv_vand_vx_u8mf2(v556, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v558 = __riscv_vsrl_vx_u8mf2(v556, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v559 = __riscv_vzext_vf2_u16m1(v557, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v560 = __riscv_vreinterpret_v_u16m1_i16m1(v559);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v561 = __riscv_vzext_vf2_u16m1(v558, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v562 = __riscv_vreinterpret_v_u16m1_i16m1(v561);
      const uint8_t* v563 = v19 + 216;
      const uint8_t* v564 = (const uint8_t*) v563;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v565 = __riscv_vle8_v_u8mf2(v564, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v566 = __riscv_vand_vx_u8mf2(v565, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v567 = __riscv_vsrl_vx_u8mf2(v565, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v568 = __riscv_vzext_vf2_u16m1(v566, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v569 = __riscv_vreinterpret_v_u16m1_i16m1(v568);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v570 = __riscv_vzext_vf2_u16m1(v567, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v571 = __riscv_vreinterpret_v_u16m1_i16m1(v570);
      const uint8_t* v572 = v19 + 232;
      const uint8_t* v573 = (const uint8_t*) v572;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v574 = __riscv_vle8_v_u8mf2(v573, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v575 = __riscv_vand_vx_u8mf2(v574, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v576 = __riscv_vsrl_vx_u8mf2(v574, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v577 = __riscv_vzext_vf2_u16m1(v575, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v578 = __riscv_vreinterpret_v_u16m1_i16m1(v577);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v579 = __riscv_vzext_vf2_u16m1(v576, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v580 = __riscv_vreinterpret_v_u16m1_i16m1(v579);
      const uint8_t* v581 = v19 + 248;
      const uint8_t* v582 = (const uint8_t*) v581;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v583 = __riscv_vle8_v_u8mf2(v582, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v584 = __riscv_vand_vx_u8mf2(v583, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v585 = __riscv_vsrl_vx_u8mf2(v583, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v586 = __riscv_vzext_vf2_u16m1(v584, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v587 = __riscv_vreinterpret_v_u16m1_i16m1(v586);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v588 = __riscv_vzext_vf2_u16m1(v585, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v589 = __riscv_vreinterpret_v_u16m1_i16m1(v588);
      const uint8_t* v590 = v19 + 264;
      const uint8_t* v591 = (const uint8_t*) v590;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v592 = __riscv_vle8_v_u8mf2(v591, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v593 = __riscv_vand_vx_u8mf2(v592, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v594 = __riscv_vsrl_vx_u8mf2(v592, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v595 = __riscv_vzext_vf2_u16m1(v593, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v596 = __riscv_vreinterpret_v_u16m1_i16m1(v595);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v597 = __riscv_vzext_vf2_u16m1(v594, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v598 = __riscv_vreinterpret_v_u16m1_i16m1(v597);
      const uint8_t* v599 = v19 + 280;
      const uint8_t* v600 = (const uint8_t*) v599;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v601 = __riscv_vle8_v_u8mf2(v600, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v602 = __riscv_vand_vx_u8mf2(v601, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v603 = __riscv_vsrl_vx_u8mf2(v601, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v604 = __riscv_vzext_vf2_u16m1(v602, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v605 = __riscv_vreinterpret_v_u16m1_i16m1(v604);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v606 = __riscv_vzext_vf2_u16m1(v603, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v607 = __riscv_vreinterpret_v_u16m1_i16m1(v606);
      const uint8_t* v608 = v19 + 296;
      const uint8_t* v609 = (const uint8_t*) v608;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v610 = __riscv_vle8_v_u8mf2(v609, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v611 = __riscv_vand_vx_u8mf2(v610, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v612 = __riscv_vsrl_vx_u8mf2(v610, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v613 = __riscv_vzext_vf2_u16m1(v611, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v614 = __riscv_vreinterpret_v_u16m1_i16m1(v613);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v615 = __riscv_vzext_vf2_u16m1(v612, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v616 = __riscv_vreinterpret_v_u16m1_i16m1(v615);
      const uint8_t* v617 = v19 + 312;
      const uint8_t* v618 = (const uint8_t*) v617;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v619 = __riscv_vle8_v_u8mf2(v618, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v620 = __riscv_vand_vx_u8mf2(v619, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v621 = __riscv_vsrl_vx_u8mf2(v619, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v622 = __riscv_vzext_vf2_u16m1(v620, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v623 = __riscv_vreinterpret_v_u16m1_i16m1(v622);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v624 = __riscv_vzext_vf2_u16m1(v621, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v625 = __riscv_vreinterpret_v_u16m1_i16m1(v624);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold
      const uint8_t* v626 = v21 + 276;
      const int16_t* v627 = (const int16_t*) v626;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v628 = *(const int16_t *)(v627);
      vint32m2_t v629 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v630 = __riscv_vwmacc_vx_i32m2(v629, v628, v490, 8);
      v36 = v630;
      vint32m2_t v631 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v632 = __riscv_vwmacc_vx_i32m2(v631, v628, v562, 8);
      v40 = v632;
      const uint8_t* v633 = v21 + 278;
      const int16_t* v634 = (const int16_t*) v633;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v635 = *(const int16_t *)(v634);
      vint32m2_t v636 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v637 = __riscv_vwmacc_vx_i32m2(v636, v635, v499, 8);
      v36 = v637;
      vint32m2_t v638 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v639 = __riscv_vwmacc_vx_i32m2(v638, v635, v571, 8);
      v40 = v639;
      const uint8_t* v640 = v21 + 280;
      const int16_t* v641 = (const int16_t*) v640;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v642 = *(const int16_t *)(v641);
      vint32m2_t v643 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v644 = __riscv_vwmacc_vx_i32m2(v643, v642, v508, 8);
      v36 = v644;
      vint32m2_t v645 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v646 = __riscv_vwmacc_vx_i32m2(v645, v642, v580, 8);
      v40 = v646;
      const uint8_t* v647 = v21 + 282;
      const int16_t* v648 = (const int16_t*) v647;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v649 = *(const int16_t *)(v648);
      vint32m2_t v650 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v651 = __riscv_vwmacc_vx_i32m2(v650, v649, v517, 8);
      v36 = v651;
      vint32m2_t v652 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v653 = __riscv_vwmacc_vx_i32m2(v652, v649, v589, 8);
      v40 = v653;
      const uint8_t* v654 = v21 + 284;
      const int16_t* v655 = (const int16_t*) v654;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v656 = *(const int16_t *)(v655);
      vint32m2_t v657 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v658 = __riscv_vwmacc_vx_i32m2(v657, v656, v526, 8);
      v36 = v658;
      vint32m2_t v659 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v660 = __riscv_vwmacc_vx_i32m2(v659, v656, v598, 8);
      v40 = v660;
      const uint8_t* v661 = v21 + 286;
      const int16_t* v662 = (const int16_t*) v661;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v663 = *(const int16_t *)(v662);
      vint32m2_t v664 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v665 = __riscv_vwmacc_vx_i32m2(v664, v663, v535, 8);
      v36 = v665;
      vint32m2_t v666 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v667 = __riscv_vwmacc_vx_i32m2(v666, v663, v607, 8);
      v40 = v667;
      const uint8_t* v668 = v21 + 288;
      const int16_t* v669 = (const int16_t*) v668;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v670 = *(const int16_t *)(v669);
      vint32m2_t v671 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v672 = __riscv_vwmacc_vx_i32m2(v671, v670, v544, 8);
      v36 = v672;
      vint32m2_t v673 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v674 = __riscv_vwmacc_vx_i32m2(v673, v670, v616, 8);
      v40 = v674;
      const uint8_t* v675 = v21 + 290;
      const int16_t* v676 = (const int16_t*) v675;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v677 = *(const int16_t *)(v676);
      vint32m2_t v678 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v679 = __riscv_vwmacc_vx_i32m2(v678, v677, v553, 8);
      v36 = v679;
      vint32m2_t v680 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v681 = __riscv_vwmacc_vx_i32m2(v680, v677, v625, 8);
      v40 = v681;
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
      vint16m1_t v692;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v693 = __riscv_vmv_v_x_i16m1(0, 8);
      v692 = v693;
      vint16m1_t v694;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v695 = __riscv_vmv_v_x_i16m1(0, 8);
      v694 = v695;
      vint16m1_t v696;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v697 = __riscv_vmv_v_x_i16m1(0, 8);
      v696 = v697;
      for (size_t v698 = 0; v698 < 16; v698 += 1) {
        size_t v699 = v698 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
        size_t v700 = 832 + v699;
        const uint8_t* v701 = v19 + v700;
        const uint8_t* v702 = (const uint8_t*) v701;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v703 = __riscv_vle8_v_u8mf2(v702, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v704 = __riscv_vand_vx_u8mf2(v703, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v705 = __riscv_vreinterpret_v_u8mf2_i8mf2(v704);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v706 = 132 + v698;
        const uint8_t* v707 = v21 + v706;
        const int8_t* v708 = (const int8_t*) v707;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v709 = *(const int8_t *)(v708);
        vint16m1_t v710 = v682;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v711 = __riscv_vwmacc_vx_i16m1(v710, v709, v705, 8);
        v682 = v711;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v712 = __riscv_vsrl_vx_u8mf2(v703, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v713 = __riscv_vand_vx_u8mf2(v712, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v714 = __riscv_vreinterpret_v_u8mf2_i8mf2(v713);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v715 = 164 + v698;
        const uint8_t* v716 = v21 + v715;
        const int8_t* v717 = (const int8_t*) v716;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v718 = *(const int8_t *)(v717);
        vint16m1_t v719 = v684;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v720 = __riscv_vwmacc_vx_i16m1(v719, v718, v714, 8);
        v684 = v720;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v721 = __riscv_vsrl_vx_u8mf2(v703, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v722 = __riscv_vand_vx_u8mf2(v721, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v723 = __riscv_vreinterpret_v_u8mf2_i8mf2(v722);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v724 = 196 + v698;
        const uint8_t* v725 = v21 + v724;
        const int8_t* v726 = (const int8_t*) v725;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v727 = *(const int8_t *)(v726);
        vint16m1_t v728 = v686;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v729 = __riscv_vwmacc_vx_i16m1(v728, v727, v723, 8);
        v686 = v729;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v730 = __riscv_vsrl_vx_u8mf2(v703, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v731 = __riscv_vand_vx_u8mf2(v730, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v732 = __riscv_vreinterpret_v_u8mf2_i8mf2(v731);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v733 = 228 + v698;
        const uint8_t* v734 = v21 + v733;
        const int8_t* v735 = (const int8_t*) v734;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v736 = *(const int8_t *)(v735);
        vint16m1_t v737 = v688;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v738 = __riscv_vwmacc_vx_i16m1(v737, v736, v732, 8);
        v688 = v738;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
        size_t v739 = 840 + v699;
        const uint8_t* v740 = v19 + v739;
        const uint8_t* v741 = (const uint8_t*) v740;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v742 = __riscv_vle8_v_u8mf2(v741, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v743 = __riscv_vand_vx_u8mf2(v742, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v744 = __riscv_vreinterpret_v_u8mf2_i8mf2(v743);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v745 = 132 + v698;
        const uint8_t* v746 = v21 + v745;
        const int8_t* v747 = (const int8_t*) v746;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v748 = *(const int8_t *)(v747);
        vint16m1_t v749 = v690;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v750 = __riscv_vwmacc_vx_i16m1(v749, v748, v744, 8);
        v690 = v750;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v751 = __riscv_vsrl_vx_u8mf2(v742, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v752 = __riscv_vand_vx_u8mf2(v751, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v753 = __riscv_vreinterpret_v_u8mf2_i8mf2(v752);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v754 = 164 + v698;
        const uint8_t* v755 = v21 + v754;
        const int8_t* v756 = (const int8_t*) v755;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v757 = *(const int8_t *)(v756);
        vint16m1_t v758 = v692;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v759 = __riscv_vwmacc_vx_i16m1(v758, v757, v753, 8);
        v692 = v759;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v760 = __riscv_vsrl_vx_u8mf2(v742, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v761 = __riscv_vand_vx_u8mf2(v760, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v762 = __riscv_vreinterpret_v_u8mf2_i8mf2(v761);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v763 = 196 + v698;
        const uint8_t* v764 = v21 + v763;
        const int8_t* v765 = (const int8_t*) v764;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v766 = *(const int8_t *)(v765);
        vint16m1_t v767 = v694;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v768 = __riscv_vwmacc_vx_i16m1(v767, v766, v762, 8);
        v694 = v768;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v769 = __riscv_vsrl_vx_u8mf2(v742, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v770 = __riscv_vand_vx_u8mf2(v769, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v771 = __riscv_vreinterpret_v_u8mf2_i8mf2(v770);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v772 = 228 + v698;
        const uint8_t* v773 = v21 + v772;
        const int8_t* v774 = (const int8_t*) v773;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v775 = *(const int8_t *)(v774);
        vint16m1_t v776 = v696;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v777 = __riscv_vwmacc_vx_i16m1(v776, v775, v771, 8);
        v696 = v777;
      }
      vint16m1_t v778 = v682;
      vint16m1_t v779 = v684;
      vint16m1_t v780 = v686;
      vint16m1_t v781 = v688;
      vint16m1_t v782 = v690;
      vint16m1_t v783 = v692;
      vint16m1_t v784 = v694;
      vint16m1_t v785 = v696;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v786 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v787 = __riscv_vwmacc_vv_i32m2(v786, v488, v778, 8);
      v34 = v787;
      vint32m2_t v788 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v789 = __riscv_vwmacc_vv_i32m2(v788, v506, v779, 8);
      v34 = v789;
      vint32m2_t v790 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v791 = __riscv_vwmacc_vv_i32m2(v790, v524, v780, 8);
      v34 = v791;
      vint32m2_t v792 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v793 = __riscv_vwmacc_vv_i32m2(v792, v542, v781, 8);
      v34 = v793;
      vint32m2_t v794 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v795 = __riscv_vwmacc_vv_i32m2(v794, v560, v782, 8);
      v38 = v795;
      vint32m2_t v796 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v797 = __riscv_vwmacc_vv_i32m2(v796, v578, v783, 8);
      v38 = v797;
      vint32m2_t v798 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v799 = __riscv_vwmacc_vv_i32m2(v798, v596, v784, 8);
      v38 = v799;
      vint32m2_t v800 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v801 = __riscv_vwmacc_vv_i32m2(v800, v614, v785, 8);
      v38 = v801;
      vint16m1_t v802;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v803 = __riscv_vmv_v_x_i16m1(0, 8);
      v802 = v803;
      vint16m1_t v804;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v805 = __riscv_vmv_v_x_i16m1(0, 8);
      v804 = v805;
      vint16m1_t v806;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v807 = __riscv_vmv_v_x_i16m1(0, 8);
      v806 = v807;
      vint16m1_t v808;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v809 = __riscv_vmv_v_x_i16m1(0, 8);
      v808 = v809;
      vint16m1_t v810;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v811 = __riscv_vmv_v_x_i16m1(0, 8);
      v810 = v811;
      vint16m1_t v812;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v813 = __riscv_vmv_v_x_i16m1(0, 8);
      v812 = v813;
      vint16m1_t v814;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v815 = __riscv_vmv_v_x_i16m1(0, 8);
      v814 = v815;
      vint16m1_t v816;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v817 = __riscv_vmv_v_x_i16m1(0, 8);
      v816 = v817;
      for (size_t v818 = 0; v818 < 16; v818 += 1) {
        size_t v819 = v818 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
        size_t v820 = 1088 + v819;
        const uint8_t* v821 = v19 + v820;
        const uint8_t* v822 = (const uint8_t*) v821;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v823 = __riscv_vle8_v_u8mf2(v822, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v824 = __riscv_vand_vx_u8mf2(v823, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v825 = __riscv_vreinterpret_v_u8mf2_i8mf2(v824);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v826 = 148 + v818;
        const uint8_t* v827 = v21 + v826;
        const int8_t* v828 = (const int8_t*) v827;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v829 = *(const int8_t *)(v828);
        vint16m1_t v830 = v802;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v831 = __riscv_vwmacc_vx_i16m1(v830, v829, v825, 8);
        v802 = v831;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v832 = __riscv_vsrl_vx_u8mf2(v823, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v833 = __riscv_vand_vx_u8mf2(v832, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v834 = __riscv_vreinterpret_v_u8mf2_i8mf2(v833);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v835 = 180 + v818;
        const uint8_t* v836 = v21 + v835;
        const int8_t* v837 = (const int8_t*) v836;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v838 = *(const int8_t *)(v837);
        vint16m1_t v839 = v804;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v840 = __riscv_vwmacc_vx_i16m1(v839, v838, v834, 8);
        v804 = v840;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v841 = __riscv_vsrl_vx_u8mf2(v823, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v842 = __riscv_vand_vx_u8mf2(v841, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v843 = __riscv_vreinterpret_v_u8mf2_i8mf2(v842);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v844 = 212 + v818;
        const uint8_t* v845 = v21 + v844;
        const int8_t* v846 = (const int8_t*) v845;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v847 = *(const int8_t *)(v846);
        vint16m1_t v848 = v806;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v849 = __riscv_vwmacc_vx_i16m1(v848, v847, v843, 8);
        v806 = v849;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v850 = __riscv_vsrl_vx_u8mf2(v823, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v851 = __riscv_vand_vx_u8mf2(v850, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v852 = __riscv_vreinterpret_v_u8mf2_i8mf2(v851);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v853 = 244 + v818;
        const uint8_t* v854 = v21 + v853;
        const int8_t* v855 = (const int8_t*) v854;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v856 = *(const int8_t *)(v855);
        vint16m1_t v857 = v808;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v858 = __riscv_vwmacc_vx_i16m1(v857, v856, v852, 8);
        v808 = v858;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_2bit_addr
        size_t v859 = 1096 + v819;
        const uint8_t* v860 = v19 + v859;
        const uint8_t* v861 = (const uint8_t*) v860;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v862 = __riscv_vle8_v_u8mf2(v861, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v863 = __riscv_vand_vx_u8mf2(v862, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v864 = __riscv_vreinterpret_v_u8mf2_i8mf2(v863);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v865 = 148 + v818;
        const uint8_t* v866 = v21 + v865;
        const int8_t* v867 = (const int8_t*) v866;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v868 = *(const int8_t *)(v867);
        vint16m1_t v869 = v810;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v870 = __riscv_vwmacc_vx_i16m1(v869, v868, v864, 8);
        v810 = v870;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v871 = __riscv_vsrl_vx_u8mf2(v862, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v872 = __riscv_vand_vx_u8mf2(v871, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v873 = __riscv_vreinterpret_v_u8mf2_i8mf2(v872);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v874 = 180 + v818;
        const uint8_t* v875 = v21 + v874;
        const int8_t* v876 = (const int8_t*) v875;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v877 = *(const int8_t *)(v876);
        vint16m1_t v878 = v812;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v879 = __riscv_vwmacc_vx_i16m1(v878, v877, v873, 8);
        v812 = v879;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v880 = __riscv_vsrl_vx_u8mf2(v862, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v881 = __riscv_vand_vx_u8mf2(v880, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v882 = __riscv_vreinterpret_v_u8mf2_i8mf2(v881);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v883 = 212 + v818;
        const uint8_t* v884 = v21 + v883;
        const int8_t* v885 = (const int8_t*) v884;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v886 = *(const int8_t *)(v885);
        vint16m1_t v887 = v814;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v888 = __riscv_vwmacc_vx_i16m1(v887, v886, v882, 8);
        v814 = v888;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v889 = __riscv_vsrl_vx_u8mf2(v862, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v890 = __riscv_vand_vx_u8mf2(v889, 0x03, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v891 = __riscv_vreinterpret_v_u8mf2_i8mf2(v890);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v892 = 244 + v818;
        const uint8_t* v893 = v21 + v892;
        const int8_t* v894 = (const int8_t*) v893;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v895 = *(const int8_t *)(v894);
        vint16m1_t v896 = v816;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v897 = __riscv_vwmacc_vx_i16m1(v896, v895, v891, 8);
        v816 = v897;
      }
      vint16m1_t v898 = v802;
      vint16m1_t v899 = v804;
      vint16m1_t v900 = v806;
      vint16m1_t v901 = v808;
      vint16m1_t v902 = v810;
      vint16m1_t v903 = v812;
      vint16m1_t v904 = v814;
      vint16m1_t v905 = v816;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v906 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v907 = __riscv_vwmacc_vv_i32m2(v906, v497, v898, 8);
      v34 = v907;
      vint32m2_t v908 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v909 = __riscv_vwmacc_vv_i32m2(v908, v515, v899, 8);
      v34 = v909;
      vint32m2_t v910 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v911 = __riscv_vwmacc_vv_i32m2(v910, v533, v900, 8);
      v34 = v911;
      vint32m2_t v912 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v913 = __riscv_vwmacc_vv_i32m2(v912, v551, v901, 8);
      v34 = v913;
      vint32m2_t v914 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v915 = __riscv_vwmacc_vv_i32m2(v914, v569, v902, 8);
      v38 = v915;
      vint32m2_t v916 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v917 = __riscv_vwmacc_vv_i32m2(v916, v587, v903, 8);
      v38 = v917;
      vint32m2_t v918 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v919 = __riscv_vwmacc_vv_i32m2(v918, v605, v904, 8);
      v38 = v919;
      vint32m2_t v920 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v921 = __riscv_vwmacc_vv_i32m2(v920, v623, v905, 8);
      v38 = v921;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v922 = (const _Float16*) v19;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v923 = __riscv_vle16_v_f16m1(v922, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v924 = __riscv_vfwcvt_f_f_v_f32m2(v923, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v925 = __riscv_vfmul_vf_f32m2(v924, v23, 8);
      vint32m2_t v926 = v34;
      vfloat32m2_t v927 = v13;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v928 = __riscv_vfcvt_f_x_v_f32m2(v926, 8);
      vfloat32m2_t v929 = __riscv_vfmacc_vv_f32m2(v927, v928, v925, 8);
      vint32m2_t v930 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v931 = __riscv_vfcvt_f_x_v_f32m2(v930, 8);
      vfloat32m2_t v932 = __riscv_vfnmsac_vv_f32m2(v929, v28, v931, 8);
      v13 = v932;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v933 = v19 + 16;
      const _Float16* v934 = (const _Float16*) v933;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v935 = __riscv_vle16_v_f16m1(v934, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v936 = __riscv_vfwcvt_f_f_v_f32m2(v935, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v937 = __riscv_vfmul_vf_f32m2(v936, v23, 8);
      vint32m2_t v938 = v38;
      vfloat32m2_t v939 = v15;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v940 = __riscv_vfcvt_f_x_v_f32m2(v938, 8);
      vfloat32m2_t v941 = __riscv_vfmacc_vv_f32m2(v939, v940, v937, 8);
      vint32m2_t v942 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v943 = __riscv_vfcvt_f_x_v_f32m2(v942, 8);
      vfloat32m2_t v944 = __riscv_vfnmsac_vv_f32m2(v941, v33, v943, 8);
      v15 = v944;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v945 = v9 * 16;
    float* v946 = v2 + v945;
    vfloat32m2_t v947 = v13;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v946, v947, 8);
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v948 = v9 * 16;
    size_t v949 = v948 + 8;
    float* v950 = v2 + v949;
    vfloat32m2_t v951 = v15;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v950, v951, 8);
  }
  return;
}


