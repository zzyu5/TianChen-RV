#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemv_q5_K_q8_K_kernel_ggml_repack_gemv_q5_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
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
    size_t v11 = v10 * 2816;
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
      size_t v18 = v17 * 2816;
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
      const uint8_t* v45 = v19 + 192;
      const uint8_t* v46 = (const uint8_t*) v45;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v47 = __riscv_vle8_v_u8mf2(v46, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v48 = __riscv_vand_vx_u8mf2(v44, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v49 = __riscv_vsrl_vx_u8mf2(v44, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v50 = __riscv_vand_vx_u8mf2(v47, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v51 = __riscv_vsll_vx_u8mf2(v50, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v52 = __riscv_vand_vx_u8mf2(v47, 0x0C, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v53 = __riscv_vsll_vx_u8mf2(v52, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v54 = __riscv_vor_vv_u8mf2(v51, v48, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v55 = __riscv_vor_vv_u8mf2(v53, v49, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v56 = __riscv_vzext_vf2_u16m1(v54, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v57 = __riscv_vreinterpret_v_u16m1_i16m1(v56);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v58 = __riscv_vzext_vf2_u16m1(v55, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v59 = __riscv_vreinterpret_v_u16m1_i16m1(v58);
      const uint8_t* v60 = v19 + 80;
      const uint8_t* v61 = (const uint8_t*) v60;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v62 = __riscv_vle8_v_u8mf2(v61, 8);
      const uint8_t* v63 = v19 + 208;
      const uint8_t* v64 = (const uint8_t*) v63;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v65 = __riscv_vle8_v_u8mf2(v64, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v66 = __riscv_vand_vx_u8mf2(v62, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v67 = __riscv_vsrl_vx_u8mf2(v62, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v68 = __riscv_vand_vx_u8mf2(v65, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v69 = __riscv_vsll_vx_u8mf2(v68, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v70 = __riscv_vand_vx_u8mf2(v65, 0x0C, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v71 = __riscv_vsll_vx_u8mf2(v70, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v72 = __riscv_vor_vv_u8mf2(v69, v66, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v73 = __riscv_vor_vv_u8mf2(v71, v67, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v74 = __riscv_vzext_vf2_u16m1(v72, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v75 = __riscv_vreinterpret_v_u16m1_i16m1(v74);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v76 = __riscv_vzext_vf2_u16m1(v73, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v77 = __riscv_vreinterpret_v_u16m1_i16m1(v76);
      const uint8_t* v78 = v19 + 96;
      const uint8_t* v79 = (const uint8_t*) v78;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v80 = __riscv_vle8_v_u8mf2(v79, 8);
      const uint8_t* v81 = v19 + 224;
      const uint8_t* v82 = (const uint8_t*) v81;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v83 = __riscv_vle8_v_u8mf2(v82, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v84 = __riscv_vand_vx_u8mf2(v80, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v85 = __riscv_vsrl_vx_u8mf2(v80, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v86 = __riscv_vand_vx_u8mf2(v83, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v87 = __riscv_vsll_vx_u8mf2(v86, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v88 = __riscv_vand_vx_u8mf2(v83, 0x0C, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v89 = __riscv_vsll_vx_u8mf2(v88, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v90 = __riscv_vor_vv_u8mf2(v87, v84, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v91 = __riscv_vor_vv_u8mf2(v89, v85, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v92 = __riscv_vzext_vf2_u16m1(v90, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v93 = __riscv_vreinterpret_v_u16m1_i16m1(v92);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v94 = __riscv_vzext_vf2_u16m1(v91, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v95 = __riscv_vreinterpret_v_u16m1_i16m1(v94);
      const uint8_t* v96 = v19 + 112;
      const uint8_t* v97 = (const uint8_t*) v96;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v98 = __riscv_vle8_v_u8mf2(v97, 8);
      const uint8_t* v99 = v19 + 240;
      const uint8_t* v100 = (const uint8_t*) v99;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v101 = __riscv_vle8_v_u8mf2(v100, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v102 = __riscv_vand_vx_u8mf2(v98, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v103 = __riscv_vsrl_vx_u8mf2(v98, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v104 = __riscv_vand_vx_u8mf2(v101, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v105 = __riscv_vsll_vx_u8mf2(v104, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v106 = __riscv_vand_vx_u8mf2(v101, 0x0C, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v107 = __riscv_vsll_vx_u8mf2(v106, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v108 = __riscv_vor_vv_u8mf2(v105, v102, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v109 = __riscv_vor_vv_u8mf2(v107, v103, 8);
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
      const uint8_t* v117 = v19 + 200;
      const uint8_t* v118 = (const uint8_t*) v117;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v119 = __riscv_vle8_v_u8mf2(v118, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v120 = __riscv_vand_vx_u8mf2(v116, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v121 = __riscv_vsrl_vx_u8mf2(v116, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v122 = __riscv_vand_vx_u8mf2(v119, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v123 = __riscv_vsll_vx_u8mf2(v122, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v124 = __riscv_vand_vx_u8mf2(v119, 0x0C, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v125 = __riscv_vsll_vx_u8mf2(v124, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v126 = __riscv_vor_vv_u8mf2(v123, v120, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v127 = __riscv_vor_vv_u8mf2(v125, v121, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v128 = __riscv_vzext_vf2_u16m1(v126, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v129 = __riscv_vreinterpret_v_u16m1_i16m1(v128);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v130 = __riscv_vzext_vf2_u16m1(v127, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v131 = __riscv_vreinterpret_v_u16m1_i16m1(v130);
      const uint8_t* v132 = v19 + 88;
      const uint8_t* v133 = (const uint8_t*) v132;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v134 = __riscv_vle8_v_u8mf2(v133, 8);
      const uint8_t* v135 = v19 + 216;
      const uint8_t* v136 = (const uint8_t*) v135;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v137 = __riscv_vle8_v_u8mf2(v136, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v138 = __riscv_vand_vx_u8mf2(v134, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v139 = __riscv_vsrl_vx_u8mf2(v134, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v140 = __riscv_vand_vx_u8mf2(v137, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v141 = __riscv_vsll_vx_u8mf2(v140, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v142 = __riscv_vand_vx_u8mf2(v137, 0x0C, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v143 = __riscv_vsll_vx_u8mf2(v142, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v144 = __riscv_vor_vv_u8mf2(v141, v138, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v145 = __riscv_vor_vv_u8mf2(v143, v139, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v146 = __riscv_vzext_vf2_u16m1(v144, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v147 = __riscv_vreinterpret_v_u16m1_i16m1(v146);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v148 = __riscv_vzext_vf2_u16m1(v145, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v149 = __riscv_vreinterpret_v_u16m1_i16m1(v148);
      const uint8_t* v150 = v19 + 104;
      const uint8_t* v151 = (const uint8_t*) v150;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v152 = __riscv_vle8_v_u8mf2(v151, 8);
      const uint8_t* v153 = v19 + 232;
      const uint8_t* v154 = (const uint8_t*) v153;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v155 = __riscv_vle8_v_u8mf2(v154, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v156 = __riscv_vand_vx_u8mf2(v152, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v157 = __riscv_vsrl_vx_u8mf2(v152, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v158 = __riscv_vand_vx_u8mf2(v155, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v159 = __riscv_vsll_vx_u8mf2(v158, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v160 = __riscv_vand_vx_u8mf2(v155, 0x0C, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v161 = __riscv_vsll_vx_u8mf2(v160, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v162 = __riscv_vor_vv_u8mf2(v159, v156, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v163 = __riscv_vor_vv_u8mf2(v161, v157, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v164 = __riscv_vzext_vf2_u16m1(v162, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v165 = __riscv_vreinterpret_v_u16m1_i16m1(v164);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v166 = __riscv_vzext_vf2_u16m1(v163, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v167 = __riscv_vreinterpret_v_u16m1_i16m1(v166);
      const uint8_t* v168 = v19 + 120;
      const uint8_t* v169 = (const uint8_t*) v168;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v170 = __riscv_vle8_v_u8mf2(v169, 8);
      const uint8_t* v171 = v19 + 248;
      const uint8_t* v172 = (const uint8_t*) v171;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v173 = __riscv_vle8_v_u8mf2(v172, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v174 = __riscv_vand_vx_u8mf2(v170, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v175 = __riscv_vsrl_vx_u8mf2(v170, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v176 = __riscv_vand_vx_u8mf2(v173, 0x03, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v177 = __riscv_vsll_vx_u8mf2(v176, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v178 = __riscv_vand_vx_u8mf2(v173, 0x0C, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v179 = __riscv_vsll_vx_u8mf2(v178, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v180 = __riscv_vor_vv_u8mf2(v177, v174, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v181 = __riscv_vor_vv_u8mf2(v179, v175, 8);
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
      const uint8_t* v189 = v21 + 262;
      const int16_t* v190 = (const int16_t*) v189;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v191 = *(const int16_t *)(v190);
      int32_t v192 = v188 + v191;
      vint32m2_t v193 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v194 = __riscv_vwmacc_vx_i32m2(v193, v192, v59, 8);
      v36 = v194;
      vint32m2_t v195 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v196 = __riscv_vwmacc_vx_i32m2(v195, v192, v131, 8);
      v40 = v196;
      const uint8_t* v197 = v21 + 264;
      const int16_t* v198 = (const int16_t*) v197;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v199 = *(const int16_t *)(v198);
      const uint8_t* v200 = v21 + 266;
      const int16_t* v201 = (const int16_t*) v200;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v202 = *(const int16_t *)(v201);
      int32_t v203 = v199 + v202;
      vint32m2_t v204 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v205 = __riscv_vwmacc_vx_i32m2(v204, v203, v77, 8);
      v36 = v205;
      vint32m2_t v206 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v207 = __riscv_vwmacc_vx_i32m2(v206, v203, v149, 8);
      v40 = v207;
      const uint8_t* v208 = v21 + 268;
      const int16_t* v209 = (const int16_t*) v208;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v210 = *(const int16_t *)(v209);
      const uint8_t* v211 = v21 + 270;
      const int16_t* v212 = (const int16_t*) v211;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v213 = *(const int16_t *)(v212);
      int32_t v214 = v210 + v213;
      vint32m2_t v215 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v216 = __riscv_vwmacc_vx_i32m2(v215, v214, v95, 8);
      v36 = v216;
      vint32m2_t v217 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v218 = __riscv_vwmacc_vx_i32m2(v217, v214, v167, 8);
      v40 = v218;
      const uint8_t* v219 = v21 + 272;
      const int16_t* v220 = (const int16_t*) v219;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v221 = *(const int16_t *)(v220);
      const uint8_t* v222 = v21 + 274;
      const int16_t* v223 = (const int16_t*) v222;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v224 = *(const int16_t *)(v223);
      int32_t v225 = v221 + v224;
      vint32m2_t v226 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v227 = __riscv_vwmacc_vx_i32m2(v226, v225, v113, 8);
      v36 = v227;
      vint32m2_t v228 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v229 = __riscv_vwmacc_vx_i32m2(v228, v225, v185, 8);
      v40 = v229;
      vint16m1_t v230;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v231 = __riscv_vmv_v_x_i16m1(0, 8);
      v230 = v231;
      vint16m1_t v232;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v233 = __riscv_vmv_v_x_i16m1(0, 8);
      v232 = v233;
      vint16m1_t v234;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v235 = __riscv_vmv_v_x_i16m1(0, 8);
      v234 = v235;
      vint16m1_t v236;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v237 = __riscv_vmv_v_x_i16m1(0, 8);
      v236 = v237;
      for (size_t v238 = 0; v238 < 16; v238 += 1) {
        size_t v239 = v238 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v240 = 768 + v239;
        const uint8_t* v241 = v19 + v240;
        const uint8_t* v242 = (const uint8_t*) v241;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v243 = __riscv_vle8_v_u8mf2(v242, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v244 = __riscv_vand_vx_u8mf2(v243, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v245 = __riscv_vsrl_vx_u8mf2(v243, 4, 8);
        size_t v246 = 256 + v239;
        const uint8_t* v247 = v19 + v246;
        const uint8_t* v248 = (const uint8_t*) v247;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v249 = __riscv_vle8_v_u8mf2(v248, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v250 = __riscv_vand_vx_u8mf2(v249, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v251 = __riscv_vsll_vx_u8mf2(v250, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v252 = __riscv_vsrl_vx_u8mf2(v249, 1, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v253 = __riscv_vand_vx_u8mf2(v252, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v254 = __riscv_vsll_vx_u8mf2(v253, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v255 = __riscv_vor_vv_u8mf2(v244, v251, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v256 = __riscv_vreinterpret_v_u8mf2_i8mf2(v255);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v257 = __riscv_vor_vv_u8mf2(v245, v254, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v258 = __riscv_vreinterpret_v_u8mf2_i8mf2(v257);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v259 = 4 + v238;
        size_t v260 = 36 + v238;
        const uint8_t* v261 = v21 + v259;
        const int8_t* v262 = (const int8_t*) v261;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v263 = *(const int8_t *)(v262);
        const uint8_t* v264 = v21 + v260;
        const int8_t* v265 = (const int8_t*) v264;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v266 = *(const int8_t *)(v265);
        vint16m1_t v267 = v230;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v268 = __riscv_vwmacc_vx_i16m1(v267, v263, v256, 8);
        v230 = v268;
        vint16m1_t v269 = v232;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v270 = __riscv_vwmacc_vx_i16m1(v269, v266, v258, 8);
        v232 = v270;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v271 = 776 + v239;
        const uint8_t* v272 = v19 + v271;
        const uint8_t* v273 = (const uint8_t*) v272;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v274 = __riscv_vle8_v_u8mf2(v273, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v275 = __riscv_vand_vx_u8mf2(v274, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v276 = __riscv_vsrl_vx_u8mf2(v274, 4, 8);
        size_t v277 = 264 + v239;
        const uint8_t* v278 = v19 + v277;
        const uint8_t* v279 = (const uint8_t*) v278;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v280 = __riscv_vle8_v_u8mf2(v279, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v281 = __riscv_vand_vx_u8mf2(v280, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v282 = __riscv_vsll_vx_u8mf2(v281, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v283 = __riscv_vsrl_vx_u8mf2(v280, 1, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v284 = __riscv_vand_vx_u8mf2(v283, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v285 = __riscv_vsll_vx_u8mf2(v284, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v286 = __riscv_vor_vv_u8mf2(v275, v282, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v287 = __riscv_vreinterpret_v_u8mf2_i8mf2(v286);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v288 = __riscv_vor_vv_u8mf2(v276, v285, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v289 = __riscv_vreinterpret_v_u8mf2_i8mf2(v288);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v290 = 4 + v238;
        size_t v291 = 36 + v238;
        const uint8_t* v292 = v21 + v290;
        const int8_t* v293 = (const int8_t*) v292;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v294 = *(const int8_t *)(v293);
        const uint8_t* v295 = v21 + v291;
        const int8_t* v296 = (const int8_t*) v295;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v297 = *(const int8_t *)(v296);
        vint16m1_t v298 = v234;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v299 = __riscv_vwmacc_vx_i16m1(v298, v294, v287, 8);
        v234 = v299;
        vint16m1_t v300 = v236;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v301 = __riscv_vwmacc_vx_i16m1(v300, v297, v289, 8);
        v236 = v301;
      }
      vint16m1_t v302 = v230;
      vint16m1_t v303 = v232;
      vint16m1_t v304 = v234;
      vint16m1_t v305 = v236;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v306 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v307 = __riscv_vwmacc_vv_i32m2(v306, v57, v302, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v308 = __riscv_vwmacc_vv_i32m2(v307, v75, v303, 8);
      v34 = v308;
      vint32m2_t v309 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v310 = __riscv_vwmacc_vv_i32m2(v309, v129, v304, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v311 = __riscv_vwmacc_vv_i32m2(v310, v147, v305, 8);
      v38 = v311;
      vint16m1_t v312;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v313 = __riscv_vmv_v_x_i16m1(0, 8);
      v312 = v313;
      vint16m1_t v314;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v315 = __riscv_vmv_v_x_i16m1(0, 8);
      v314 = v315;
      vint16m1_t v316;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v317 = __riscv_vmv_v_x_i16m1(0, 8);
      v316 = v317;
      vint16m1_t v318;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v319 = __riscv_vmv_v_x_i16m1(0, 8);
      v318 = v319;
      for (size_t v320 = 0; v320 < 16; v320 += 1) {
        size_t v321 = v320 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v322 = 1024 + v321;
        const uint8_t* v323 = v19 + v322;
        const uint8_t* v324 = (const uint8_t*) v323;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v325 = __riscv_vle8_v_u8mf2(v324, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v326 = __riscv_vand_vx_u8mf2(v325, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v327 = __riscv_vsrl_vx_u8mf2(v325, 4, 8);
        size_t v328 = 512 + v321;
        const uint8_t* v329 = v19 + v328;
        const uint8_t* v330 = (const uint8_t*) v329;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v331 = __riscv_vle8_v_u8mf2(v330, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v332 = __riscv_vand_vx_u8mf2(v331, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v333 = __riscv_vsll_vx_u8mf2(v332, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v334 = __riscv_vsrl_vx_u8mf2(v331, 1, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v335 = __riscv_vand_vx_u8mf2(v334, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v336 = __riscv_vsll_vx_u8mf2(v335, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v337 = __riscv_vor_vv_u8mf2(v326, v333, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v338 = __riscv_vreinterpret_v_u8mf2_i8mf2(v337);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v339 = __riscv_vor_vv_u8mf2(v327, v336, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v340 = __riscv_vreinterpret_v_u8mf2_i8mf2(v339);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v341 = 20 + v320;
        size_t v342 = 52 + v320;
        const uint8_t* v343 = v21 + v341;
        const int8_t* v344 = (const int8_t*) v343;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v345 = *(const int8_t *)(v344);
        const uint8_t* v346 = v21 + v342;
        const int8_t* v347 = (const int8_t*) v346;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v348 = *(const int8_t *)(v347);
        vint16m1_t v349 = v312;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v350 = __riscv_vwmacc_vx_i16m1(v349, v345, v338, 8);
        v312 = v350;
        vint16m1_t v351 = v314;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v352 = __riscv_vwmacc_vx_i16m1(v351, v348, v340, 8);
        v314 = v352;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v353 = 1032 + v321;
        const uint8_t* v354 = v19 + v353;
        const uint8_t* v355 = (const uint8_t*) v354;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v356 = __riscv_vle8_v_u8mf2(v355, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v357 = __riscv_vand_vx_u8mf2(v356, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v358 = __riscv_vsrl_vx_u8mf2(v356, 4, 8);
        size_t v359 = 520 + v321;
        const uint8_t* v360 = v19 + v359;
        const uint8_t* v361 = (const uint8_t*) v360;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v362 = __riscv_vle8_v_u8mf2(v361, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v363 = __riscv_vand_vx_u8mf2(v362, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v364 = __riscv_vsll_vx_u8mf2(v363, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v365 = __riscv_vsrl_vx_u8mf2(v362, 1, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v366 = __riscv_vand_vx_u8mf2(v365, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v367 = __riscv_vsll_vx_u8mf2(v366, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v368 = __riscv_vor_vv_u8mf2(v357, v364, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v369 = __riscv_vreinterpret_v_u8mf2_i8mf2(v368);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v370 = __riscv_vor_vv_u8mf2(v358, v367, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v371 = __riscv_vreinterpret_v_u8mf2_i8mf2(v370);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v372 = 20 + v320;
        size_t v373 = 52 + v320;
        const uint8_t* v374 = v21 + v372;
        const int8_t* v375 = (const int8_t*) v374;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v376 = *(const int8_t *)(v375);
        const uint8_t* v377 = v21 + v373;
        const int8_t* v378 = (const int8_t*) v377;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v379 = *(const int8_t *)(v378);
        vint16m1_t v380 = v316;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v381 = __riscv_vwmacc_vx_i16m1(v380, v376, v369, 8);
        v316 = v381;
        vint16m1_t v382 = v318;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v383 = __riscv_vwmacc_vx_i16m1(v382, v379, v371, 8);
        v318 = v383;
      }
      vint16m1_t v384 = v312;
      vint16m1_t v385 = v314;
      vint16m1_t v386 = v316;
      vint16m1_t v387 = v318;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v388 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v389 = __riscv_vwmacc_vv_i32m2(v388, v57, v384, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v390 = __riscv_vwmacc_vv_i32m2(v389, v75, v385, 8);
      v34 = v390;
      vint32m2_t v391 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v392 = __riscv_vwmacc_vv_i32m2(v391, v129, v386, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v393 = __riscv_vwmacc_vv_i32m2(v392, v147, v387, 8);
      v38 = v393;
      vint16m1_t v394;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v395 = __riscv_vmv_v_x_i16m1(0, 8);
      v394 = v395;
      vint16m1_t v396;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v397 = __riscv_vmv_v_x_i16m1(0, 8);
      v396 = v397;
      vint16m1_t v398;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v399 = __riscv_vmv_v_x_i16m1(0, 8);
      v398 = v399;
      vint16m1_t v400;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v401 = __riscv_vmv_v_x_i16m1(0, 8);
      v400 = v401;
      for (size_t v402 = 0; v402 < 16; v402 += 1) {
        size_t v403 = v402 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v404 = 1280 + v403;
        const uint8_t* v405 = v19 + v404;
        const uint8_t* v406 = (const uint8_t*) v405;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v407 = __riscv_vle8_v_u8mf2(v406, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v408 = __riscv_vand_vx_u8mf2(v407, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v409 = __riscv_vsrl_vx_u8mf2(v407, 4, 8);
        size_t v410 = 256 + v403;
        const uint8_t* v411 = v19 + v410;
        const uint8_t* v412 = (const uint8_t*) v411;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v413 = __riscv_vle8_v_u8mf2(v412, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v414 = __riscv_vsrl_vx_u8mf2(v413, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v415 = __riscv_vand_vx_u8mf2(v414, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v416 = __riscv_vsll_vx_u8mf2(v415, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v417 = __riscv_vsrl_vx_u8mf2(v413, 3, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v418 = __riscv_vand_vx_u8mf2(v417, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v419 = __riscv_vsll_vx_u8mf2(v418, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v420 = __riscv_vor_vv_u8mf2(v408, v416, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v421 = __riscv_vreinterpret_v_u8mf2_i8mf2(v420);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v422 = __riscv_vor_vv_u8mf2(v409, v419, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v423 = __riscv_vreinterpret_v_u8mf2_i8mf2(v422);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v424 = 68 + v402;
        size_t v425 = 100 + v402;
        const uint8_t* v426 = v21 + v424;
        const int8_t* v427 = (const int8_t*) v426;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v428 = *(const int8_t *)(v427);
        const uint8_t* v429 = v21 + v425;
        const int8_t* v430 = (const int8_t*) v429;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v431 = *(const int8_t *)(v430);
        vint16m1_t v432 = v394;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v433 = __riscv_vwmacc_vx_i16m1(v432, v428, v421, 8);
        v394 = v433;
        vint16m1_t v434 = v396;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v435 = __riscv_vwmacc_vx_i16m1(v434, v431, v423, 8);
        v396 = v435;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v436 = 1288 + v403;
        const uint8_t* v437 = v19 + v436;
        const uint8_t* v438 = (const uint8_t*) v437;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v439 = __riscv_vle8_v_u8mf2(v438, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v440 = __riscv_vand_vx_u8mf2(v439, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v441 = __riscv_vsrl_vx_u8mf2(v439, 4, 8);
        size_t v442 = 264 + v403;
        const uint8_t* v443 = v19 + v442;
        const uint8_t* v444 = (const uint8_t*) v443;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v445 = __riscv_vle8_v_u8mf2(v444, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v446 = __riscv_vsrl_vx_u8mf2(v445, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v447 = __riscv_vand_vx_u8mf2(v446, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v448 = __riscv_vsll_vx_u8mf2(v447, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v449 = __riscv_vsrl_vx_u8mf2(v445, 3, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v450 = __riscv_vand_vx_u8mf2(v449, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v451 = __riscv_vsll_vx_u8mf2(v450, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v452 = __riscv_vor_vv_u8mf2(v440, v448, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v453 = __riscv_vreinterpret_v_u8mf2_i8mf2(v452);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v454 = __riscv_vor_vv_u8mf2(v441, v451, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v455 = __riscv_vreinterpret_v_u8mf2_i8mf2(v454);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v456 = 68 + v402;
        size_t v457 = 100 + v402;
        const uint8_t* v458 = v21 + v456;
        const int8_t* v459 = (const int8_t*) v458;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v460 = *(const int8_t *)(v459);
        const uint8_t* v461 = v21 + v457;
        const int8_t* v462 = (const int8_t*) v461;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v463 = *(const int8_t *)(v462);
        vint16m1_t v464 = v398;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v465 = __riscv_vwmacc_vx_i16m1(v464, v460, v453, 8);
        v398 = v465;
        vint16m1_t v466 = v400;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v467 = __riscv_vwmacc_vx_i16m1(v466, v463, v455, 8);
        v400 = v467;
      }
      vint16m1_t v468 = v394;
      vint16m1_t v469 = v396;
      vint16m1_t v470 = v398;
      vint16m1_t v471 = v400;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v472 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v473 = __riscv_vwmacc_vv_i32m2(v472, v93, v468, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v474 = __riscv_vwmacc_vv_i32m2(v473, v111, v469, 8);
      v34 = v474;
      vint32m2_t v475 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v476 = __riscv_vwmacc_vv_i32m2(v475, v165, v470, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v477 = __riscv_vwmacc_vv_i32m2(v476, v183, v471, 8);
      v38 = v477;
      vint16m1_t v478;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v479 = __riscv_vmv_v_x_i16m1(0, 8);
      v478 = v479;
      vint16m1_t v480;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v481 = __riscv_vmv_v_x_i16m1(0, 8);
      v480 = v481;
      vint16m1_t v482;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v483 = __riscv_vmv_v_x_i16m1(0, 8);
      v482 = v483;
      vint16m1_t v484;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v485 = __riscv_vmv_v_x_i16m1(0, 8);
      v484 = v485;
      for (size_t v486 = 0; v486 < 16; v486 += 1) {
        size_t v487 = v486 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v488 = 1536 + v487;
        const uint8_t* v489 = v19 + v488;
        const uint8_t* v490 = (const uint8_t*) v489;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v491 = __riscv_vle8_v_u8mf2(v490, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v492 = __riscv_vand_vx_u8mf2(v491, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v493 = __riscv_vsrl_vx_u8mf2(v491, 4, 8);
        size_t v494 = 512 + v487;
        const uint8_t* v495 = v19 + v494;
        const uint8_t* v496 = (const uint8_t*) v495;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v497 = __riscv_vle8_v_u8mf2(v496, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v498 = __riscv_vsrl_vx_u8mf2(v497, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v499 = __riscv_vand_vx_u8mf2(v498, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v500 = __riscv_vsll_vx_u8mf2(v499, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v501 = __riscv_vsrl_vx_u8mf2(v497, 3, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v502 = __riscv_vand_vx_u8mf2(v501, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v503 = __riscv_vsll_vx_u8mf2(v502, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v504 = __riscv_vor_vv_u8mf2(v492, v500, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v505 = __riscv_vreinterpret_v_u8mf2_i8mf2(v504);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v506 = __riscv_vor_vv_u8mf2(v493, v503, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v507 = __riscv_vreinterpret_v_u8mf2_i8mf2(v506);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v508 = 84 + v486;
        size_t v509 = 116 + v486;
        const uint8_t* v510 = v21 + v508;
        const int8_t* v511 = (const int8_t*) v510;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v512 = *(const int8_t *)(v511);
        const uint8_t* v513 = v21 + v509;
        const int8_t* v514 = (const int8_t*) v513;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v515 = *(const int8_t *)(v514);
        vint16m1_t v516 = v478;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v517 = __riscv_vwmacc_vx_i16m1(v516, v512, v505, 8);
        v478 = v517;
        vint16m1_t v518 = v480;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v519 = __riscv_vwmacc_vx_i16m1(v518, v515, v507, 8);
        v480 = v519;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v520 = 1544 + v487;
        const uint8_t* v521 = v19 + v520;
        const uint8_t* v522 = (const uint8_t*) v521;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v523 = __riscv_vle8_v_u8mf2(v522, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v524 = __riscv_vand_vx_u8mf2(v523, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v525 = __riscv_vsrl_vx_u8mf2(v523, 4, 8);
        size_t v526 = 520 + v487;
        const uint8_t* v527 = v19 + v526;
        const uint8_t* v528 = (const uint8_t*) v527;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v529 = __riscv_vle8_v_u8mf2(v528, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v530 = __riscv_vsrl_vx_u8mf2(v529, 2, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v531 = __riscv_vand_vx_u8mf2(v530, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v532 = __riscv_vsll_vx_u8mf2(v531, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v533 = __riscv_vsrl_vx_u8mf2(v529, 3, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v534 = __riscv_vand_vx_u8mf2(v533, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v535 = __riscv_vsll_vx_u8mf2(v534, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v536 = __riscv_vor_vv_u8mf2(v524, v532, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v537 = __riscv_vreinterpret_v_u8mf2_i8mf2(v536);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v538 = __riscv_vor_vv_u8mf2(v525, v535, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v539 = __riscv_vreinterpret_v_u8mf2_i8mf2(v538);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v540 = 84 + v486;
        size_t v541 = 116 + v486;
        const uint8_t* v542 = v21 + v540;
        const int8_t* v543 = (const int8_t*) v542;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v544 = *(const int8_t *)(v543);
        const uint8_t* v545 = v21 + v541;
        const int8_t* v546 = (const int8_t*) v545;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v547 = *(const int8_t *)(v546);
        vint16m1_t v548 = v482;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v549 = __riscv_vwmacc_vx_i16m1(v548, v544, v537, 8);
        v482 = v549;
        vint16m1_t v550 = v484;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v551 = __riscv_vwmacc_vx_i16m1(v550, v547, v539, 8);
        v484 = v551;
      }
      vint16m1_t v552 = v478;
      vint16m1_t v553 = v480;
      vint16m1_t v554 = v482;
      vint16m1_t v555 = v484;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v556 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v557 = __riscv_vwmacc_vv_i32m2(v556, v93, v552, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v558 = __riscv_vwmacc_vv_i32m2(v557, v111, v553, 8);
      v34 = v558;
      vint32m2_t v559 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v560 = __riscv_vwmacc_vv_i32m2(v559, v165, v554, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v561 = __riscv_vwmacc_vv_i32m2(v560, v183, v555, 8);
      v38 = v561;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
      const uint8_t* v562 = v19 + 128;
      const uint8_t* v563 = (const uint8_t*) v562;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v564 = __riscv_vle8_v_u8mf2(v563, 8);
      const uint8_t* v565 = v19 + 192;
      const uint8_t* v566 = (const uint8_t*) v565;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v567 = __riscv_vle8_v_u8mf2(v566, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v568 = __riscv_vand_vx_u8mf2(v564, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v569 = __riscv_vsrl_vx_u8mf2(v564, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v570 = __riscv_vand_vx_u8mf2(v567, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v571 = __riscv_vand_vx_u8mf2(v567, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v572 = __riscv_vsrl_vx_u8mf2(v571, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v573 = __riscv_vor_vv_u8mf2(v570, v568, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v574 = __riscv_vor_vv_u8mf2(v572, v569, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v575 = __riscv_vzext_vf2_u16m1(v573, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v576 = __riscv_vreinterpret_v_u16m1_i16m1(v575);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v577 = __riscv_vzext_vf2_u16m1(v574, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v578 = __riscv_vreinterpret_v_u16m1_i16m1(v577);
      const uint8_t* v579 = v19 + 144;
      const uint8_t* v580 = (const uint8_t*) v579;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v581 = __riscv_vle8_v_u8mf2(v580, 8);
      const uint8_t* v582 = v19 + 208;
      const uint8_t* v583 = (const uint8_t*) v582;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v584 = __riscv_vle8_v_u8mf2(v583, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v585 = __riscv_vand_vx_u8mf2(v581, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v586 = __riscv_vsrl_vx_u8mf2(v581, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v587 = __riscv_vand_vx_u8mf2(v584, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v588 = __riscv_vand_vx_u8mf2(v584, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v589 = __riscv_vsrl_vx_u8mf2(v588, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v590 = __riscv_vor_vv_u8mf2(v587, v585, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v591 = __riscv_vor_vv_u8mf2(v589, v586, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v592 = __riscv_vzext_vf2_u16m1(v590, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v593 = __riscv_vreinterpret_v_u16m1_i16m1(v592);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v594 = __riscv_vzext_vf2_u16m1(v591, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v595 = __riscv_vreinterpret_v_u16m1_i16m1(v594);
      const uint8_t* v596 = v19 + 160;
      const uint8_t* v597 = (const uint8_t*) v596;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v598 = __riscv_vle8_v_u8mf2(v597, 8);
      const uint8_t* v599 = v19 + 224;
      const uint8_t* v600 = (const uint8_t*) v599;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v601 = __riscv_vle8_v_u8mf2(v600, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v602 = __riscv_vand_vx_u8mf2(v598, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v603 = __riscv_vsrl_vx_u8mf2(v598, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v604 = __riscv_vand_vx_u8mf2(v601, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v605 = __riscv_vand_vx_u8mf2(v601, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v606 = __riscv_vsrl_vx_u8mf2(v605, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v607 = __riscv_vor_vv_u8mf2(v604, v602, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v608 = __riscv_vor_vv_u8mf2(v606, v603, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v609 = __riscv_vzext_vf2_u16m1(v607, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v610 = __riscv_vreinterpret_v_u16m1_i16m1(v609);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v611 = __riscv_vzext_vf2_u16m1(v608, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v612 = __riscv_vreinterpret_v_u16m1_i16m1(v611);
      const uint8_t* v613 = v19 + 176;
      const uint8_t* v614 = (const uint8_t*) v613;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v615 = __riscv_vle8_v_u8mf2(v614, 8);
      const uint8_t* v616 = v19 + 240;
      const uint8_t* v617 = (const uint8_t*) v616;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v618 = __riscv_vle8_v_u8mf2(v617, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v619 = __riscv_vand_vx_u8mf2(v615, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v620 = __riscv_vsrl_vx_u8mf2(v615, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v621 = __riscv_vand_vx_u8mf2(v618, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v622 = __riscv_vand_vx_u8mf2(v618, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v623 = __riscv_vsrl_vx_u8mf2(v622, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v624 = __riscv_vor_vv_u8mf2(v621, v619, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v625 = __riscv_vor_vv_u8mf2(v623, v620, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v626 = __riscv_vzext_vf2_u16m1(v624, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v627 = __riscv_vreinterpret_v_u16m1_i16m1(v626);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v628 = __riscv_vzext_vf2_u16m1(v625, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v629 = __riscv_vreinterpret_v_u16m1_i16m1(v628);
      const uint8_t* v630 = v19 + 136;
      const uint8_t* v631 = (const uint8_t*) v630;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v632 = __riscv_vle8_v_u8mf2(v631, 8);
      const uint8_t* v633 = v19 + 200;
      const uint8_t* v634 = (const uint8_t*) v633;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v635 = __riscv_vle8_v_u8mf2(v634, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v636 = __riscv_vand_vx_u8mf2(v632, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v637 = __riscv_vsrl_vx_u8mf2(v632, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v638 = __riscv_vand_vx_u8mf2(v635, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v639 = __riscv_vand_vx_u8mf2(v635, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v640 = __riscv_vsrl_vx_u8mf2(v639, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v641 = __riscv_vor_vv_u8mf2(v638, v636, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v642 = __riscv_vor_vv_u8mf2(v640, v637, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v643 = __riscv_vzext_vf2_u16m1(v641, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v644 = __riscv_vreinterpret_v_u16m1_i16m1(v643);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v645 = __riscv_vzext_vf2_u16m1(v642, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v646 = __riscv_vreinterpret_v_u16m1_i16m1(v645);
      const uint8_t* v647 = v19 + 152;
      const uint8_t* v648 = (const uint8_t*) v647;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v649 = __riscv_vle8_v_u8mf2(v648, 8);
      const uint8_t* v650 = v19 + 216;
      const uint8_t* v651 = (const uint8_t*) v650;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v652 = __riscv_vle8_v_u8mf2(v651, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v653 = __riscv_vand_vx_u8mf2(v649, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v654 = __riscv_vsrl_vx_u8mf2(v649, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v655 = __riscv_vand_vx_u8mf2(v652, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v656 = __riscv_vand_vx_u8mf2(v652, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v657 = __riscv_vsrl_vx_u8mf2(v656, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v658 = __riscv_vor_vv_u8mf2(v655, v653, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v659 = __riscv_vor_vv_u8mf2(v657, v654, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v660 = __riscv_vzext_vf2_u16m1(v658, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v661 = __riscv_vreinterpret_v_u16m1_i16m1(v660);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v662 = __riscv_vzext_vf2_u16m1(v659, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v663 = __riscv_vreinterpret_v_u16m1_i16m1(v662);
      const uint8_t* v664 = v19 + 168;
      const uint8_t* v665 = (const uint8_t*) v664;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v666 = __riscv_vle8_v_u8mf2(v665, 8);
      const uint8_t* v667 = v19 + 232;
      const uint8_t* v668 = (const uint8_t*) v667;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v669 = __riscv_vle8_v_u8mf2(v668, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v670 = __riscv_vand_vx_u8mf2(v666, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v671 = __riscv_vsrl_vx_u8mf2(v666, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v672 = __riscv_vand_vx_u8mf2(v669, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v673 = __riscv_vand_vx_u8mf2(v669, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v674 = __riscv_vsrl_vx_u8mf2(v673, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v675 = __riscv_vor_vv_u8mf2(v672, v670, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v676 = __riscv_vor_vv_u8mf2(v674, v671, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v677 = __riscv_vzext_vf2_u16m1(v675, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v678 = __riscv_vreinterpret_v_u16m1_i16m1(v677);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v679 = __riscv_vzext_vf2_u16m1(v676, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v680 = __riscv_vreinterpret_v_u16m1_i16m1(v679);
      const uint8_t* v681 = v19 + 184;
      const uint8_t* v682 = (const uint8_t*) v681;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v683 = __riscv_vle8_v_u8mf2(v682, 8);
      const uint8_t* v684 = v19 + 248;
      const uint8_t* v685 = (const uint8_t*) v684;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v686 = __riscv_vle8_v_u8mf2(v685, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v687 = __riscv_vand_vx_u8mf2(v683, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v688 = __riscv_vsrl_vx_u8mf2(v683, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v689 = __riscv_vand_vx_u8mf2(v686, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v690 = __riscv_vand_vx_u8mf2(v686, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v691 = __riscv_vsrl_vx_u8mf2(v690, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v692 = __riscv_vor_vv_u8mf2(v689, v687, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v693 = __riscv_vor_vv_u8mf2(v691, v688, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v694 = __riscv_vzext_vf2_u16m1(v692, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v695 = __riscv_vreinterpret_v_u16m1_i16m1(v694);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v696 = __riscv_vzext_vf2_u16m1(v693, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v697 = __riscv_vreinterpret_v_u16m1_i16m1(v696);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold
      const uint8_t* v698 = v21 + 276;
      const int16_t* v699 = (const int16_t*) v698;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v700 = *(const int16_t *)(v699);
      const uint8_t* v701 = v21 + 278;
      const int16_t* v702 = (const int16_t*) v701;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v703 = *(const int16_t *)(v702);
      int32_t v704 = v700 + v703;
      vint32m2_t v705 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v706 = __riscv_vwmacc_vx_i32m2(v705, v704, v578, 8);
      v36 = v706;
      vint32m2_t v707 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v708 = __riscv_vwmacc_vx_i32m2(v707, v704, v646, 8);
      v40 = v708;
      const uint8_t* v709 = v21 + 280;
      const int16_t* v710 = (const int16_t*) v709;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v711 = *(const int16_t *)(v710);
      const uint8_t* v712 = v21 + 282;
      const int16_t* v713 = (const int16_t*) v712;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v714 = *(const int16_t *)(v713);
      int32_t v715 = v711 + v714;
      vint32m2_t v716 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v717 = __riscv_vwmacc_vx_i32m2(v716, v715, v595, 8);
      v36 = v717;
      vint32m2_t v718 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v719 = __riscv_vwmacc_vx_i32m2(v718, v715, v663, 8);
      v40 = v719;
      const uint8_t* v720 = v21 + 284;
      const int16_t* v721 = (const int16_t*) v720;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v722 = *(const int16_t *)(v721);
      const uint8_t* v723 = v21 + 286;
      const int16_t* v724 = (const int16_t*) v723;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v725 = *(const int16_t *)(v724);
      int32_t v726 = v722 + v725;
      vint32m2_t v727 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v728 = __riscv_vwmacc_vx_i32m2(v727, v726, v612, 8);
      v36 = v728;
      vint32m2_t v729 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v730 = __riscv_vwmacc_vx_i32m2(v729, v726, v680, 8);
      v40 = v730;
      const uint8_t* v731 = v21 + 288;
      const int16_t* v732 = (const int16_t*) v731;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v733 = *(const int16_t *)(v732);
      const uint8_t* v734 = v21 + 290;
      const int16_t* v735 = (const int16_t*) v734;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v736 = *(const int16_t *)(v735);
      int32_t v737 = v733 + v736;
      vint32m2_t v738 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v739 = __riscv_vwmacc_vx_i32m2(v738, v737, v629, 8);
      v36 = v739;
      vint32m2_t v740 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v741 = __riscv_vwmacc_vx_i32m2(v740, v737, v697, 8);
      v40 = v741;
      vint16m1_t v742;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v743 = __riscv_vmv_v_x_i16m1(0, 8);
      v742 = v743;
      vint16m1_t v744;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v745 = __riscv_vmv_v_x_i16m1(0, 8);
      v744 = v745;
      vint16m1_t v746;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v747 = __riscv_vmv_v_x_i16m1(0, 8);
      v746 = v747;
      vint16m1_t v748;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v749 = __riscv_vmv_v_x_i16m1(0, 8);
      v748 = v749;
      for (size_t v750 = 0; v750 < 16; v750 += 1) {
        size_t v751 = v750 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v752 = 1792 + v751;
        const uint8_t* v753 = v19 + v752;
        const uint8_t* v754 = (const uint8_t*) v753;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v755 = __riscv_vle8_v_u8mf2(v754, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v756 = __riscv_vand_vx_u8mf2(v755, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v757 = __riscv_vsrl_vx_u8mf2(v755, 4, 8);
        size_t v758 = 256 + v751;
        const uint8_t* v759 = v19 + v758;
        const uint8_t* v760 = (const uint8_t*) v759;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v761 = __riscv_vle8_v_u8mf2(v760, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v762 = __riscv_vsrl_vx_u8mf2(v761, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v763 = __riscv_vand_vx_u8mf2(v762, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v764 = __riscv_vsll_vx_u8mf2(v763, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v765 = __riscv_vsrl_vx_u8mf2(v761, 5, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v766 = __riscv_vand_vx_u8mf2(v765, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v767 = __riscv_vsll_vx_u8mf2(v766, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v768 = __riscv_vor_vv_u8mf2(v756, v764, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v769 = __riscv_vreinterpret_v_u8mf2_i8mf2(v768);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v770 = __riscv_vor_vv_u8mf2(v757, v767, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v771 = __riscv_vreinterpret_v_u8mf2_i8mf2(v770);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v772 = 132 + v750;
        size_t v773 = 164 + v750;
        const uint8_t* v774 = v21 + v772;
        const int8_t* v775 = (const int8_t*) v774;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v776 = *(const int8_t *)(v775);
        const uint8_t* v777 = v21 + v773;
        const int8_t* v778 = (const int8_t*) v777;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v779 = *(const int8_t *)(v778);
        vint16m1_t v780 = v742;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v781 = __riscv_vwmacc_vx_i16m1(v780, v776, v769, 8);
        v742 = v781;
        vint16m1_t v782 = v744;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v783 = __riscv_vwmacc_vx_i16m1(v782, v779, v771, 8);
        v744 = v783;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v784 = 1800 + v751;
        const uint8_t* v785 = v19 + v784;
        const uint8_t* v786 = (const uint8_t*) v785;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v787 = __riscv_vle8_v_u8mf2(v786, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v788 = __riscv_vand_vx_u8mf2(v787, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v789 = __riscv_vsrl_vx_u8mf2(v787, 4, 8);
        size_t v790 = 264 + v751;
        const uint8_t* v791 = v19 + v790;
        const uint8_t* v792 = (const uint8_t*) v791;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v793 = __riscv_vle8_v_u8mf2(v792, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v794 = __riscv_vsrl_vx_u8mf2(v793, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v795 = __riscv_vand_vx_u8mf2(v794, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v796 = __riscv_vsll_vx_u8mf2(v795, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v797 = __riscv_vsrl_vx_u8mf2(v793, 5, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v798 = __riscv_vand_vx_u8mf2(v797, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v799 = __riscv_vsll_vx_u8mf2(v798, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v800 = __riscv_vor_vv_u8mf2(v788, v796, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v801 = __riscv_vreinterpret_v_u8mf2_i8mf2(v800);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v802 = __riscv_vor_vv_u8mf2(v789, v799, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v803 = __riscv_vreinterpret_v_u8mf2_i8mf2(v802);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v804 = 132 + v750;
        size_t v805 = 164 + v750;
        const uint8_t* v806 = v21 + v804;
        const int8_t* v807 = (const int8_t*) v806;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v808 = *(const int8_t *)(v807);
        const uint8_t* v809 = v21 + v805;
        const int8_t* v810 = (const int8_t*) v809;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v811 = *(const int8_t *)(v810);
        vint16m1_t v812 = v746;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v813 = __riscv_vwmacc_vx_i16m1(v812, v808, v801, 8);
        v746 = v813;
        vint16m1_t v814 = v748;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v815 = __riscv_vwmacc_vx_i16m1(v814, v811, v803, 8);
        v748 = v815;
      }
      vint16m1_t v816 = v742;
      vint16m1_t v817 = v744;
      vint16m1_t v818 = v746;
      vint16m1_t v819 = v748;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v820 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v821 = __riscv_vwmacc_vv_i32m2(v820, v576, v816, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v822 = __riscv_vwmacc_vv_i32m2(v821, v593, v817, 8);
      v34 = v822;
      vint32m2_t v823 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v824 = __riscv_vwmacc_vv_i32m2(v823, v644, v818, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v825 = __riscv_vwmacc_vv_i32m2(v824, v661, v819, 8);
      v38 = v825;
      vint16m1_t v826;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v827 = __riscv_vmv_v_x_i16m1(0, 8);
      v826 = v827;
      vint16m1_t v828;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v829 = __riscv_vmv_v_x_i16m1(0, 8);
      v828 = v829;
      vint16m1_t v830;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v831 = __riscv_vmv_v_x_i16m1(0, 8);
      v830 = v831;
      vint16m1_t v832;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v833 = __riscv_vmv_v_x_i16m1(0, 8);
      v832 = v833;
      for (size_t v834 = 0; v834 < 16; v834 += 1) {
        size_t v835 = v834 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v836 = 2048 + v835;
        const uint8_t* v837 = v19 + v836;
        const uint8_t* v838 = (const uint8_t*) v837;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v839 = __riscv_vle8_v_u8mf2(v838, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v840 = __riscv_vand_vx_u8mf2(v839, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v841 = __riscv_vsrl_vx_u8mf2(v839, 4, 8);
        size_t v842 = 512 + v835;
        const uint8_t* v843 = v19 + v842;
        const uint8_t* v844 = (const uint8_t*) v843;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v845 = __riscv_vle8_v_u8mf2(v844, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v846 = __riscv_vsrl_vx_u8mf2(v845, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v847 = __riscv_vand_vx_u8mf2(v846, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v848 = __riscv_vsll_vx_u8mf2(v847, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v849 = __riscv_vsrl_vx_u8mf2(v845, 5, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v850 = __riscv_vand_vx_u8mf2(v849, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v851 = __riscv_vsll_vx_u8mf2(v850, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v852 = __riscv_vor_vv_u8mf2(v840, v848, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v853 = __riscv_vreinterpret_v_u8mf2_i8mf2(v852);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v854 = __riscv_vor_vv_u8mf2(v841, v851, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v855 = __riscv_vreinterpret_v_u8mf2_i8mf2(v854);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v856 = 148 + v834;
        size_t v857 = 180 + v834;
        const uint8_t* v858 = v21 + v856;
        const int8_t* v859 = (const int8_t*) v858;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v860 = *(const int8_t *)(v859);
        const uint8_t* v861 = v21 + v857;
        const int8_t* v862 = (const int8_t*) v861;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v863 = *(const int8_t *)(v862);
        vint16m1_t v864 = v826;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v865 = __riscv_vwmacc_vx_i16m1(v864, v860, v853, 8);
        v826 = v865;
        vint16m1_t v866 = v828;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v867 = __riscv_vwmacc_vx_i16m1(v866, v863, v855, 8);
        v828 = v867;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v868 = 2056 + v835;
        const uint8_t* v869 = v19 + v868;
        const uint8_t* v870 = (const uint8_t*) v869;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v871 = __riscv_vle8_v_u8mf2(v870, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v872 = __riscv_vand_vx_u8mf2(v871, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v873 = __riscv_vsrl_vx_u8mf2(v871, 4, 8);
        size_t v874 = 520 + v835;
        const uint8_t* v875 = v19 + v874;
        const uint8_t* v876 = (const uint8_t*) v875;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v877 = __riscv_vle8_v_u8mf2(v876, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v878 = __riscv_vsrl_vx_u8mf2(v877, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v879 = __riscv_vand_vx_u8mf2(v878, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v880 = __riscv_vsll_vx_u8mf2(v879, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v881 = __riscv_vsrl_vx_u8mf2(v877, 5, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v882 = __riscv_vand_vx_u8mf2(v881, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v883 = __riscv_vsll_vx_u8mf2(v882, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v884 = __riscv_vor_vv_u8mf2(v872, v880, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v885 = __riscv_vreinterpret_v_u8mf2_i8mf2(v884);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v886 = __riscv_vor_vv_u8mf2(v873, v883, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v887 = __riscv_vreinterpret_v_u8mf2_i8mf2(v886);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v888 = 148 + v834;
        size_t v889 = 180 + v834;
        const uint8_t* v890 = v21 + v888;
        const int8_t* v891 = (const int8_t*) v890;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v892 = *(const int8_t *)(v891);
        const uint8_t* v893 = v21 + v889;
        const int8_t* v894 = (const int8_t*) v893;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v895 = *(const int8_t *)(v894);
        vint16m1_t v896 = v830;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v897 = __riscv_vwmacc_vx_i16m1(v896, v892, v885, 8);
        v830 = v897;
        vint16m1_t v898 = v832;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v899 = __riscv_vwmacc_vx_i16m1(v898, v895, v887, 8);
        v832 = v899;
      }
      vint16m1_t v900 = v826;
      vint16m1_t v901 = v828;
      vint16m1_t v902 = v830;
      vint16m1_t v903 = v832;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v904 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v905 = __riscv_vwmacc_vv_i32m2(v904, v576, v900, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v906 = __riscv_vwmacc_vv_i32m2(v905, v593, v901, 8);
      v34 = v906;
      vint32m2_t v907 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v908 = __riscv_vwmacc_vv_i32m2(v907, v644, v902, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v909 = __riscv_vwmacc_vv_i32m2(v908, v661, v903, 8);
      v38 = v909;
      vint16m1_t v910;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v911 = __riscv_vmv_v_x_i16m1(0, 8);
      v910 = v911;
      vint16m1_t v912;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v913 = __riscv_vmv_v_x_i16m1(0, 8);
      v912 = v913;
      vint16m1_t v914;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v915 = __riscv_vmv_v_x_i16m1(0, 8);
      v914 = v915;
      vint16m1_t v916;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v917 = __riscv_vmv_v_x_i16m1(0, 8);
      v916 = v917;
      for (size_t v918 = 0; v918 < 16; v918 += 1) {
        size_t v919 = v918 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v920 = 2304 + v919;
        const uint8_t* v921 = v19 + v920;
        const uint8_t* v922 = (const uint8_t*) v921;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v923 = __riscv_vle8_v_u8mf2(v922, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v924 = __riscv_vand_vx_u8mf2(v923, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v925 = __riscv_vsrl_vx_u8mf2(v923, 4, 8);
        size_t v926 = 256 + v919;
        const uint8_t* v927 = v19 + v926;
        const uint8_t* v928 = (const uint8_t*) v927;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v929 = __riscv_vle8_v_u8mf2(v928, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v930 = __riscv_vsrl_vx_u8mf2(v929, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v931 = __riscv_vand_vx_u8mf2(v930, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v932 = __riscv_vsll_vx_u8mf2(v931, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v933 = __riscv_vsrl_vx_u8mf2(v929, 7, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v934 = __riscv_vand_vx_u8mf2(v933, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v935 = __riscv_vsll_vx_u8mf2(v934, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v936 = __riscv_vor_vv_u8mf2(v924, v932, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v937 = __riscv_vreinterpret_v_u8mf2_i8mf2(v936);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v938 = __riscv_vor_vv_u8mf2(v925, v935, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v939 = __riscv_vreinterpret_v_u8mf2_i8mf2(v938);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v940 = 196 + v918;
        size_t v941 = 228 + v918;
        const uint8_t* v942 = v21 + v940;
        const int8_t* v943 = (const int8_t*) v942;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v944 = *(const int8_t *)(v943);
        const uint8_t* v945 = v21 + v941;
        const int8_t* v946 = (const int8_t*) v945;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v947 = *(const int8_t *)(v946);
        vint16m1_t v948 = v910;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v949 = __riscv_vwmacc_vx_i16m1(v948, v944, v937, 8);
        v910 = v949;
        vint16m1_t v950 = v912;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v951 = __riscv_vwmacc_vx_i16m1(v950, v947, v939, 8);
        v912 = v951;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v952 = 2312 + v919;
        const uint8_t* v953 = v19 + v952;
        const uint8_t* v954 = (const uint8_t*) v953;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v955 = __riscv_vle8_v_u8mf2(v954, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v956 = __riscv_vand_vx_u8mf2(v955, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v957 = __riscv_vsrl_vx_u8mf2(v955, 4, 8);
        size_t v958 = 264 + v919;
        const uint8_t* v959 = v19 + v958;
        const uint8_t* v960 = (const uint8_t*) v959;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v961 = __riscv_vle8_v_u8mf2(v960, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v962 = __riscv_vsrl_vx_u8mf2(v961, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v963 = __riscv_vand_vx_u8mf2(v962, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v964 = __riscv_vsll_vx_u8mf2(v963, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v965 = __riscv_vsrl_vx_u8mf2(v961, 7, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v966 = __riscv_vand_vx_u8mf2(v965, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v967 = __riscv_vsll_vx_u8mf2(v966, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v968 = __riscv_vor_vv_u8mf2(v956, v964, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v969 = __riscv_vreinterpret_v_u8mf2_i8mf2(v968);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v970 = __riscv_vor_vv_u8mf2(v957, v967, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v971 = __riscv_vreinterpret_v_u8mf2_i8mf2(v970);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v972 = 196 + v918;
        size_t v973 = 228 + v918;
        const uint8_t* v974 = v21 + v972;
        const int8_t* v975 = (const int8_t*) v974;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v976 = *(const int8_t *)(v975);
        const uint8_t* v977 = v21 + v973;
        const int8_t* v978 = (const int8_t*) v977;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v979 = *(const int8_t *)(v978);
        vint16m1_t v980 = v914;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v981 = __riscv_vwmacc_vx_i16m1(v980, v976, v969, 8);
        v914 = v981;
        vint16m1_t v982 = v916;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v983 = __riscv_vwmacc_vx_i16m1(v982, v979, v971, 8);
        v916 = v983;
      }
      vint16m1_t v984 = v910;
      vint16m1_t v985 = v912;
      vint16m1_t v986 = v914;
      vint16m1_t v987 = v916;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v988 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v989 = __riscv_vwmacc_vv_i32m2(v988, v610, v984, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v990 = __riscv_vwmacc_vv_i32m2(v989, v627, v985, 8);
      v34 = v990;
      vint32m2_t v991 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v992 = __riscv_vwmacc_vv_i32m2(v991, v678, v986, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v993 = __riscv_vwmacc_vv_i32m2(v992, v695, v987, 8);
      v38 = v993;
      vint16m1_t v994;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v995 = __riscv_vmv_v_x_i16m1(0, 8);
      v994 = v995;
      vint16m1_t v996;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v997 = __riscv_vmv_v_x_i16m1(0, 8);
      v996 = v997;
      vint16m1_t v998;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v999 = __riscv_vmv_v_x_i16m1(0, 8);
      v998 = v999;
      vint16m1_t v1000;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1001 = __riscv_vmv_v_x_i16m1(0, 8);
      v1000 = v1001;
      for (size_t v1002 = 0; v1002 < 16; v1002 += 1) {
        size_t v1003 = v1002 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v1004 = 2560 + v1003;
        const uint8_t* v1005 = v19 + v1004;
        const uint8_t* v1006 = (const uint8_t*) v1005;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1007 = __riscv_vle8_v_u8mf2(v1006, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1008 = __riscv_vand_vx_u8mf2(v1007, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1009 = __riscv_vsrl_vx_u8mf2(v1007, 4, 8);
        size_t v1010 = 512 + v1003;
        const uint8_t* v1011 = v19 + v1010;
        const uint8_t* v1012 = (const uint8_t*) v1011;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1013 = __riscv_vle8_v_u8mf2(v1012, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1014 = __riscv_vsrl_vx_u8mf2(v1013, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1015 = __riscv_vand_vx_u8mf2(v1014, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1016 = __riscv_vsll_vx_u8mf2(v1015, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1017 = __riscv_vsrl_vx_u8mf2(v1013, 7, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1018 = __riscv_vand_vx_u8mf2(v1017, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1019 = __riscv_vsll_vx_u8mf2(v1018, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1020 = __riscv_vor_vv_u8mf2(v1008, v1016, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1021 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1020);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1022 = __riscv_vor_vv_u8mf2(v1009, v1019, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1023 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1022);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v1024 = 212 + v1002;
        size_t v1025 = 244 + v1002;
        const uint8_t* v1026 = v21 + v1024;
        const int8_t* v1027 = (const int8_t*) v1026;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1028 = *(const int8_t *)(v1027);
        const uint8_t* v1029 = v21 + v1025;
        const int8_t* v1030 = (const int8_t*) v1029;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1031 = *(const int8_t *)(v1030);
        vint16m1_t v1032 = v994;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1033 = __riscv_vwmacc_vx_i16m1(v1032, v1028, v1021, 8);
        v994 = v1033;
        vint16m1_t v1034 = v996;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1035 = __riscv_vwmacc_vx_i16m1(v1034, v1031, v1023, 8);
        v996 = v1035;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v1036 = 2568 + v1003;
        const uint8_t* v1037 = v19 + v1036;
        const uint8_t* v1038 = (const uint8_t*) v1037;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1039 = __riscv_vle8_v_u8mf2(v1038, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1040 = __riscv_vand_vx_u8mf2(v1039, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1041 = __riscv_vsrl_vx_u8mf2(v1039, 4, 8);
        size_t v1042 = 520 + v1003;
        const uint8_t* v1043 = v19 + v1042;
        const uint8_t* v1044 = (const uint8_t*) v1043;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v1045 = __riscv_vle8_v_u8mf2(v1044, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1046 = __riscv_vsrl_vx_u8mf2(v1045, 6, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1047 = __riscv_vand_vx_u8mf2(v1046, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1048 = __riscv_vsll_vx_u8mf2(v1047, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v1049 = __riscv_vsrl_vx_u8mf2(v1045, 7, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v1050 = __riscv_vand_vx_u8mf2(v1049, 0x01, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
        vuint8mf2_t v1051 = __riscv_vsll_vx_u8mf2(v1050, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1052 = __riscv_vor_vv_u8mf2(v1040, v1048, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1053 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1052);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
        vuint8mf2_t v1054 = __riscv_vor_vv_u8mf2(v1041, v1051, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v1055 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1054);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v1056 = 212 + v1002;
        size_t v1057 = 244 + v1002;
        const uint8_t* v1058 = v21 + v1056;
        const int8_t* v1059 = (const int8_t*) v1058;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1060 = *(const int8_t *)(v1059);
        const uint8_t* v1061 = v21 + v1057;
        const int8_t* v1062 = (const int8_t*) v1061;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v1063 = *(const int8_t *)(v1062);
        vint16m1_t v1064 = v998;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1065 = __riscv_vwmacc_vx_i16m1(v1064, v1060, v1053, 8);
        v998 = v1065;
        vint16m1_t v1066 = v1000;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v1067 = __riscv_vwmacc_vx_i16m1(v1066, v1063, v1055, 8);
        v1000 = v1067;
      }
      vint16m1_t v1068 = v994;
      vint16m1_t v1069 = v996;
      vint16m1_t v1070 = v998;
      vint16m1_t v1071 = v1000;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v1072 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1073 = __riscv_vwmacc_vv_i32m2(v1072, v610, v1068, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1074 = __riscv_vwmacc_vv_i32m2(v1073, v627, v1069, 8);
      v34 = v1074;
      vint32m2_t v1075 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1076 = __riscv_vwmacc_vv_i32m2(v1075, v678, v1070, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1077 = __riscv_vwmacc_vv_i32m2(v1076, v695, v1071, 8);
      v38 = v1077;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v1078 = (const _Float16*) v19;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v1079 = __riscv_vle16_v_f16m1(v1078, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v1080 = __riscv_vfwcvt_f_f_v_f32m2(v1079, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v1081 = __riscv_vfmul_vf_f32m2(v1080, v23, 8);
      vint32m2_t v1082 = v34;
      vfloat32m2_t v1083 = v13;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v1084 = __riscv_vfcvt_f_x_v_f32m2(v1082, 8);
      vfloat32m2_t v1085 = __riscv_vfmacc_vv_f32m2(v1083, v1084, v1081, 8);
      vint32m2_t v1086 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v1087 = __riscv_vfcvt_f_x_v_f32m2(v1086, 8);
      vfloat32m2_t v1088 = __riscv_vfnmsac_vv_f32m2(v1085, v28, v1087, 8);
      v13 = v1088;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v1089 = v19 + 16;
      const _Float16* v1090 = (const _Float16*) v1089;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v1091 = __riscv_vle16_v_f16m1(v1090, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v1092 = __riscv_vfwcvt_f_f_v_f32m2(v1091, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v1093 = __riscv_vfmul_vf_f32m2(v1092, v23, 8);
      vint32m2_t v1094 = v38;
      vfloat32m2_t v1095 = v15;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v1096 = __riscv_vfcvt_f_x_v_f32m2(v1094, 8);
      vfloat32m2_t v1097 = __riscv_vfmacc_vv_f32m2(v1095, v1096, v1093, 8);
      vint32m2_t v1098 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v1099 = __riscv_vfcvt_f_x_v_f32m2(v1098, 8);
      vfloat32m2_t v1100 = __riscv_vfnmsac_vv_f32m2(v1097, v33, v1099, 8);
      v15 = v1100;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v1101 = v9 * 16;
    float* v1102 = v2 + v1101;
    vfloat32m2_t v1103 = v13;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v1102, v1103, 8);
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v1104 = v9 * 16;
    size_t v1105 = v1104 + 8;
    float* v1106 = v2 + v1105;
    vfloat32m2_t v1107 = v15;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v1106, v1107, 8);
  }
  return;
}


