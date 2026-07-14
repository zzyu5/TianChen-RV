#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
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
    size_t v11 = v10 * 2304;
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
      size_t v18 = v17 * 2304;
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
        size_t v240 = 256 + v239;
        const uint8_t* v241 = v19 + v240;
        const uint8_t* v242 = (const uint8_t*) v241;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v243 = __riscv_vle8_v_u8mf2(v242, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v244 = __riscv_vand_vx_u8mf2(v243, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v245 = __riscv_vreinterpret_v_u8mf2_i8mf2(v244);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v246 = __riscv_vsrl_vx_u8mf2(v243, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v247 = __riscv_vreinterpret_v_u8mf2_i8mf2(v246);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v248 = 4 + v238;
        size_t v249 = 36 + v238;
        const uint8_t* v250 = v21 + v248;
        const int8_t* v251 = (const int8_t*) v250;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v252 = *(const int8_t *)(v251);
        const uint8_t* v253 = v21 + v249;
        const int8_t* v254 = (const int8_t*) v253;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v255 = *(const int8_t *)(v254);
        vint16m1_t v256 = v230;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v257 = __riscv_vwmacc_vx_i16m1(v256, v252, v245, 8);
        v230 = v257;
        vint16m1_t v258 = v232;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v259 = __riscv_vwmacc_vx_i16m1(v258, v255, v247, 8);
        v232 = v259;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v260 = 264 + v239;
        const uint8_t* v261 = v19 + v260;
        const uint8_t* v262 = (const uint8_t*) v261;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v263 = __riscv_vle8_v_u8mf2(v262, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v264 = __riscv_vand_vx_u8mf2(v263, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v265 = __riscv_vreinterpret_v_u8mf2_i8mf2(v264);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v266 = __riscv_vsrl_vx_u8mf2(v263, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v267 = __riscv_vreinterpret_v_u8mf2_i8mf2(v266);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v268 = 4 + v238;
        size_t v269 = 36 + v238;
        const uint8_t* v270 = v21 + v268;
        const int8_t* v271 = (const int8_t*) v270;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v272 = *(const int8_t *)(v271);
        const uint8_t* v273 = v21 + v269;
        const int8_t* v274 = (const int8_t*) v273;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v275 = *(const int8_t *)(v274);
        vint16m1_t v276 = v234;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v277 = __riscv_vwmacc_vx_i16m1(v276, v272, v265, 8);
        v234 = v277;
        vint16m1_t v278 = v236;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v279 = __riscv_vwmacc_vx_i16m1(v278, v275, v267, 8);
        v236 = v279;
      }
      vint16m1_t v280 = v230;
      vint16m1_t v281 = v232;
      vint16m1_t v282 = v234;
      vint16m1_t v283 = v236;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v284 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v285 = __riscv_vwmacc_vv_i32m2(v284, v57, v280, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v286 = __riscv_vwmacc_vv_i32m2(v285, v75, v281, 8);
      v34 = v286;
      vint32m2_t v287 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v288 = __riscv_vwmacc_vv_i32m2(v287, v129, v282, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v289 = __riscv_vwmacc_vv_i32m2(v288, v147, v283, 8);
      v38 = v289;
      vint16m1_t v290;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v291 = __riscv_vmv_v_x_i16m1(0, 8);
      v290 = v291;
      vint16m1_t v292;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v293 = __riscv_vmv_v_x_i16m1(0, 8);
      v292 = v293;
      vint16m1_t v294;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v295 = __riscv_vmv_v_x_i16m1(0, 8);
      v294 = v295;
      vint16m1_t v296;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v297 = __riscv_vmv_v_x_i16m1(0, 8);
      v296 = v297;
      for (size_t v298 = 0; v298 < 16; v298 += 1) {
        size_t v299 = v298 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v300 = 512 + v299;
        const uint8_t* v301 = v19 + v300;
        const uint8_t* v302 = (const uint8_t*) v301;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v303 = __riscv_vle8_v_u8mf2(v302, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v304 = __riscv_vand_vx_u8mf2(v303, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v305 = __riscv_vreinterpret_v_u8mf2_i8mf2(v304);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v306 = __riscv_vsrl_vx_u8mf2(v303, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v307 = __riscv_vreinterpret_v_u8mf2_i8mf2(v306);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v308 = 20 + v298;
        size_t v309 = 52 + v298;
        const uint8_t* v310 = v21 + v308;
        const int8_t* v311 = (const int8_t*) v310;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v312 = *(const int8_t *)(v311);
        const uint8_t* v313 = v21 + v309;
        const int8_t* v314 = (const int8_t*) v313;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v315 = *(const int8_t *)(v314);
        vint16m1_t v316 = v290;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v317 = __riscv_vwmacc_vx_i16m1(v316, v312, v305, 8);
        v290 = v317;
        vint16m1_t v318 = v292;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v319 = __riscv_vwmacc_vx_i16m1(v318, v315, v307, 8);
        v292 = v319;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v320 = 520 + v299;
        const uint8_t* v321 = v19 + v320;
        const uint8_t* v322 = (const uint8_t*) v321;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v323 = __riscv_vle8_v_u8mf2(v322, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v324 = __riscv_vand_vx_u8mf2(v323, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v325 = __riscv_vreinterpret_v_u8mf2_i8mf2(v324);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v326 = __riscv_vsrl_vx_u8mf2(v323, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v327 = __riscv_vreinterpret_v_u8mf2_i8mf2(v326);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v328 = 20 + v298;
        size_t v329 = 52 + v298;
        const uint8_t* v330 = v21 + v328;
        const int8_t* v331 = (const int8_t*) v330;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v332 = *(const int8_t *)(v331);
        const uint8_t* v333 = v21 + v329;
        const int8_t* v334 = (const int8_t*) v333;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v335 = *(const int8_t *)(v334);
        vint16m1_t v336 = v294;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v337 = __riscv_vwmacc_vx_i16m1(v336, v332, v325, 8);
        v294 = v337;
        vint16m1_t v338 = v296;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v339 = __riscv_vwmacc_vx_i16m1(v338, v335, v327, 8);
        v296 = v339;
      }
      vint16m1_t v340 = v290;
      vint16m1_t v341 = v292;
      vint16m1_t v342 = v294;
      vint16m1_t v343 = v296;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v344 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v345 = __riscv_vwmacc_vv_i32m2(v344, v57, v340, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v346 = __riscv_vwmacc_vv_i32m2(v345, v75, v341, 8);
      v34 = v346;
      vint32m2_t v347 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v348 = __riscv_vwmacc_vv_i32m2(v347, v129, v342, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v349 = __riscv_vwmacc_vv_i32m2(v348, v147, v343, 8);
      v38 = v349;
      vint16m1_t v350;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v351 = __riscv_vmv_v_x_i16m1(0, 8);
      v350 = v351;
      vint16m1_t v352;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v353 = __riscv_vmv_v_x_i16m1(0, 8);
      v352 = v353;
      vint16m1_t v354;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v355 = __riscv_vmv_v_x_i16m1(0, 8);
      v354 = v355;
      vint16m1_t v356;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v357 = __riscv_vmv_v_x_i16m1(0, 8);
      v356 = v357;
      for (size_t v358 = 0; v358 < 16; v358 += 1) {
        size_t v359 = v358 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v360 = 768 + v359;
        const uint8_t* v361 = v19 + v360;
        const uint8_t* v362 = (const uint8_t*) v361;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v363 = __riscv_vle8_v_u8mf2(v362, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v364 = __riscv_vand_vx_u8mf2(v363, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v365 = __riscv_vreinterpret_v_u8mf2_i8mf2(v364);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v366 = __riscv_vsrl_vx_u8mf2(v363, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v367 = __riscv_vreinterpret_v_u8mf2_i8mf2(v366);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v368 = 68 + v358;
        size_t v369 = 100 + v358;
        const uint8_t* v370 = v21 + v368;
        const int8_t* v371 = (const int8_t*) v370;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v372 = *(const int8_t *)(v371);
        const uint8_t* v373 = v21 + v369;
        const int8_t* v374 = (const int8_t*) v373;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v375 = *(const int8_t *)(v374);
        vint16m1_t v376 = v350;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v377 = __riscv_vwmacc_vx_i16m1(v376, v372, v365, 8);
        v350 = v377;
        vint16m1_t v378 = v352;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v379 = __riscv_vwmacc_vx_i16m1(v378, v375, v367, 8);
        v352 = v379;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v380 = 776 + v359;
        const uint8_t* v381 = v19 + v380;
        const uint8_t* v382 = (const uint8_t*) v381;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v383 = __riscv_vle8_v_u8mf2(v382, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v384 = __riscv_vand_vx_u8mf2(v383, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v385 = __riscv_vreinterpret_v_u8mf2_i8mf2(v384);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v386 = __riscv_vsrl_vx_u8mf2(v383, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v387 = __riscv_vreinterpret_v_u8mf2_i8mf2(v386);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v388 = 68 + v358;
        size_t v389 = 100 + v358;
        const uint8_t* v390 = v21 + v388;
        const int8_t* v391 = (const int8_t*) v390;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v392 = *(const int8_t *)(v391);
        const uint8_t* v393 = v21 + v389;
        const int8_t* v394 = (const int8_t*) v393;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v395 = *(const int8_t *)(v394);
        vint16m1_t v396 = v354;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v397 = __riscv_vwmacc_vx_i16m1(v396, v392, v385, 8);
        v354 = v397;
        vint16m1_t v398 = v356;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v399 = __riscv_vwmacc_vx_i16m1(v398, v395, v387, 8);
        v356 = v399;
      }
      vint16m1_t v400 = v350;
      vint16m1_t v401 = v352;
      vint16m1_t v402 = v354;
      vint16m1_t v403 = v356;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v404 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v405 = __riscv_vwmacc_vv_i32m2(v404, v93, v400, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v406 = __riscv_vwmacc_vv_i32m2(v405, v111, v401, 8);
      v34 = v406;
      vint32m2_t v407 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v408 = __riscv_vwmacc_vv_i32m2(v407, v165, v402, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v409 = __riscv_vwmacc_vv_i32m2(v408, v183, v403, 8);
      v38 = v409;
      vint16m1_t v410;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v411 = __riscv_vmv_v_x_i16m1(0, 8);
      v410 = v411;
      vint16m1_t v412;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v413 = __riscv_vmv_v_x_i16m1(0, 8);
      v412 = v413;
      vint16m1_t v414;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v415 = __riscv_vmv_v_x_i16m1(0, 8);
      v414 = v415;
      vint16m1_t v416;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v417 = __riscv_vmv_v_x_i16m1(0, 8);
      v416 = v417;
      for (size_t v418 = 0; v418 < 16; v418 += 1) {
        size_t v419 = v418 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v420 = 1024 + v419;
        const uint8_t* v421 = v19 + v420;
        const uint8_t* v422 = (const uint8_t*) v421;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v423 = __riscv_vle8_v_u8mf2(v422, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v424 = __riscv_vand_vx_u8mf2(v423, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v425 = __riscv_vreinterpret_v_u8mf2_i8mf2(v424);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v426 = __riscv_vsrl_vx_u8mf2(v423, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v427 = __riscv_vreinterpret_v_u8mf2_i8mf2(v426);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v428 = 84 + v418;
        size_t v429 = 116 + v418;
        const uint8_t* v430 = v21 + v428;
        const int8_t* v431 = (const int8_t*) v430;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v432 = *(const int8_t *)(v431);
        const uint8_t* v433 = v21 + v429;
        const int8_t* v434 = (const int8_t*) v433;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v435 = *(const int8_t *)(v434);
        vint16m1_t v436 = v410;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v437 = __riscv_vwmacc_vx_i16m1(v436, v432, v425, 8);
        v410 = v437;
        vint16m1_t v438 = v412;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v439 = __riscv_vwmacc_vx_i16m1(v438, v435, v427, 8);
        v412 = v439;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v440 = 1032 + v419;
        const uint8_t* v441 = v19 + v440;
        const uint8_t* v442 = (const uint8_t*) v441;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v443 = __riscv_vle8_v_u8mf2(v442, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v444 = __riscv_vand_vx_u8mf2(v443, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v445 = __riscv_vreinterpret_v_u8mf2_i8mf2(v444);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v446 = __riscv_vsrl_vx_u8mf2(v443, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v447 = __riscv_vreinterpret_v_u8mf2_i8mf2(v446);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v448 = 84 + v418;
        size_t v449 = 116 + v418;
        const uint8_t* v450 = v21 + v448;
        const int8_t* v451 = (const int8_t*) v450;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v452 = *(const int8_t *)(v451);
        const uint8_t* v453 = v21 + v449;
        const int8_t* v454 = (const int8_t*) v453;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v455 = *(const int8_t *)(v454);
        vint16m1_t v456 = v414;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v457 = __riscv_vwmacc_vx_i16m1(v456, v452, v445, 8);
        v414 = v457;
        vint16m1_t v458 = v416;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v459 = __riscv_vwmacc_vx_i16m1(v458, v455, v447, 8);
        v416 = v459;
      }
      vint16m1_t v460 = v410;
      vint16m1_t v461 = v412;
      vint16m1_t v462 = v414;
      vint16m1_t v463 = v416;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v464 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v465 = __riscv_vwmacc_vv_i32m2(v464, v93, v460, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v466 = __riscv_vwmacc_vv_i32m2(v465, v111, v461, 8);
      v34 = v466;
      vint32m2_t v467 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v468 = __riscv_vwmacc_vv_i32m2(v467, v165, v462, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v469 = __riscv_vwmacc_vv_i32m2(v468, v183, v463, 8);
      v38 = v469;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
      const uint8_t* v470 = v19 + 128;
      const uint8_t* v471 = (const uint8_t*) v470;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v472 = __riscv_vle8_v_u8mf2(v471, 8);
      const uint8_t* v473 = v19 + 192;
      const uint8_t* v474 = (const uint8_t*) v473;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v475 = __riscv_vle8_v_u8mf2(v474, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v476 = __riscv_vand_vx_u8mf2(v472, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v477 = __riscv_vsrl_vx_u8mf2(v472, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v478 = __riscv_vand_vx_u8mf2(v475, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v479 = __riscv_vand_vx_u8mf2(v475, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v480 = __riscv_vsrl_vx_u8mf2(v479, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v481 = __riscv_vor_vv_u8mf2(v478, v476, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v482 = __riscv_vor_vv_u8mf2(v480, v477, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v483 = __riscv_vzext_vf2_u16m1(v481, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v484 = __riscv_vreinterpret_v_u16m1_i16m1(v483);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v485 = __riscv_vzext_vf2_u16m1(v482, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v486 = __riscv_vreinterpret_v_u16m1_i16m1(v485);
      const uint8_t* v487 = v19 + 144;
      const uint8_t* v488 = (const uint8_t*) v487;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v489 = __riscv_vle8_v_u8mf2(v488, 8);
      const uint8_t* v490 = v19 + 208;
      const uint8_t* v491 = (const uint8_t*) v490;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v492 = __riscv_vle8_v_u8mf2(v491, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v493 = __riscv_vand_vx_u8mf2(v489, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v494 = __riscv_vsrl_vx_u8mf2(v489, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v495 = __riscv_vand_vx_u8mf2(v492, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v496 = __riscv_vand_vx_u8mf2(v492, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v497 = __riscv_vsrl_vx_u8mf2(v496, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v498 = __riscv_vor_vv_u8mf2(v495, v493, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v499 = __riscv_vor_vv_u8mf2(v497, v494, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v500 = __riscv_vzext_vf2_u16m1(v498, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v501 = __riscv_vreinterpret_v_u16m1_i16m1(v500);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v502 = __riscv_vzext_vf2_u16m1(v499, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v503 = __riscv_vreinterpret_v_u16m1_i16m1(v502);
      const uint8_t* v504 = v19 + 160;
      const uint8_t* v505 = (const uint8_t*) v504;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v506 = __riscv_vle8_v_u8mf2(v505, 8);
      const uint8_t* v507 = v19 + 224;
      const uint8_t* v508 = (const uint8_t*) v507;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v509 = __riscv_vle8_v_u8mf2(v508, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v510 = __riscv_vand_vx_u8mf2(v506, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v511 = __riscv_vsrl_vx_u8mf2(v506, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v512 = __riscv_vand_vx_u8mf2(v509, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v513 = __riscv_vand_vx_u8mf2(v509, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v514 = __riscv_vsrl_vx_u8mf2(v513, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v515 = __riscv_vor_vv_u8mf2(v512, v510, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v516 = __riscv_vor_vv_u8mf2(v514, v511, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v517 = __riscv_vzext_vf2_u16m1(v515, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v518 = __riscv_vreinterpret_v_u16m1_i16m1(v517);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v519 = __riscv_vzext_vf2_u16m1(v516, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v520 = __riscv_vreinterpret_v_u16m1_i16m1(v519);
      const uint8_t* v521 = v19 + 176;
      const uint8_t* v522 = (const uint8_t*) v521;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v523 = __riscv_vle8_v_u8mf2(v522, 8);
      const uint8_t* v524 = v19 + 240;
      const uint8_t* v525 = (const uint8_t*) v524;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v526 = __riscv_vle8_v_u8mf2(v525, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v527 = __riscv_vand_vx_u8mf2(v523, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v528 = __riscv_vsrl_vx_u8mf2(v523, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v529 = __riscv_vand_vx_u8mf2(v526, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v530 = __riscv_vand_vx_u8mf2(v526, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v531 = __riscv_vsrl_vx_u8mf2(v530, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v532 = __riscv_vor_vv_u8mf2(v529, v527, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v533 = __riscv_vor_vv_u8mf2(v531, v528, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v534 = __riscv_vzext_vf2_u16m1(v532, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v535 = __riscv_vreinterpret_v_u16m1_i16m1(v534);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v536 = __riscv_vzext_vf2_u16m1(v533, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v537 = __riscv_vreinterpret_v_u16m1_i16m1(v536);
      const uint8_t* v538 = v19 + 136;
      const uint8_t* v539 = (const uint8_t*) v538;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v540 = __riscv_vle8_v_u8mf2(v539, 8);
      const uint8_t* v541 = v19 + 200;
      const uint8_t* v542 = (const uint8_t*) v541;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v543 = __riscv_vle8_v_u8mf2(v542, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v544 = __riscv_vand_vx_u8mf2(v540, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v545 = __riscv_vsrl_vx_u8mf2(v540, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v546 = __riscv_vand_vx_u8mf2(v543, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v547 = __riscv_vand_vx_u8mf2(v543, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v548 = __riscv_vsrl_vx_u8mf2(v547, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v549 = __riscv_vor_vv_u8mf2(v546, v544, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v550 = __riscv_vor_vv_u8mf2(v548, v545, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v551 = __riscv_vzext_vf2_u16m1(v549, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v552 = __riscv_vreinterpret_v_u16m1_i16m1(v551);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v553 = __riscv_vzext_vf2_u16m1(v550, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v554 = __riscv_vreinterpret_v_u16m1_i16m1(v553);
      const uint8_t* v555 = v19 + 152;
      const uint8_t* v556 = (const uint8_t*) v555;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v557 = __riscv_vle8_v_u8mf2(v556, 8);
      const uint8_t* v558 = v19 + 216;
      const uint8_t* v559 = (const uint8_t*) v558;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v560 = __riscv_vle8_v_u8mf2(v559, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v561 = __riscv_vand_vx_u8mf2(v557, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v562 = __riscv_vsrl_vx_u8mf2(v557, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v563 = __riscv_vand_vx_u8mf2(v560, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v564 = __riscv_vand_vx_u8mf2(v560, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v565 = __riscv_vsrl_vx_u8mf2(v564, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v566 = __riscv_vor_vv_u8mf2(v563, v561, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v567 = __riscv_vor_vv_u8mf2(v565, v562, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v568 = __riscv_vzext_vf2_u16m1(v566, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v569 = __riscv_vreinterpret_v_u16m1_i16m1(v568);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v570 = __riscv_vzext_vf2_u16m1(v567, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v571 = __riscv_vreinterpret_v_u16m1_i16m1(v570);
      const uint8_t* v572 = v19 + 168;
      const uint8_t* v573 = (const uint8_t*) v572;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v574 = __riscv_vle8_v_u8mf2(v573, 8);
      const uint8_t* v575 = v19 + 232;
      const uint8_t* v576 = (const uint8_t*) v575;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v577 = __riscv_vle8_v_u8mf2(v576, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v578 = __riscv_vand_vx_u8mf2(v574, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v579 = __riscv_vsrl_vx_u8mf2(v574, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v580 = __riscv_vand_vx_u8mf2(v577, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v581 = __riscv_vand_vx_u8mf2(v577, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v582 = __riscv_vsrl_vx_u8mf2(v581, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v583 = __riscv_vor_vv_u8mf2(v580, v578, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v584 = __riscv_vor_vv_u8mf2(v582, v579, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v585 = __riscv_vzext_vf2_u16m1(v583, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v586 = __riscv_vreinterpret_v_u16m1_i16m1(v585);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v587 = __riscv_vzext_vf2_u16m1(v584, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v588 = __riscv_vreinterpret_v_u16m1_i16m1(v587);
      const uint8_t* v589 = v19 + 184;
      const uint8_t* v590 = (const uint8_t*) v589;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v591 = __riscv_vle8_v_u8mf2(v590, 8);
      const uint8_t* v592 = v19 + 248;
      const uint8_t* v593 = (const uint8_t*) v592;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v594 = __riscv_vle8_v_u8mf2(v593, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v595 = __riscv_vand_vx_u8mf2(v591, 0x0F, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v596 = __riscv_vsrl_vx_u8mf2(v591, 4, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v597 = __riscv_vand_vx_u8mf2(v594, 0x30, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v598 = __riscv_vand_vx_u8mf2(v594, 0xC0, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v599 = __riscv_vsrl_vx_u8mf2(v598, 2, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v600 = __riscv_vor_vv_u8mf2(v597, v595, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v601 = __riscv_vor_vv_u8mf2(v599, v596, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v602 = __riscv_vzext_vf2_u16m1(v600, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v603 = __riscv_vreinterpret_v_u16m1_i16m1(v602);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v604 = __riscv_vzext_vf2_u16m1(v601, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v605 = __riscv_vreinterpret_v_u16m1_i16m1(v604);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_bsums_fold
      const uint8_t* v606 = v21 + 276;
      const int16_t* v607 = (const int16_t*) v606;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v608 = *(const int16_t *)(v607);
      const uint8_t* v609 = v21 + 278;
      const int16_t* v610 = (const int16_t*) v609;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v611 = *(const int16_t *)(v610);
      int32_t v612 = v608 + v611;
      vint32m2_t v613 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v614 = __riscv_vwmacc_vx_i32m2(v613, v612, v486, 8);
      v36 = v614;
      vint32m2_t v615 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v616 = __riscv_vwmacc_vx_i32m2(v615, v612, v554, 8);
      v40 = v616;
      const uint8_t* v617 = v21 + 280;
      const int16_t* v618 = (const int16_t*) v617;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v619 = *(const int16_t *)(v618);
      const uint8_t* v620 = v21 + 282;
      const int16_t* v621 = (const int16_t*) v620;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v622 = *(const int16_t *)(v621);
      int32_t v623 = v619 + v622;
      vint32m2_t v624 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v625 = __riscv_vwmacc_vx_i32m2(v624, v623, v503, 8);
      v36 = v625;
      vint32m2_t v626 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v627 = __riscv_vwmacc_vx_i32m2(v626, v623, v571, 8);
      v40 = v627;
      const uint8_t* v628 = v21 + 284;
      const int16_t* v629 = (const int16_t*) v628;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v630 = *(const int16_t *)(v629);
      const uint8_t* v631 = v21 + 286;
      const int16_t* v632 = (const int16_t*) v631;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v633 = *(const int16_t *)(v632);
      int32_t v634 = v630 + v633;
      vint32m2_t v635 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v636 = __riscv_vwmacc_vx_i32m2(v635, v634, v520, 8);
      v36 = v636;
      vint32m2_t v637 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v638 = __riscv_vwmacc_vx_i32m2(v637, v634, v588, 8);
      v40 = v638;
      const uint8_t* v639 = v21 + 288;
      const int16_t* v640 = (const int16_t*) v639;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v641 = *(const int16_t *)(v640);
      const uint8_t* v642 = v21 + 290;
      const int16_t* v643 = (const int16_t*) v642;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v644 = *(const int16_t *)(v643);
      int32_t v645 = v641 + v644;
      vint32m2_t v646 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v647 = __riscv_vwmacc_vx_i32m2(v646, v645, v537, 8);
      v36 = v647;
      vint32m2_t v648 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v649 = __riscv_vwmacc_vx_i32m2(v648, v645, v605, 8);
      v40 = v649;
      vint16m1_t v650;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v651 = __riscv_vmv_v_x_i16m1(0, 8);
      v650 = v651;
      vint16m1_t v652;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v653 = __riscv_vmv_v_x_i16m1(0, 8);
      v652 = v653;
      vint16m1_t v654;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v655 = __riscv_vmv_v_x_i16m1(0, 8);
      v654 = v655;
      vint16m1_t v656;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v657 = __riscv_vmv_v_x_i16m1(0, 8);
      v656 = v657;
      for (size_t v658 = 0; v658 < 16; v658 += 1) {
        size_t v659 = v658 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v660 = 1280 + v659;
        const uint8_t* v661 = v19 + v660;
        const uint8_t* v662 = (const uint8_t*) v661;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v663 = __riscv_vle8_v_u8mf2(v662, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v664 = __riscv_vand_vx_u8mf2(v663, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v665 = __riscv_vreinterpret_v_u8mf2_i8mf2(v664);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v666 = __riscv_vsrl_vx_u8mf2(v663, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v667 = __riscv_vreinterpret_v_u8mf2_i8mf2(v666);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v668 = 132 + v658;
        size_t v669 = 164 + v658;
        const uint8_t* v670 = v21 + v668;
        const int8_t* v671 = (const int8_t*) v670;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v672 = *(const int8_t *)(v671);
        const uint8_t* v673 = v21 + v669;
        const int8_t* v674 = (const int8_t*) v673;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v675 = *(const int8_t *)(v674);
        vint16m1_t v676 = v650;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v677 = __riscv_vwmacc_vx_i16m1(v676, v672, v665, 8);
        v650 = v677;
        vint16m1_t v678 = v652;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v679 = __riscv_vwmacc_vx_i16m1(v678, v675, v667, 8);
        v652 = v679;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v680 = 1288 + v659;
        const uint8_t* v681 = v19 + v680;
        const uint8_t* v682 = (const uint8_t*) v681;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v683 = __riscv_vle8_v_u8mf2(v682, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v684 = __riscv_vand_vx_u8mf2(v683, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v685 = __riscv_vreinterpret_v_u8mf2_i8mf2(v684);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v686 = __riscv_vsrl_vx_u8mf2(v683, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v687 = __riscv_vreinterpret_v_u8mf2_i8mf2(v686);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v688 = 132 + v658;
        size_t v689 = 164 + v658;
        const uint8_t* v690 = v21 + v688;
        const int8_t* v691 = (const int8_t*) v690;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v692 = *(const int8_t *)(v691);
        const uint8_t* v693 = v21 + v689;
        const int8_t* v694 = (const int8_t*) v693;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v695 = *(const int8_t *)(v694);
        vint16m1_t v696 = v654;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v697 = __riscv_vwmacc_vx_i16m1(v696, v692, v685, 8);
        v654 = v697;
        vint16m1_t v698 = v656;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v699 = __riscv_vwmacc_vx_i16m1(v698, v695, v687, 8);
        v656 = v699;
      }
      vint16m1_t v700 = v650;
      vint16m1_t v701 = v652;
      vint16m1_t v702 = v654;
      vint16m1_t v703 = v656;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v704 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v705 = __riscv_vwmacc_vv_i32m2(v704, v484, v700, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v706 = __riscv_vwmacc_vv_i32m2(v705, v501, v701, 8);
      v34 = v706;
      vint32m2_t v707 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v708 = __riscv_vwmacc_vv_i32m2(v707, v552, v702, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v709 = __riscv_vwmacc_vv_i32m2(v708, v569, v703, 8);
      v38 = v709;
      vint16m1_t v710;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v711 = __riscv_vmv_v_x_i16m1(0, 8);
      v710 = v711;
      vint16m1_t v712;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v713 = __riscv_vmv_v_x_i16m1(0, 8);
      v712 = v713;
      vint16m1_t v714;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v715 = __riscv_vmv_v_x_i16m1(0, 8);
      v714 = v715;
      vint16m1_t v716;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v717 = __riscv_vmv_v_x_i16m1(0, 8);
      v716 = v717;
      for (size_t v718 = 0; v718 < 16; v718 += 1) {
        size_t v719 = v718 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v720 = 1536 + v719;
        const uint8_t* v721 = v19 + v720;
        const uint8_t* v722 = (const uint8_t*) v721;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v723 = __riscv_vle8_v_u8mf2(v722, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v724 = __riscv_vand_vx_u8mf2(v723, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v725 = __riscv_vreinterpret_v_u8mf2_i8mf2(v724);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v726 = __riscv_vsrl_vx_u8mf2(v723, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v727 = __riscv_vreinterpret_v_u8mf2_i8mf2(v726);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v728 = 148 + v718;
        size_t v729 = 180 + v718;
        const uint8_t* v730 = v21 + v728;
        const int8_t* v731 = (const int8_t*) v730;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v732 = *(const int8_t *)(v731);
        const uint8_t* v733 = v21 + v729;
        const int8_t* v734 = (const int8_t*) v733;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v735 = *(const int8_t *)(v734);
        vint16m1_t v736 = v710;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v737 = __riscv_vwmacc_vx_i16m1(v736, v732, v725, 8);
        v710 = v737;
        vint16m1_t v738 = v712;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v739 = __riscv_vwmacc_vx_i16m1(v738, v735, v727, 8);
        v712 = v739;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v740 = 1544 + v719;
        const uint8_t* v741 = v19 + v740;
        const uint8_t* v742 = (const uint8_t*) v741;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v743 = __riscv_vle8_v_u8mf2(v742, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v744 = __riscv_vand_vx_u8mf2(v743, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v745 = __riscv_vreinterpret_v_u8mf2_i8mf2(v744);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v746 = __riscv_vsrl_vx_u8mf2(v743, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v747 = __riscv_vreinterpret_v_u8mf2_i8mf2(v746);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v748 = 148 + v718;
        size_t v749 = 180 + v718;
        const uint8_t* v750 = v21 + v748;
        const int8_t* v751 = (const int8_t*) v750;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v752 = *(const int8_t *)(v751);
        const uint8_t* v753 = v21 + v749;
        const int8_t* v754 = (const int8_t*) v753;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v755 = *(const int8_t *)(v754);
        vint16m1_t v756 = v714;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v757 = __riscv_vwmacc_vx_i16m1(v756, v752, v745, 8);
        v714 = v757;
        vint16m1_t v758 = v716;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v759 = __riscv_vwmacc_vx_i16m1(v758, v755, v747, 8);
        v716 = v759;
      }
      vint16m1_t v760 = v710;
      vint16m1_t v761 = v712;
      vint16m1_t v762 = v714;
      vint16m1_t v763 = v716;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v764 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v765 = __riscv_vwmacc_vv_i32m2(v764, v484, v760, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v766 = __riscv_vwmacc_vv_i32m2(v765, v501, v761, 8);
      v34 = v766;
      vint32m2_t v767 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v768 = __riscv_vwmacc_vv_i32m2(v767, v552, v762, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v769 = __riscv_vwmacc_vv_i32m2(v768, v569, v763, 8);
      v38 = v769;
      vint16m1_t v770;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v771 = __riscv_vmv_v_x_i16m1(0, 8);
      v770 = v771;
      vint16m1_t v772;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v773 = __riscv_vmv_v_x_i16m1(0, 8);
      v772 = v773;
      vint16m1_t v774;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v775 = __riscv_vmv_v_x_i16m1(0, 8);
      v774 = v775;
      vint16m1_t v776;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v777 = __riscv_vmv_v_x_i16m1(0, 8);
      v776 = v777;
      for (size_t v778 = 0; v778 < 16; v778 += 1) {
        size_t v779 = v778 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v780 = 1792 + v779;
        const uint8_t* v781 = v19 + v780;
        const uint8_t* v782 = (const uint8_t*) v781;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v783 = __riscv_vle8_v_u8mf2(v782, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v784 = __riscv_vand_vx_u8mf2(v783, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v785 = __riscv_vreinterpret_v_u8mf2_i8mf2(v784);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v786 = __riscv_vsrl_vx_u8mf2(v783, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v787 = __riscv_vreinterpret_v_u8mf2_i8mf2(v786);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v788 = 196 + v778;
        size_t v789 = 228 + v778;
        const uint8_t* v790 = v21 + v788;
        const int8_t* v791 = (const int8_t*) v790;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v792 = *(const int8_t *)(v791);
        const uint8_t* v793 = v21 + v789;
        const int8_t* v794 = (const int8_t*) v793;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v795 = *(const int8_t *)(v794);
        vint16m1_t v796 = v770;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v797 = __riscv_vwmacc_vx_i16m1(v796, v792, v785, 8);
        v770 = v797;
        vint16m1_t v798 = v772;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v799 = __riscv_vwmacc_vx_i16m1(v798, v795, v787, 8);
        v772 = v799;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v800 = 1800 + v779;
        const uint8_t* v801 = v19 + v800;
        const uint8_t* v802 = (const uint8_t*) v801;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v803 = __riscv_vle8_v_u8mf2(v802, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v804 = __riscv_vand_vx_u8mf2(v803, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v805 = __riscv_vreinterpret_v_u8mf2_i8mf2(v804);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v806 = __riscv_vsrl_vx_u8mf2(v803, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v807 = __riscv_vreinterpret_v_u8mf2_i8mf2(v806);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v808 = 196 + v778;
        size_t v809 = 228 + v778;
        const uint8_t* v810 = v21 + v808;
        const int8_t* v811 = (const int8_t*) v810;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v812 = *(const int8_t *)(v811);
        const uint8_t* v813 = v21 + v809;
        const int8_t* v814 = (const int8_t*) v813;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v815 = *(const int8_t *)(v814);
        vint16m1_t v816 = v774;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v817 = __riscv_vwmacc_vx_i16m1(v816, v812, v805, 8);
        v774 = v817;
        vint16m1_t v818 = v776;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v819 = __riscv_vwmacc_vx_i16m1(v818, v815, v807, 8);
        v776 = v819;
      }
      vint16m1_t v820 = v770;
      vint16m1_t v821 = v772;
      vint16m1_t v822 = v774;
      vint16m1_t v823 = v776;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v824 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v825 = __riscv_vwmacc_vv_i32m2(v824, v518, v820, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v826 = __riscv_vwmacc_vv_i32m2(v825, v535, v821, 8);
      v34 = v826;
      vint32m2_t v827 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v828 = __riscv_vwmacc_vv_i32m2(v827, v586, v822, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v829 = __riscv_vwmacc_vv_i32m2(v828, v603, v823, 8);
      v38 = v829;
      vint16m1_t v830;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v831 = __riscv_vmv_v_x_i16m1(0, 8);
      v830 = v831;
      vint16m1_t v832;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v833 = __riscv_vmv_v_x_i16m1(0, 8);
      v832 = v833;
      vint16m1_t v834;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v835 = __riscv_vmv_v_x_i16m1(0, 8);
      v834 = v835;
      vint16m1_t v836;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v837 = __riscv_vmv_v_x_i16m1(0, 8);
      v836 = v837;
      for (size_t v838 = 0; v838 < 16; v838 += 1) {
        size_t v839 = v838 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v840 = 2048 + v839;
        const uint8_t* v841 = v19 + v840;
        const uint8_t* v842 = (const uint8_t*) v841;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v843 = __riscv_vle8_v_u8mf2(v842, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v844 = __riscv_vand_vx_u8mf2(v843, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v845 = __riscv_vreinterpret_v_u8mf2_i8mf2(v844);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v846 = __riscv_vsrl_vx_u8mf2(v843, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v847 = __riscv_vreinterpret_v_u8mf2_i8mf2(v846);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v848 = 212 + v838;
        size_t v849 = 244 + v838;
        const uint8_t* v850 = v21 + v848;
        const int8_t* v851 = (const int8_t*) v850;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v852 = *(const int8_t *)(v851);
        const uint8_t* v853 = v21 + v849;
        const int8_t* v854 = (const int8_t*) v853;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v855 = *(const int8_t *)(v854);
        vint16m1_t v856 = v830;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v857 = __riscv_vwmacc_vx_i16m1(v856, v852, v845, 8);
        v830 = v857;
        vint16m1_t v858 = v832;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v859 = __riscv_vwmacc_vx_i16m1(v858, v855, v847, 8);
        v832 = v859;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
        size_t v860 = 2056 + v839;
        const uint8_t* v861 = v19 + v860;
        const uint8_t* v862 = (const uint8_t*) v861;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
        vuint8mf2_t v863 = __riscv_vle8_v_u8mf2(v862, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
        vuint8mf2_t v864 = __riscv_vand_vx_u8mf2(v863, 0x0F, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v865 = __riscv_vreinterpret_v_u8mf2_i8mf2(v864);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
        vuint8mf2_t v866 = __riscv_vsrl_vx_u8mf2(v863, 4, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
        vint8mf2_t v867 = __riscv_vreinterpret_v_u8mf2_i8mf2(v866);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
        size_t v868 = 212 + v838;
        size_t v869 = 244 + v838;
        const uint8_t* v870 = v21 + v868;
        const int8_t* v871 = (const int8_t*) v870;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v872 = *(const int8_t *)(v871);
        const uint8_t* v873 = v21 + v869;
        const int8_t* v874 = (const int8_t*) v873;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v875 = *(const int8_t *)(v874);
        vint16m1_t v876 = v834;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v877 = __riscv_vwmacc_vx_i16m1(v876, v872, v865, 8);
        v834 = v877;
        vint16m1_t v878 = v836;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
        vint16m1_t v879 = __riscv_vwmacc_vx_i16m1(v878, v875, v867, 8);
        v836 = v879;
      }
      vint16m1_t v880 = v830;
      vint16m1_t v881 = v832;
      vint16m1_t v882 = v834;
      vint16m1_t v883 = v836;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v884 = v34;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v885 = __riscv_vwmacc_vv_i32m2(v884, v518, v880, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v886 = __riscv_vwmacc_vv_i32m2(v885, v535, v881, 8);
      v34 = v886;
      vint32m2_t v887 = v38;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v888 = __riscv_vwmacc_vv_i32m2(v887, v586, v882, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v889 = __riscv_vwmacc_vv_i32m2(v888, v603, v883, 8);
      v38 = v889;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v890 = (const _Float16*) v19;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v891 = __riscv_vle16_v_f16m1(v890, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v892 = __riscv_vfwcvt_f_f_v_f32m2(v891, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v893 = __riscv_vfmul_vf_f32m2(v892, v23, 8);
      vint32m2_t v894 = v34;
      vfloat32m2_t v895 = v13;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v896 = __riscv_vfcvt_f_x_v_f32m2(v894, 8);
      vfloat32m2_t v897 = __riscv_vfmacc_vv_f32m2(v895, v896, v893, 8);
      vint32m2_t v898 = v36;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v899 = __riscv_vfcvt_f_x_v_f32m2(v898, 8);
      vfloat32m2_t v900 = __riscv_vfnmsac_vv_f32m2(v897, v28, v899, 8);
      v13 = v900;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v901 = v19 + 16;
      const _Float16* v902 = (const _Float16*) v901;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v903 = __riscv_vle16_v_f16m1(v902, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v904 = __riscv_vfwcvt_f_f_v_f32m2(v903, 8);
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v905 = __riscv_vfmul_vf_f32m2(v904, v23, 8);
      vint32m2_t v906 = v38;
      vfloat32m2_t v907 = v15;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v908 = __riscv_vfcvt_f_x_v_f32m2(v906, 8);
      vfloat32m2_t v909 = __riscv_vfmacc_vv_f32m2(v907, v908, v905, 8);
      vint32m2_t v910 = v40;
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
      // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v911 = __riscv_vfcvt_f_x_v_f32m2(v910, 8);
      vfloat32m2_t v912 = __riscv_vfnmsac_vv_f32m2(v909, v33, v911, 8);
      v15 = v912;
    }
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v913 = v9 * 16;
    float* v914 = v2 + v913;
    vfloat32m2_t v915 = v13;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v914, v915, 8);
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
    size_t v916 = v9 * 16;
    size_t v917 = v916 + 8;
    float* v918 = v2 + v917;
    vfloat32m2_t v919 = v15;
    // weft_emitc.source_op=weft_rvv.typed_repack_gemv_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v918, v919, 8);
  }
  return;
}


